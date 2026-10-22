# ============================================================================
# ADVANCED RAG PIPELINE WITH PINECONE & RERANKING
# Implements hybrid retrieval with BGE reranking (exactly as provided code)
# ============================================================================

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZATION - PINECONE CONNECTION
# ============================================================================

class PineconeConnector:
    """Handles Pinecone connection and management"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pinecone connector
        
        Args:
            api_key: Pinecone API key (uses env var if not provided)
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.index = None
        self.pc = None
        self.index_name = "financial-knowledge"
        
        self._connect()
    
    def _connect(self):
        """Connect to Pinecone index"""
        try:
            if not Pinecone:
                logger.error("Pinecone not installed. Install: pip install pinecone-client")
                return
            
            logger.info("Connecting to Pinecone...")
            
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            
            logger.info(f"✅ Connected to Pinecone index '{self.index_name}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Pinecone index: {e}")
            self.index = None
    
    def is_connected(self) -> bool:
        """Check if connected to Pinecone"""
        return self.index is not None


# ============================================================================
# EMBEDDING MODEL INITIALIZATION
# ============================================================================

def init_embedding_model(device: str = "cpu") -> Optional[SentenceTransformer]:
    """
    Initialize FinBERT embedding model
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        SentenceTransformer model or None
    """
    try:
        logger.info("Loading FinBERT embedding model...")
        embed_model = SentenceTransformer(
            "FinLang/finance-embeddings-investopedia",
            device=device
        )
        logger.info("✅ Embedding model loaded")
        return embed_model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


# ============================================================================
# BGE RERANKER INITIALIZATION
# ============================================================================

def init_reranker(device: str = "cpu") -> Optional[FlagReranker]:
    """
    Initialize BGE Reranker with fallback settings
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        FlagReranker model or None
    """
    try:
        logger.info("Loading BGE Reranker...")
        
        # Check if we can use FP16
        use_fp16 = (
            device == "cuda" and 
            torch.cuda.is_available() and 
            torch.cuda.get_device_capability(0)[0] >= 7
        )
        
        reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=use_fp16,
            device=device
        )
        logger.info(f"✅ BGE Reranker loaded (FP16: {use_fp16})")
        return reranker
        
    except Exception as e:
        logger.warning(f"BGE Reranker with FP16 failed: {e}")
        
        try:
            logger.info("Trying BGE Reranker with FP32 fallback...")
            reranker = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=False,
                device=device
            )
            logger.info("✅ BGE Reranker loaded with FP32")
            return reranker
            
        except Exception as e2:
            logger.error(f"❌ BGE Reranker failed to load: {e2}")
            return None


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Handles text preprocessing for BM25"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 retrieval
        
        Args:
            text: Raw text to preprocess
        
        Returns:
            List of preprocessed tokens
        """
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and stem
            processed = [
                self.stemmer.stem(token)
                for token in tokens
                if token.isalnum() and token not in self.stop_words
            ]
            
            return processed
            
        except Exception as e:
            logger.warning(f"Text preprocessing error: {e}")
            return text.lower().split()


# ============================================================================
# SCORE NORMALIZATION
# ============================================================================

def min_max_scale(scores: List[float]) -> List[float]:
    """
    Min-max normalization for scores
    
    Args:
        scores: List of scores
    
    Returns:
        Normalized scores (0-1 range)
    """
    if not scores or len(scores) == 0:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)
    
    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


# ============================================================================
# HYBRID RETRIEVAL FUNCTION
# ============================================================================

def hybrid_retrieval(
    query: str,
    index,
    embed_model: SentenceTransformer,
    reranker: Optional[FlagReranker] = None,
    preprocessor: Optional[TextPreprocessor] = None,
    top_k: int = 3,
    alpha: float = 0.6
) -> List[Document]:
    """
    Hybrid retrieval combining vector search, BM25, and reranking
    
    Args:
        query: Search query
        index: Pinecone index
        embed_model: Embedding model
        reranker: Optional reranker model
        preprocessor: Text preprocessor
        top_k: Number of results to return
        alpha: Weight for vector search (1-alpha for BM25)
    
    Returns:
        List of retrieved documents
    """
    
    if not index:
        logger.warning("Pinecone index not available")
        return []
    
    if not embed_model:
        logger.warning("Embedding model not available")
        return []
    
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    try:
        # Step 1: Vector retrieval (get more than top_k for potential reranking)
        logger.info(f"Step 1: Vector retrieval for query: {query[:60]}...")
        
        initial_k = max(20, top_k * 7)  # Get 7x more for later filtering
        query_emb = embed_model.encode(query).tolist()
        
        vector_results = index.query(
            vector=query_emb,
            top_k=initial_k,
            include_metadata=True
        )
        
        # Build candidate pool
        candidates = {}
        for match in vector_results["matches"]:
            text = match.get("metadata", {}).get("text")
            if text and text not in candidates:
                candidates[text] = {
                    "text": text,
                    "vector_score": match["score"],
                    "source": match.get("metadata", {}).get("source", "N/A"),
                    "metadata": match.get("metadata", {}),
                    "id": match.get("id", text)
                }
        
        if not candidates:
            logger.warning("No candidates from vector retrieval")
            return []
        
        logger.info(f"✓ Vector retrieval returned {len(candidates)} candidates")
        
        # Step 2: BM25 Ranking
        logger.info("Step 2: BM25 ranking...")
        
        candidate_texts = [item["text"] for item in candidates.values()]
        corpus = [preprocessor.preprocess_text(doc) for doc in candidate_texts]
        
        try:
            bm25 = BM25Okapi(corpus)
            query_tokens = preprocessor.preprocess_text(query)
            bm25_scores = bm25.get_scores(query_tokens)
        except (ValueError, ZeroDivisionError):
            logger.warning("BM25 scoring failed, using alpha=1.0 (vector only)")
            alpha = 1.0
            bm25_scores = np.array([0.0] * len(candidates))
        
        # Step 3: Score Normalization
        logger.info("Step 3: Score normalization...")
        
        vector_scores = [item["vector_score"] for item in candidates.values()]
        norm_vector_scores = min_max_scale(vector_scores)
        norm_bm25_scores = min_max_scale(bm25_scores.tolist())
        
        # Step 4: Hybrid Score Calculation
        logger.info("Step 4: Hybrid score calculation...")
        
        candidates_list = list(candidates.values())
        for i, item in enumerate(candidates_list):
            item["vector_score_normalized"] = norm_vector_scores[i]
            item["bm25_score_normalized"] = norm_bm25_scores[i]
            item["hybrid_score"] = (
                (alpha * item["vector_score_normalized"]) +
                ((1 - alpha) * item["bm25_score_normalized"])
            )
        
        # Sort by hybrid score
        candidates_list = sorted(
            candidates_list,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        # Step 5: BGE Reranking (if available)
        if reranker and len(candidates_list) > 1:
            logger.info("Step 5: BGE reranking...")
            
            rerank_candidates = candidates_list[:20]  # Rerank top 20
            pairs = [[query, item["text"]] for item in rerank_candidates]
            
            try:
                rerank_scores = reranker.compute_score(pairs)
                for item, score in zip(rerank_candidates, rerank_scores):
                    item["rerank_score"] = float(score)
                
                # Sort by rerank score
                rerank_candidates = sorted(
                    rerank_candidates,
                    key=lambda x: x["rerank_score"],
                    reverse=True
                )
                
                reranked_texts = {item["text"] for item in rerank_candidates}
                remaining_candidates = [
                    item for item in candidates_list
                    if item["text"] not in reranked_texts
                ]
                
                final_list = rerank_candidates + remaining_candidates
                logger.info(f"✓ Reranking complete")
                
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using hybrid scores")
                final_list = candidates_list
        else:
            final_list = candidates_list
        
        # Step 6: Return top_k results
        logger.info(f"Step 6: Returning top {top_k} results...")
        
        final_results = final_list[:top_k]
        
        documents = [
            Document(
                page_content=item["text"],
                metadata={
                    "source": item.get("source", "N/A"),
                    "id": item.get("id", "N/A"),
                    "vector_score": item.get("vector_score_normalized", 0),
                    "bm25_score": item.get("bm25_score_normalized", 0),
                    "hybrid_score": item.get("hybrid_score", 0),
                    "rerank_score": item.get("rerank_score", None),
                    **item.get("metadata", {})
                }
            )
            for item in final_results
        ]
        
        logger.info(f"✅ Hybrid retrieval complete ({len(documents)} documents)")
        
        return documents
        
    except Exception as e:
        logger.error(f"Hybrid retrieval error: {e}")
        return []


# ============================================================================
# HYBRID RETRIEVER CLASS (LangChain compatible)
# ============================================================================

class HybridRetriever:
    """
    Hybrid retriever combining vector search, BM25, and reranking
    """
    
    def __init__(self,
                 pinecone_index,
                 embed_model: SentenceTransformer,
                 reranker: Optional[FlagReranker] = None,
                 top_k: int = 3,
                 alpha: float = 0.6,
                 device: str = "cpu"):
        """
        Initialize hybrid retriever
        
        Args:
            pinecone_index: Pinecone index object
            embed_model: Embedding model
            reranker: Optional reranker
            top_k: Number of results
            alpha: Vector weight
            device: Device to use
        """
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.reranker = reranker
        self.top_k = top_k
        self.alpha = alpha
        self.device = device
        self.preprocessor = TextPreprocessor()
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents for query
        
        Args:
            query: Search query
        
        Returns:
            List of documents
        """
        return hybrid_retrieval(
            query=query,
            index=self.pinecone_index,
            embed_model=self.embed_model,
            reranker=self.reranker,
            preprocessor=self.preprocessor,
            top_k=self.top_k,
            alpha=self.alpha
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        LangChain compatible method for retrieving documents
        
        Args:
            query: Search query
        
        Returns:
            List of documents
        """
        return self.retrieve(query)


__all__ = [
    'PineconeConnector',
    'init_embedding_model',
    'init_reranker',
    'TextPreprocessor',
    'min_max_scale',
    'hybrid_retrieval',
    'HybridRetriever'
]
