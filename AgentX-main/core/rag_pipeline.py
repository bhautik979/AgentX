import os
import json
import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAGPipeline:
    """Enhanced RAG pipeline with hybrid retrieval (BM25 + Vector Search)"""
    
    def __init__(self, 
                 pinecone_api_key: str,
                 embed_model_name: str = "FinLang/finance-embeddings-investopedia",
                 index_name: str = "financial-knowledge-pc"):
        
        logger.info("🚀 Initializing Financial RAG Pipeline...")
        
        # Initialize Embeddings
        logger.info(f"📚 Loading embedding model: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embed_dim = 768
        
        # Initialize Pinecone
        logger.info(f"🔌 Connecting to Pinecone index: {index_name}")
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
            self.index_name = index_name
        except Exception as e:
            logger.error(f"❌ Failed to connect to Pinecone: {e}")
            raise
        
        # Initialize BM25
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        
        # Tokenizer
        self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
        
        logger.info("✅ RAG Pipeline initialized successfully")
    
    def load_documents_for_bm25(self, jsonl_file: str):
        """Load documents into memory for BM25 indexing"""
        logger.info(f"📖 Loading documents from {jsonl_file} for BM25...")
        
        documents = []
        doc_ids = []
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        record = json.loads(line)
                        documents.append(record.get('text', ''))
                        doc_ids.append(record.get('doc_id', f'doc_{idx}'))
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Skipping malformed JSON at line {idx}")
                        continue
        except FileNotFoundError:
            logger.error(f"❌ File not found: {jsonl_file}")
            return
        
        # Tokenize for BM25
        logger.info(f"🔄 Tokenizing {len(documents)} documents for BM25...")
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
        self.doc_ids = doc_ids
        
        logger.info(f"✅ BM25 index ready with {len(documents)} documents")
    
    def retrieve_dense(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve using vector similarity (Pinecone)"""
        try:
            query_embedding = self.embed_model.encode(query).tolist()
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            retrieved = []
            for match in results['matches']:
                retrieved.append({
                    'text': match['metadata'].get('retrieval_text', ''),
                    'score': match['score'],
                    'source': match['metadata'].get('source_file', 'Unknown'),
                    'type': 'dense'
                })
            
            logger.info(f"✅ Dense retrieval returned {len(retrieved)} results")
            return retrieved
            
        except Exception as e:
            logger.error(f"❌ Dense retrieval failed: {e}")
            return []
    
    def retrieve_sparse(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve using BM25 keyword search"""
        if self.bm25 is None:
            logger.warning("⚠️ BM25 not initialized. Call load_documents_for_bm25() first")
            return []
        
        try:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            retrieved = []
            for idx in top_indices:
                if scores[idx] > 0:
                    retrieved.append({
                        'text': self.documents[idx],
                        'score': scores[idx],
                        'doc_id': self.doc_ids[idx],
                        'type': 'sparse'
                    })
            
            logger.info(f"✅ Sparse retrieval returned {len(retrieved)} results")
            return retrieved
            
        except Exception as e:
            logger.error(f"❌ Sparse retrieval failed: {e}")
            return []
    
    def hybrid_retrieve(self, 
                       query: str, 
                       top_k: int = 5,
                       dense_weight: float = 0.6,
                       sparse_weight: float = 0.4) -> List[str]:
        """Combine dense and sparse retrieval for better results"""
        logger.info(f"🔄 Running hybrid retrieval for: '{query}'")
        
        # Get dense results
        dense_results = self.retrieve_dense(query, top_k=top_k*2)
        dense_dict = {}
        
        if dense_results:
            max_dense_score = max(r['score'] for r in dense_results)
            for result in dense_results:
                normalized_score = result['score'] / (max_dense_score + 1e-6)
                dense_dict[result['text']] = normalized_score * dense_weight
        
        # Get sparse results
        sparse_results = self.retrieve_sparse(query, top_k=top_k*2)
        sparse_dict = {}
        
        if sparse_results:
            max_sparse_score = max(r['score'] for r in sparse_results)
            for result in sparse_results:
                normalized_score = result['score'] / (max_sparse_score + 1e-6)
                key = result['text']
                if key in sparse_dict:
                    sparse_dict[key] += normalized_score * sparse_weight
                else:
                    sparse_dict[key] = normalized_score * sparse_weight
        
        # Combine scores
        combined_scores = {}
        for text, score in dense_dict.items():
            combined_scores[text] = score + sparse_dict.get(text, 0)
        for text, score in sparse_dict.items():
            if text not in combined_scores:
                combined_scores[text] = score
        
        # Sort and return top-k
        sorted_texts = sorted(combined_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        
        results = [text for text, _ in sorted_texts[:top_k]]
        logger.info(f"✅ Hybrid retrieval returned {len(results)} final results")
        
        return results
    
    def retrieve_with_reranking(self,
                               query: str,
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve and rerank using BGE reranker"""
        logger.info(f"🎯 Retrieving with reranking for: '{query}'")
        
        # Hybrid retrieval
        chunks = self.hybrid_retrieve(query, top_k=top_k*3)
        
        if len(chunks) == 0:
            logger.warning("⚠️ No chunks retrieved")
            return []
        
        # Try to rerank with BGE
        try:
            from FlagEmbedding import FlagReranker
            logger.info("🔄 Loading BGE Reranker...")
            reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
            
            pairs = [[query, chunk] for chunk in chunks]
            scores = reranker.compute_score(pairs, batch_size=32, num_process=1)
            
            ranked = sorted(zip(chunks, scores), 
                          key=lambda x: x[1], 
                          reverse=True)
            
            final_results = ranked[:top_k]
            logger.info(f"✅ Reranking complete. Returning {len(final_results)} results")
            
            return final_results
            
        except ImportError:
            logger.warning("⚠️ FlagEmbedding not installed. Returning hybrid results without reranking")
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}. Using hybrid results")
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
    
    def format_context(self, chunks: List[str], max_tokens: int = 2048) -> str:
        """Format retrieved chunks into context string for LLM"""
        context_parts = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks, 1):
            tokens = len(self.tokenizer.encode(chunk))
            if total_tokens + tokens > max_tokens:
                logger.info(f"⚠️ Reached max token limit. Using {i-1} chunks")
                break
            context_parts.append(f"[Source {i}]\n{chunk}")
            total_tokens += tokens
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"✅ Context formatted with {total_tokens} tokens")
        
        return context


if __name__ == "__main__":
    from config import PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL
    
    # Initialize
    rag = FinancialRAGPipeline(
        pinecone_api_key=PINECONE_API_KEY,
        embed_model_name=EMBED_MODEL,
        index_name=PINECONE_INDEX
    )
    
    # Test query
    test_query = "How to create an emergency fund?"
    logger.info(f"\n🧪 Testing with query: '{test_query}'")
    
    results = rag.retrieve_with_reranking(test_query, top_k=3)
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n[Result {i}] (Score: {score:.3f})")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)