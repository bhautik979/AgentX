import os
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import tiktoken
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from numpy.linalg import norm
import multiprocessing

# Fix for multiprocessing on non-Unix systems
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("[WARNING] FlagEmbedding not installed. Install with: pip install FlagEmbedding")

# ----------------------------
# CONFIGURATION
# ----------------------------
OUTPUT_EMB_FILE = "/ssd_scratch/karan.p/knowledge_base_embeddings_semantic_parent_child.jsonl"

EMBED_MODEL = "all-mpnet-base-v2"  # 768-dimensional embeddings
VECTOR_DIM = 768
TOP_K = 5

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_4Ndnwt_6wZ5exov2n7isQMJLnGtSFmFwidxmAuzc6nzKAAdabWV1zyScDTK59vV8krjF3F")
PINECONE_ENV = "us-east-1"
INDEX_NAME = "financial-knowledge"

# BGE Reranker configuration
USE_BGE_RERANKER = True
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

embed_model = SentenceTransformer(EMBED_MODEL, cache_folder="/ssd_scratch/karan.p")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if INDEX_NAME not in pc.list_indexes().names():
    print(f"[DEBUG] Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)

# ----------------------------
# LOAD BGE RERANKER (Global - Load Once)
# ----------------------------
def load_reranker():
    """Load reranker once globally to avoid repeated loading."""
    global reranker
    
    if reranker is not None:
        return reranker
    
    if USE_BGE_RERANKER and RERANKER_AVAILABLE:
        try:
            print(f"[INFO] Loading BGE Reranker: {BGE_RERANKER_MODEL}...")
            reranker = FlagReranker(BGE_RERANKER_MODEL, use_fp16=True)
            print(f"[INFO] BGE Reranker loaded successfully!")
            return reranker
        except Exception as e:
            print(f"[ERROR] Failed to load BGE Reranker: {e}")
            print("[INFO] Make sure you have:")
            print("  1. Installed FlagEmbedding: pip install FlagEmbedding")
            print("  2. Sufficient disk space for model download (~1GB)")
            print("  3. Internet connection for first-time model download")
            return None
    elif USE_BGE_RERANKER and not RERANKER_AVAILABLE:
        print("[ERROR] BGE Reranker requested but FlagEmbedding not installed!")
        print("Install with: pip install FlagEmbedding")
        return None
    
    return None

reranker = None  # Initialize as None


def rerank_query(query, top_k=TOP_K):
    """
    Enhanced retrieval with BGE reranking and duplicate removal.
    """
    
    # Fetch more initial candidates to handle duplicates
    multiplier = 10
    initial_k = top_k * multiplier

    print(f"\n[DEBUG] Fetching {initial_k} initial candidates...")
    
    query_vector = embed_model.encode(query).tolist()
    
    results = index.query(
        vector=query_vector,
        top_k=initial_k,
        include_metadata=True
    )
    
    candidates = []
    seen_texts = set()
    
    for match in results["matches"]:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "").strip()
        
        if not text:
            print(f"[WARNING] Empty text for ID: {match.get('id')}")
            continue
        
        # Remove duplicates
        if text not in seen_texts:
            seen_texts.add(text)
            candidates.append({
                "text": text,
                "initial_score": match["score"],
                "source": metadata.get("source", "unknown"),
                "doc_name": metadata.get("doc_name", "N/A"),
                "page": metadata.get("page", "N/A"),
                "id": match.get("id", "N/A")
            })

    print(f"[DEBUG] Found {len(candidates)} unique candidates after filtering.")

    # Apply BGE reranking
    if reranker and candidates:
        print(f"[DEBUG] Applying BGE Reranker on {len(candidates)} candidates...")
        
        # Prepare pairs for reranking: [[query, document], ...]
        pairs = [[query, item["text"]] for item in candidates]
        
        try:
            # Compute reranking scores with num_process=1 to avoid multiprocessing issues
            rerank_scores = reranker.compute_score(pairs, batch_size=32, num_process=1)
            
            # Add rerank scores to candidates
            for item, score in zip(candidates, rerank_scores):
                item["rerank_score"] = float(score)
            
            # Sort by rerank score (descending)
            ranked_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            
            print(f"[INFO] Reranking complete. Returning top {top_k} results.")
            return ranked_results[:top_k]
            
        except Exception as e:
            print(f"[ERROR] Reranking failed: {e}")
            print(f"[INFO] Falling back to initial similarity scores...")
            ranked_results = sorted(candidates, key=lambda x: x["initial_score"], reverse=True)
            return ranked_results[:top_k]
    
    # Fallback if reranker not available
    print(f"[DEBUG] BGE Reranker not available. Sorting by initial score...")
    ranked_results = sorted(candidates, key=lambda x: x["initial_score"], reverse=True)
    return ranked_results[:top_k]


# ----------------------------
# EXAMPLE QUERY
# ----------------------------
if __name__ == '__main__':
    # Load reranker once before queries
    print("[INFO] Initializing reranker...")
    load_reranker()
    
    query_text = "What is an emergency fund?"
    print(f"\n{'='*80}")
    print(f"QUERY: {query_text}")
    print(f"{'='*80}")

    top_results = rerank_query(query_text, top_k=TOP_K)

    print("\n" + "="*80)
    print("Top-K Results:")
    print("="*80 + "\n")

    if not top_results:
        print("No results found.")
    else:
        for i, result in enumerate(top_results, 1):
            print(f"--- Result {i} ---")
            
            if "rerank_score" in result:
                print(f"Rerank Score: {result['rerank_score']:.4f}")
            else:
                print(f"Initial Score: {result['initial_score']:.4f}")
            
            print(f"Source: {result['source']}")
            print(f"Document: {result['doc_name']}")
            print(f"Page: {result['page']}")
            print(f"ID: {result['id']}")
            print(f"\nText:\n{result['text'][:500]}...")
            print()