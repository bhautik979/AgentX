import os
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import tiktoken
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from numpy.linalg import norm
from FlagEmbedding import FlagReranker

# ----------------------------
# CONFIGURATION
# ----------------------------
INPUT_FILE = "/home/karan.padariya/LMA_Major_project/knowledge_base_cleaned_segmented.jsonl"
OUTPUT_EMB_FILE = "/ssd_scratch/karan.p /knowledge_base_embeddings.jsonl"

EMBED_MODEL = "FinLang/finance-embeddings-investopedia"  # Instead of all-mpnet-base-v2
VECTOR_DIM = 768
TOP_K = 5

PINECONE_API_KEY = "pcsk_6dXVFm_8Lom6Lcr77R14vYjin9ejjXWUMH2dFwhdoKM1zCPJMWNqYbjtnRTqGuyTUAb25a"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "financial-knowledge"

USE_BGE_RERANKER = True
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

embed_model = SentenceTransformer(EMBED_MODEL)

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

reranker = None
if USE_BGE_RERANKER:
    try:
        reranker = FlagReranker(BGE_RERANKER_MODEL, use_fp16=True)
        print(f"[INFO] BGE Reranker loaded")
    except Exception as e:
        print(f"[WARNING] Reranker not available: {e}")

# ----------------------------
# MODIFIED rerank_query FUNCTION
# ----------------------------
def rerank_query(query, top_k=TOP_K):
    """
    Enhanced retrieval with BGE reranking, duplicate removal, 
    and detailed output.
    """
    
    # --- MODIFICATION HERE ---
    # Get initial candidates (e.g., 7x to 10x more to handle duplicates)
    # This increases the chance of finding top_k *unique* documents.
    multiplier = 7 
    initial_k = top_k * multiplier
    # --- END MODIFICATION ---

    print(f"[DEBUG] Fetching {initial_k} initial candidates for reranking...")
    
    results = index.query(
        vector=embed_model.encode(query).tolist(),
        top_k=initial_k,
        include_metadata=True
    )
    
    candidates = []
    seen_texts = set()  # --- To handle duplicates ---
    
    for match in results["matches"]:
        # Use .get() for safe access
        text = match.get("metadata", {}).get("text")
        if not text:
            continue
            
        # --- Check for duplicates ---
        if text not in seen_texts:
            seen_texts.add(text)
            candidates.append({
                "text": text,
                "initial_score": match["score"],
                # --- Assumes metadata has a "source" key for chunk name ---
                # --- Adjust "source" if your metadata key is different ---
                "source": match.get("metadata", {}).get("source", "N/A"), 
                "id": match.get("id", "N/A")  # This is the vector's unique ID
            })

    print(f"[DEBUG] Found {len(candidates)} unique candidates after filtering.")

    # Apply BGE reranking if available and we have candidates
    if reranker and candidates:
        print(f"[DEBUG] Applying BGE Reranker...")
        pairs = [[query, item["text"]] for item in candidates]
        rerank_scores = reranker.compute_score(pairs)
        
        # Add rerank scores to candidates
        for item, score in zip(candidates, rerank_scores):
            item["rerank_score"] = score
            
        # Sort by new rerank scores
        ranked_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return the top_k results
        return ranked_results[:top_k]
    
    # Fallback to original similarity sort if no reranker
    # (Duplicates are already handled)
    print(f"[DEBUG] Reranker not used. Sorting by initial score.")
    ranked_results = sorted(candidates, key=lambda x: x["initial_score"], reverse=True)
    return ranked_results[:top_k]

# ----------------------------
# MODIFIED Example Query
# ----------------------------
query_text = "What is an emergency fund?"
top_results = rerank_query(query_text)

print("\nTop-K Reranked Chunks:\n")
if not top_results:
    print("No results found.")

for i, result in enumerate(top_results, 1):
    print(f"--- Result {i} ---")
    
    # Print chunk name (source) and ID
    print(f"SOURCE: {result.get('source', 'N/A')} (ID: {result.get('id', 'N/A')})")
    
    # Print the relevant score
    if "rerank_score" in result:
        print(f"RERANK SCORE: {result['rerank_score']:.4f}")
    else:
        print(f"INITIAL SCORE: {result['initial_score']:.4f}")
        
    # Print the text
    print(f"TEXT: {result['text']}\n")