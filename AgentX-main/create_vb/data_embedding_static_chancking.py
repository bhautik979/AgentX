import os
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import tiktoken
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from numpy.linalg import norm

# ----------------------------
# CONFIGURATION
# ----------------------------
INPUT_FILE = "/home2/sathya.marisetti/lma_major/knowledge_base_cleaned_segmented.jsonl"
OUTPUT_EMB_FILE = "/home2/sathya.marisetti/lma_major/knowledge_base_embeddings.jsonl"

EMBED_MODEL = "all-mpnet-base-v2"
VECTOR_DIM = 768
CHUNK_SIZES = [2048, 512, 128]
OVERLAP_TOKENS = 50
TOP_K = 5
UPSERT_BATCH_SIZE = 50

PINECONE_API_KEY = ""
PINECONE_ENV = "us-east-1"
INDEX_NAME = "financial-knowledge"

# ----------------------------
# INITIALIZE MODELS
# ----------------------------
print("[DEBUG] Initializing models...")
enc = tiktoken.encoding_for_model("text-embedding-3-small")
embed_model = SentenceTransformer(EMBED_MODEL)

# ----------------------------
# Initialize Pinecone (v5)
# ----------------------------
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
# HELPER FUNCTIONS
# ----------------------------
def tokenize_count(text):
    return len(enc.encode(text))

def split_with_overlap(sentences, max_tokens, overlap=OVERLAP_TOKENS):
    chunks = []
    current_chunk = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = tokenize_count(sent)
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_len = tokenize_count(s)
                if overlap_tokens + s_len > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_len
            current_chunk = overlap_sentences
            current_tokens = sum(tokenize_count(s) for s in current_chunk)
        current_chunk.append(sent)
        current_tokens += sent_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def segment_text_multi_granularity(text, chunk_sizes=CHUNK_SIZES):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    all_chunks = {}
    for size in chunk_sizes:
        chunks_for_size = []
        for para in paragraphs:
            if tokenize_count(para) < size:
                if size == max(chunk_sizes):
                    chunks_for_size.append(para)
                continue
            sentences = sent_tokenize(para)
            chunks = split_with_overlap(sentences, size, OVERLAP_TOKENS)
            chunks_for_size.extend(chunks)
        all_chunks[size] = chunks_for_size
    return all_chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ----------------------------
# PIPELINE: Chunk + Embed + Index (batched)
# ----------------------------
processed_chunks = []
upsert_batch = []

print("[DEBUG] Starting chunking, embedding, and indexing...")

with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    for line_num, line in enumerate(tqdm(infile, desc="Processing documents"), 1):
        try:
            record = json.loads(line)
            text = record.get("text", "")
            multi_chunks = segment_text_multi_granularity(text)

            for size, chunks in multi_chunks.items():
                for i, chunk in enumerate(chunks, 1):
                    doc_id = f"{record.get('doc_id','doc')}_size{size}_chunk{i}"
                    parent_doc_id = record.get('doc_id','doc')
                    embedding = embed_model.encode(chunk).tolist()

                    chunk_record = {
                        "doc_id": doc_id,
                        "parent_doc_id": parent_doc_id,
                        "source_file": record.get("source_file"),
                        "page_number": record.get("page_number"),
                        "text": chunk,
                        "token_count": tokenize_count(chunk),
                        "granularity": size,
                        "embedding": embedding,
                        "metadata": record.get("metadata", {})
                    }

                    processed_chunks.append(chunk_record)
                # Create a full metadata dictionary for Pinecone
                    metadata_for_pinecone = {
                        "parent_doc_id": parent_doc_id,
                        "text": chunk,  # <-- This is the text you want to retrieve
                        "source_file": record.get("source_file"),
                        "page_number": record.get("page_number"),
                        "granularity": size,
                        "token_count": tokenize_count(chunk)
                    }
                    upsert_batch.append((doc_id, embedding, metadata_for_pinecone))
                    
                    if len(upsert_batch) >= UPSERT_BATCH_SIZE:
                        index.upsert(vectors=upsert_batch)
                        upsert_batch = []

            if line_num % 10 == 0:
                print(f"[DEBUG] Processed {line_num} documents...")

        except Exception as e:
            print(f"[ERROR] Failed to process line {line_num}: {e}")

if upsert_batch:
    index.upsert(vectors=upsert_batch)

# Save embeddings to file
with open(OUTPUT_EMB_FILE, "w", encoding="utf-8") as f:
    for rec in tqdm(processed_chunks, desc="Saving embeddings"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Chunking, embedding, and indexing complete! {len(processed_chunks)} chunks processed.")

# ----------------------------
# Reranking Query Function (Option 1: cosine similarity)
# ----------------------------
def rerank_query(query, top_k=TOP_K):
    print(f"[DEBUG] Embedding query: {query}")
    query_emb = embed_model.encode(query)

    print("[DEBUG] Retrieving top-K candidates from Pinecone...")
    results = index.query(
        vector=query_emb.tolist(),
        top_k=top_k,
        include_metadata=True,
        include_values=True
    )

    candidates = []
    candidate_embs = []

    for match in results["matches"]:
        text = match["metadata"].get("text", "") or match["id"]
        emb = np.array(match["values"])
        candidates.append(text)
        candidate_embs.append(emb)

    print(f"[DEBUG] Retrieved {len(candidates)} candidates")

    if not candidates:
        return []

    sims = [cosine_similarity(query_emb, emb) for emb in candidate_embs]
    ranked_indices = np.argsort(sims)[::-1]
    ranked_candidates = [candidates[i] for i in ranked_indices]

    print("[DEBUG] Reranking complete based on cosine similarity.")
    return ranked_candidates[:top_k]

# ----------------------------
# Example Query
# ----------------------------
query_text = "what is finance"
top_results = rerank_query(query_text)

print("\nTop-K Reranked Chunks:\n")
for i, chunk in enumerate(top_results, 1):
    print(f"{i}. {chunk}\n")
