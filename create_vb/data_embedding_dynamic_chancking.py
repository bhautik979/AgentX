import os
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import tiktoken
from sentence_transformers import SentenceTransformer, util
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from numpy.linalg import norm
from FlagEmbedding import FlagReranker

# ----------------------------
# CONFIGURATION
# ----------------------------
INPUT_FILE = "/home/karan.padariya/LMA_Major_project/knowledge_base_cleaned_segmented.jsonl"
OUTPUT_EMB_FILE = "/ssd_scratch/karan.p/knowledge_base_embeddings_semantic_parent_child.jsonl"

EMBED_MODEL = "FinLang/finance-embeddings-investopedia"
VECTOR_DIM = 768
CHUNK_SIZES = [2048, 512, 128]  # multi-granularity (used to generate chunks)
PARENT_SIZE = max(CHUNK_SIZES)  # 2048 tokens - Stored as the context text
CHILD_SIZE = min(CHUNK_SIZES)   # 128 tokens - Used for the vector embedding
SEMANTIC_THRESHOLD = 0.55       # lower → finer chunks
TOP_K = 5
UPSERT_BATCH_SIZE = 50
OVERLAP_TOKENS = 50             # overlap between consecutive chunks

PINECONE_API_KEY = "pcsk_4Ndnwt_6wZ5exov2n7isQMJLnGtSFmFwidxmAuzc6nzKAAdabWV1zyScDTK59vV8krjF3F"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "financial-knowledge-pc"  # Use a new index name

USE_BGE_RERANKER = True
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ----------------------------
# INITIALIZE MODELS
# ----------------------------
print("[DEBUG] Initializing models...")
enc = tiktoken.encoding_for_model("text-embedding-3-small")
embed_model = SentenceTransformer(EMBED_MODEL, cache_folder="/ssd_scratch/karan.p")

# ----------------------------
# Initialize Pinecone
# ----------------------------
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if INDEX_NAME not in pc.list_indexes().names():
    print(f"[DEBUG] Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

index = pc.Index(INDEX_NAME)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def tokenize_count(text):
    return len(enc.encode(text))

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def semantic_chunk(sentences, threshold=SEMANTIC_THRESHOLD):
    if not sentences:
        return []
    embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(util.cos_sim(embeddings[i - 1], embeddings[i]))
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def segment_text_hybrid(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    all_chunks = {}
    for size in CHUNK_SIZES:
        chunks_for_size = []
        for para in paragraphs:
            sentences = sent_tokenize(para)
            semantic_chunks = semantic_chunk(sentences, threshold=SEMANTIC_THRESHOLD)
            for chunk in semantic_chunks:
                chunk_tokens = enc.encode(chunk)
                start = 0
                while start < len(chunk_tokens):
                    end = min(start + size, len(chunk_tokens))
                    chunk_text = enc.decode(chunk_tokens[start:end])
                    chunks_for_size.append(chunk_text)
                    if end == len(chunk_tokens):
                        break
                    start = end - OVERLAP_TOKENS
        all_chunks[size] = chunks_for_size
    return all_chunks

# ----------------------------
# PIPELINE: Chunk + Embed (Child) + Index (Parent Text in Metadata)
# ----------------------------
processed_chunks = []
upsert_batch = []

print(f"[DEBUG] Starting Parent-Child Indexing (Parent Size: {PARENT_SIZE}, Child Size: {CHILD_SIZE})...")

with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    for line_num, line in enumerate(tqdm(infile, desc="Processing documents"), 1):
        try:
            record = json.loads(line)
            text = record.get("text", "")
            multi_chunks = segment_text_hybrid(text)

            parent_chunks = multi_chunks.get(PARENT_SIZE, [])
            child_chunks = multi_chunks.get(CHILD_SIZE, [])

            if not parent_chunks or not child_chunks:
                print(f"[WARNING] Doc {line_num} has missing parent or child chunks. Skipping.")
                continue

            num_children = len(child_chunks)
            num_parents = len(parent_chunks)

            for i, child_chunk in enumerate(child_chunks, 1):
                parent_idx = min(int((i - 1) * num_parents / num_children), num_parents - 1)
                parent_text = parent_chunks[parent_idx]

                child_doc_id = f"{record.get('doc_id','doc')}_child{CHILD_SIZE}_chunk{i}"
                parent_doc_id = record.get("doc_id", "doc")
                embedding = embed_model.encode(child_chunk).tolist()

                chunk_record = {
                    "doc_id": child_doc_id,
                    "parent_doc_id": parent_doc_id,
                    "child_text": child_chunk,
                    "parent_text": parent_text,
                    "token_count": tokenize_count(child_chunk),
                    "granularity": CHILD_SIZE,
                    "embedding": embedding,
                    "metadata": record.get("metadata", {}),
                }
                processed_chunks.append(chunk_record)

                metadata_for_pinecone = {
                    "parent_doc_id": parent_doc_id,
                    "retrieval_text": parent_text,
                    "source_file": record.get("source_file"),
                    "page_number": record.get("page_number"),
                    "granularity": CHILD_SIZE,
                }
                upsert_batch.append((child_doc_id, embedding, metadata_for_pinecone))

                if len(upsert_batch) >= UPSERT_BATCH_SIZE:
                    index.upsert(vectors=upsert_batch)
                    upsert_batch = []

            if line_num % 10 == 0:
                print(f"[DEBUG] Processed {line_num} documents...")

        except Exception as e:
            print(f"[ERROR] Failed to process line {line_num}: {e}")

if upsert_batch:
    index.upsert(vectors=upsert_batch)

with open(OUTPUT_EMB_FILE, "w", encoding="utf-8") as f:
    for rec in tqdm(processed_chunks, desc="Saving embeddings"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Parent-Child Indexing complete! {len(processed_chunks)} child vectors indexed.")

# ----------------------------
# Reranking Query Function
# ----------------------------
reranker = None
if USE_BGE_RERANKER:
    try:
        reranker = FlagReranker(BGE_RERANKER_MODEL, use_fp16=True)
        print(f"[INFO] BGE Reranker loaded")
    except Exception as e:
        print(f"[WARNING] Reranker not available: {e}")

def rerank_query(query, top_k=TOP_K):
    initial_k = top_k * 3 if reranker else top_k
    results = index.query(
        vector=embed_model.encode(query).tolist(),
        top_k=initial_k,
        include_metadata=True,
    )
    candidates = [(match["metadata"]["retrieval_text"], match["score"]) for match in results["matches"]]
    if reranker and len(candidates) > top_k:
        pairs = [[query, text] for text, _ in candidates]
        rerank_scores = reranker.compute_score(pairs)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [text for (text, _), _ in ranked[:top_k]]
    return [text for text, _ in candidates[:top_k]]

# ----------------------------
# Example Query
# ----------------------------
query_text = "What is an emergency fund?"
top_results = rerank_query(query_text)

print("\nTop-K Reranked PARENT Chunks (Context):\n")
for i, chunk in enumerate(top_results, 1):
    print(f"{i}. {chunk}\n")
