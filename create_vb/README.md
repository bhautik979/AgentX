# Financial Knowledge Base - RAG System

## 📋 Project Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline for financial documents. The system extracts data from PDF files, processes and cleans the text, applies chunking strategies (both static and dynamic), generates embeddings using domain-specific models, and implements advanced retrieval techniques with reranking mechanisms.

## 🚀 Features

- **PDF Data Extraction**: Extract text, tables, and metadata from financial PDFs
- **Data Cleaning & Segmentation**: Remove duplicates, normalize text, and segment into manageable chunks
- **Dual Chunking Strategies**:
  - **Static Chunking**: Fixed-size chunks with overlap across multiple granularities
  - **Dynamic Chunking**: Semantic-based parent-child chunking for context-aware retrieval
- **Domain-Specific Embeddings**: Uses `FinLang/finance-embeddings-investopedia` model
- **Vector Database**: Pinecone integration for efficient similarity search
- **Advanced Reranking**:
  - Cosine similarity-based reranking
  - BGE (BAAI) reranker for improved relevance

---

## 📂 Project Structure

```
.
├── data_extraction.py                      # Step 1: PDF extraction
├── data_cleaning.py                        # Step 2: Text cleaning & segmentation
├── data_embedding_static_chancking.py      # Step 3a: Static chunking + embedding
├── data_embedding_dynamic_chancking.py     # Step 3b: Dynamic chunking + embedding
├── reranking_cosine.py                     # Step 4a: Retrieval with cosine reranking
├── reranking_BGE.py                        # Step 4b: Retrieval with BGE reranking
└── README.md                               # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Required Libraries

```bash
pip install pdfplumber PyPDF2 camelot-py nltk tiktoken sentence-transformers pinecone-client numpy tqdm FlagEmbedding
```

### Additional Setup

```python
# Download NLTK punkt tokenizer
import nltk
nltk.download('punkt')
```

---

## 📝 Step-by-Step Usage

### **Step 1: Data Extraction from PDFs**

Extract text, tables, and metadata from PDF documents.

**Script**: `data_extraction.py`

**Configuration**:
```python
INPUT_DIR = "/path/to/pdf/folder"        # Folder containing PDF files
OUTPUT_FILE = "knowledge_base.jsonl"     # Output JSONL file
MAX_PAGES = None                         # Set to None for all pages
```

**Run**:
```bash
python data_extraction.py
```

**Output**: Creates `knowledge_base.jsonl` with extracted data in JSONL format.

**Output Format**:
```json
{
  "doc_id": "filename.pdf_p1",
  "source_file": "filename.pdf",
  "page_number": 1,
  "text": "Extracted text content...",
  "tables": [...],
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "subject": "Subject",
    "keywords": "Keywords"
  }
}
```

---

### **Step 2: Data Cleaning and Segmentation**

Clean the extracted text, remove duplicates, normalize formatting, and segment into chunks.

**Script**: `data_cleaning.py`

**Configuration**:
```python
INPUT_FILE = "knowledge_base.jsonl"
OUTPUT_FILE = "knowledge_base_cleaned_segmented.jsonl"
MAX_TOKENS_PER_SEGMENT = 500
```

**Run**:
```bash
python data_cleaning.py
```

**Output**: Creates `knowledge_base_cleaned_segmented.jsonl` with cleaned and segmented text.

**Processing Steps**:
1. Remove non-breaking spaces and encoding artifacts
2. Deduplicate repeated lines
3. Normalize text (lowercase, whitespace, punctuation)
4. Segment text into chunks based on token limits
5. Track token counts using tiktoken

---

### **Step 3: Chunking and Embedding**

Choose one of the two chunking strategies:

#### **Option A: Static Chunking** (Fixed-size chunks)

**Script**: `data_embedding_static_chancking.py`

**Configuration**:
```python
INPUT_FILE = "knowledge_base_cleaned_segmented.jsonl"
OUTPUT_EMB_FILE = "knowledge_base_embeddings.jsonl"
EMBED_MODEL = "all-mpnet-base-v2"
CHUNK_SIZES = [2048, 512, 128]  # Multi-granularity chunks
OVERLAP_TOKENS = 50
PINECONE_API_KEY = "your_pinecone_api_key"
INDEX_NAME = "financial-knowledge"
```

**Features**:
- Creates chunks at multiple granularities (2048, 512, 128 tokens)
- Applies overlap between consecutive chunks
- Generates embeddings using SentenceTransformer
- Uploads vectors to Pinecone

**Run**:
```bash
python data_embedding_static_chancking.py
```

---

#### **Option B: Dynamic Chunking** (Semantic parent-child strategy)

**Script**: `data_embedding_dynamic_chancking.py`

**Configuration**:
```python
INPUT_FILE = "knowledge_base_cleaned_segmented.jsonl"
OUTPUT_EMB_FILE = "knowledge_base_embeddings_semantic_parent_child.jsonl"
EMBED_MODEL = "FinLang/finance-embeddings-investopedia"
CHUNK_SIZES = [2048, 512, 128]
PARENT_SIZE = 2048  # Context stored in metadata
CHILD_SIZE = 128    # Used for embedding
SEMANTIC_THRESHOLD = 0.55  # Similarity threshold for semantic chunking
```

**Features**:
- Semantic chunking based on sentence similarity
- Parent-child relationship: child chunks are embedded, parent chunks stored as context
- Better context preservation for retrieval
- Uses finance-specific embedding model

**Run**:
```bash
python data_embedding_dynamic_chancking.py
```

**Output**: Both options create embeddings file and upload vectors to Pinecone.

---

### **Step 4: Retrieval and Reranking**

Choose one of the reranking techniques:

#### **Option A: Cosine Similarity Reranking**

**Script**: `reranking_cosine.py`

**Configuration**:
```python
EMBED_MODEL = "FinLang/finance-embeddings-investopedia"
TOP_K = 5
INDEX_NAME = "financial-knowledge"
```

**Features**:
- Fetches initial candidates from Pinecone
- Removes duplicate results
- Reranks using cosine similarity scores
- Returns top-K most relevant chunks

**Run**:
```bash
python reranking_cosine.py
```

**Query Function**:
```python
query_text = "What is an emergency fund?"
top_results = rerank_query(query_text, top_k=5)
```

---

#### **Option B: BGE Reranking** (Advanced)

**Script**: `reranking_BGE.py`

**Configuration**:
```python
EMBED_MODEL = "all-mpnet-base-v2"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
TOP_K = 5
USE_BGE_RERANKER = True
```

**Features**:
- Fetches more initial candidates (10x multiplier)
- Applies state-of-the-art BGE reranker
- Significantly improves relevance ranking
- Falls back to cosine similarity if reranker unavailable

**Run**:
```bash
python reranking_BGE.py
```

**Query Function**:
```python
query_text = "What is an emergency fund?"
top_results = rerank_query(query_text, top_k=5)
```

---

## 🔄 Complete Pipeline Workflow

```
┌─────────────────────┐
│   PDF Documents     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 1. Data Extraction  │  ← data_extraction.py
│   (pdfplumber,      │
│    PyPDF2, camelot) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. Data Cleaning    │  ← data_cleaning.py
│   & Segmentation    │
│   (dedup, normalize)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. Chunking         │  ← Choose one:
│   Strategy          │     • Static (fixed-size)
│                     │     • Dynamic (semantic)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. Embedding        │  ← FinLang/finance-embeddings
│   Generation        │     or all-mpnet-base-v2
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. Vector Database  │  ← Pinecone (cosine metric)
│   Indexing          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. Query &          │  ← Choose one:
│   Reranking         │     • Cosine similarity
│                     │     • BGE reranker
└─────────────────────┘
```

---

## ⚙️ Configuration Details

### Environment Variables

```bash
export PINECONE_API_KEY="your_api_key_here"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZES` | `[2048, 512, 128]` | Token sizes for multi-granularity chunking |
| `OVERLAP_TOKENS` | `50` | Overlap between consecutive chunks |
| `SEMANTIC_THRESHOLD` | `0.55` | Similarity threshold for semantic chunking |
| `VECTOR_DIM` | `768` | Embedding dimension |
| `TOP_K` | `5` | Number of results to return |
| `UPSERT_BATCH_SIZE` | `50` | Batch size for Pinecone uploads |

---

## 📊 Model Information

### Embedding Models

1. **FinLang/finance-embeddings-investopedia**
   - Domain-specific model for financial text
   - 768-dimensional embeddings
   - Optimized for finance terminology

2. **all-mpnet-base-v2**
   - General-purpose sentence transformer
   - 768-dimensional embeddings
   - Good baseline performance

### Reranking Model

- **BAAI/bge-reranker-v2-m3**
  - State-of-the-art cross-encoder reranker
  - Significantly improves retrieval relevance
  - Requires FlagEmbedding library

---

## 🎯 Usage Examples

### Basic Query Example

```python
from reranking_BGE import rerank_query

# Initialize (done automatically in script)
# Query the system
query = "What is an emergency fund?"
results = rerank_query(query, top_k=5)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['rerank_score']:.4f}")
    print(f"   Text: {result['text'][:200]}...")
    print(f"   Source: {result['source']}")
    print()
```

### Custom Query Function

```python
def search_financial_docs(query, top_k=5, use_bge=True):
    """
    Search financial documents with custom parameters
    """
    if use_bge:
        from reranking_BGE import rerank_query
    else:
        from reranking_cosine import rerank_query
    
    results = rerank_query(query, top_k=top_k)
    return results
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Pinecone Index Not Found**
   - Ensure `PINECONE_API_KEY` is set correctly
   - Index is created automatically on first run
   - Check Pinecone dashboard for index status

2. **Memory Issues During Embedding**
   - Reduce `UPSERT_BATCH_SIZE`
   - Process fewer documents at a time
   - Use GPU if available for faster processing

3. **BGE Reranker Not Loading**
   - Install FlagEmbedding: `pip install FlagEmbedding`
   - Ensure sufficient disk space (~1GB for model)
   - Check internet connection for first-time download

4. **Token Limit Exceeded**
   - Adjust `MAX_TOKENS_PER_SEGMENT` in data_cleaning.py
   - Reduce `CHUNK_SIZES` values
   - Increase `OVERLAP_TOKENS` for better context

---

## 📈 Performance Tips

1. **Optimize Chunking**
   - Use dynamic chunking for better semantic coherence
   - Adjust `SEMANTIC_THRESHOLD` based on data characteristics

2. **Improve Retrieval**
   - Use BGE reranker for best results
   - Increase initial candidate multiplier in reranking

3. **Speed Up Processing**
   - Use batch processing for embeddings
   - Cache models locally (set `cache_folder` parameter)
   - Use GPU for embedding generation

---

## 📚 Dependencies

```
pdfplumber>=0.9.0
PyPDF2>=3.0.0
camelot-py>=0.11.0
nltk>=3.8
tiktoken>=0.5.0
sentence-transformers>=2.2.0
pinecone-client>=3.0.0
numpy>=1.24.0
tqdm>=4.65.0
FlagEmbedding>=1.2.0
```

---

## 🔐 Security Notes

- **API Keys**: Never commit API keys to version control
- Use environment variables or secure key management
- Rotate API keys regularly
- The Pinecone API keys in the code should be replaced with your own

---

## 🤝 Contributing

To extend this project:

1. Add new chunking strategies in separate scripts
2. Implement additional reranking methods
3. Integrate with LLM for answer generation
4. Add evaluation metrics for retrieval quality

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 👥 Contact

For questions or issues, please refer to the documentation of individual libraries or contact the project maintainers.

---

## 🎓 References

- **Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **Pinecone Documentation**: [https://docs.pinecone.io/](https://docs.pinecone.io/)
- **BGE Reranker**: [https://github.com/FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- **FinLang Models**: Specialized for financial domain embeddings

---

**Happy Building! 🚀**
