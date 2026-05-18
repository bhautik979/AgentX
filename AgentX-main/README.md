# 🤖 AgentX — Financial Literacy RAG System

> An intelligent, conversational AI agent for personalized financial guidance — powered by Retrieval-Augmented Generation (RAG), hybrid search, and local/cloud LLMs.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [What It Does](#-what-it-does)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [How to Run Locally](#-how-to-run-locally)
- [Environment Variables](#-environment-variables)
- [Configuration Reference](#-configuration-reference)
- [Knowledge Base Pipeline](#-knowledge-base-pipeline)
- [Troubleshooting](#-troubleshooting)

---

## 🧩 Problem Statement

Millions of people — especially in India — lack access to personalized, expert-level financial guidance. Generic financial advice online is often too broad, not actionable, and not adapted to individual income/expense situations.

**AgentX solves this by:**
- Grounding answers in a curated financial knowledge base (RAG), not hallucination
- Understanding user intent (budgeting, savings, emergency fund, Q&A)
- Maintaining multi-turn conversation memory per user
- Generating downloadable personalized budget reports (PDF/DOCX)
- Running fully locally via Ollama — **no cloud LLM cost required**

---

## ✨ What It Does

| Feature | Description |
|--------|-------------|
| 💬 **Financial Q&A** | Ask any personal finance question; answers grounded in indexed knowledge |
| 📊 **Budget Planning** | Get personalized budget advice using the 50/30/20 rule |
| 🆘 **Emergency Fund Calculator** | Auto-calculates 6-month emergency fund target with actionable savings plan |
| 📄 **Report Generation** | Download budget reports as DOCX or PDF (with charts) |
| 🧠 **Memory** | Tracks per-user conversation history and financial profile across turns |
| 🔍 **Hybrid Retrieval** | Combines dense (vector) + sparse (BM25) search + BGE reranking for best results |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER / CLIENT                        │
│              (REST API calls via /docs or curl)             │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Application (main.py)              │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  /api/ask   │  │  /api/chat   │  │  /api/reports    │   │
│  │  Q&A intent │  │  Multi-turn  │  │  Budget/EmFund   │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │
│         └────────────────┼────────────────────┘             │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │             FinancialAdvisorAgent (core/agent.py)     │  │
│  │                                                       │  │
│  │  1. Intent Detection  (budget / emergency / Q&A / …) │  │
│  │  2. Entity Extraction (income, expenses, goals)       │  │
│  │  3. Conversation Memory (per user_id)                 │  │
│  │  4. Routes to the right handler                       │  │
│  └────────┬──────────────────────────────┬───────────────┘  │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌─────────────────┐          ┌──────────────────────────┐  │
│  │  FinancialRAG   │          │      LLMHandler           │  │
│  │  Pipeline       │          │                          │  │
│  │  (core/rag_     │          │  ┌──────────┐            │  │
│  │   pipeline.py)  │          │  │  Ollama  │  (local)   │  │
│  │                 │          │  │ (Mistral)│            │  │
│  │ Dense Retrieval │          │  └──────────┘            │  │
│  │  (Pinecone)     │          │  ┌──────────┐            │  │
│  │ +               │          │  │  OpenAI  │  (cloud)   │  │
│  │ Sparse Retrieval│          │  │  (GPT-x) │            │  │
│  │  (BM25)         │          │  └──────────┘            │  │
│  │ +               │          └──────────────────────────┘  │
│  │ BGE Reranker    │                                         │
│  └────────┬────────┘                                         │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          Pinecone Vector Database                       │ │
│  │  (financial-knowledge-pc index, 768-dim embeddings)    │ │
│  │  Model: FinLang/finance-embeddings-investopedia        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          BudgetPlanner Tool (api/tools.py)              │ │
│  │  Generates DOCX / PDF reports with charts               │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### Knowledge Base Build Pipeline (one-time setup)

```
  PDF Financial Books / Documents
           │
           ▼
  [1] data_extraction.py         → Extracts text, tables, metadata
           │
           ▼
  [2] data_cleaning.py           → Deduplication, normalization, segmentation
           │
           ▼
  [3a] Static Chunking           → Fixed-size chunks (2048 / 512 / 128 tokens)
  [3b] Dynamic Chunking          → Semantic parent-child chunking
           │
           ▼
  [4] Embedding Generation       → FinLang/finance-embeddings-investopedia (768-dim)
           │
           ▼
  [5] Pinecone Upload            → Indexed and ready for retrieval
```

---

## 🛠️ Tech Stack

### Backend & API
| Layer | Technology |
|-------|-----------|
| Web Framework | **FastAPI** + Uvicorn |
| Data Validation | **Pydantic v2** |

### AI / ML
| Component | Technology |
|-----------|-----------|
| LLM (Local) | **Ollama** (Mistral, Llama 3, etc.) |
| LLM (Cloud) | **OpenAI GPT** (optional) |
| Orchestration | **LangChain** |
| Embeddings | **FinLang/finance-embeddings-investopedia** (domain-specific, 768-dim) |
| Dense Retrieval | **Pinecone** (vector similarity search) |
| Sparse Retrieval | **BM25** (`rank-bm25`) |
| Reranking | **BAAI/bge-reranker-v2-m3** (FlagEmbedding) |
| Tokenizer | **tiktoken** |

### Data & Reports
| Component | Technology |
|-----------|-----------|
| NLP preprocessing | **NLTK** |
| Numerical processing | **NumPy**, **Pandas** |
| DOCX generation | **python-docx** |
| PDF + charts | **ReportLab**, **Matplotlib**, **Pillow** |

### Dev / Testing
| Component | Technology |
|-----------|-----------|
| Testing | **pytest**, **pytest-asyncio** |
| Config | **python-dotenv** |
| Progress | **tqdm** |

---

## 📂 Project Structure

```
AgentX-main/
│
├── main.py                        # FastAPI app + all route definitions
├── config.py                      # Loads .env config with validation
├── requirements.txt               # Python dependencies
├── setup.sh                       # One-command environment setup (Linux/macOS)
├── test_all.py                    # Test suite for all endpoints
├── .env.example                   # Template for environment variables
│
├── core/                          # Core AI logic
│   ├── __init__.py
│   ├── agent.py                   # FinancialAdvisorAgent — intent detection, conversation memory, routing
│   ├── rag_pipeline.py            # Hybrid RAG: dense (Pinecone) + sparse (BM25) + BGE reranking
│   └── llm_handler.py             # Unified LLM abstraction: Ollama + OpenAI
│
├── api/                           # API models & tools
│   ├── __init__.py
│   ├── models.py                  # Pydantic request/response schemas
│   └── tools.py                   # BudgetPlanner: DOCX & PDF report generation
│
├── create_vb/                     # Knowledge base creation pipeline (run once)
│   ├── data_extraction.py         # Step 1: Extract text from PDFs
│   ├── data_cleaning.py           # Step 2: Clean and segment text
│   ├── data_embedding_static_chancking.py   # Step 3a: Static chunking + embed + upload
│   ├── data_embedding_dynamic_chancking.py  # Step 3b: Dynamic semantic chunking + embed + upload
│   ├── reranking_cosine.py        # Retrieval test: cosine similarity reranking
│   ├── reranking_BGE.py           # Retrieval test: BGE cross-encoder reranking
│   └── README.md                  # Detailed knowledge base pipeline docs
│
├── pre-processing/
│   └── data_cleaning.py           # Standalone text cleaning utilities
│
└── data/
    └── reports/                   # Generated budget reports saved here
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | System info and endpoint map |
| `GET` | `/health` | Health check for all components |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/api/v1/ask` | Ask a financial question (single turn) |
| `POST` | `/api/v1/chat/message` | Send a message (multi-turn conversation) |
| `GET` | `/api/v1/chat/history/{user_id}` | Fetch conversation history |
| `POST` | `/api/v1/reports/budget` | Generate a personalized budget report |
| `POST` | `/api/v1/reports/emergency-fund` | Calculate emergency fund target |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "query": "How do I build an emergency fund on Rs 30,000 salary?"}'
```

---

## 🚀 How to Run Locally

### Prerequisites

- Python **3.9+**
- [Ollama](https://ollama.ai/) installed and running locally (for local LLM)
- A free [Pinecone](https://www.pinecone.io/) account and API key
- Git

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/AgentX.git
cd AgentX
```

---

### Step 2 — Set up a virtual environment

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Download the required NLTK tokenizer data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

---

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values (see [Environment Variables](#-environment-variables) below).

---

### Step 5 — Start the local LLM (Ollama)

> Skip this step if you are using OpenAI instead.

```bash
# Pull and start the Mistral model
ollama pull mistral
ollama run mistral
```

Leave this running in a **separate terminal**.

---

### Step 6 — Run the API server

```bash
python main.py
```

The server will start at: **http://localhost:8000**

- 📖 Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🔄 Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

### Step 7 — (Optional) Run tests

```bash
pytest test_all.py -v
```

---

### Quick Setup Script (Linux/macOS only)

```bash
chmod +x setup.sh
./setup.sh
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```env
# ── Pinecone (Required) ──────────────────────────────────────
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=us-east-1
PINECONE_INDEX=financial-knowledge-pc

# ── LLM Configuration ─────────────────────────────────────────
LLM_TYPE=ollama           # Options: "ollama" (local) or "openai" (cloud)
LLM_MODEL=mistral         # Ollama: mistral / llama3 | OpenAI: gpt-3.5-turbo / gpt-4

# ── Ollama (if LLM_TYPE=ollama) ───────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434

# ── OpenAI (if LLM_TYPE=openai) ──────────────────────────────
# OPENAI_API_KEY=sk-your-openai-api-key-here

# ── Embedding ─────────────────────────────────────────────────
EMBED_MODEL=FinLang/finance-embeddings-investopedia

# ── Server ────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# ── RAG Parameters ────────────────────────────────────────────
TOP_K=5
SEMANTIC_THRESHOLD=0.55
REPORT_OUTPUT_DIR=./data/reports
```

---

## 📐 Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | `5` | Number of retrieved chunks passed to LLM |
| `SEMANTIC_THRESHOLD` | `0.55` | Min similarity score for dynamic semantic chunking |
| `LLM_TYPE` | `ollama` | LLM backend: `ollama` or `openai` |
| `LLM_MODEL` | `mistral` | Model name for chosen LLM backend |
| `EMBED_MODEL` | `FinLang/finance-embeddings-investopedia` | Sentence embedding model |
| `PINECONE_INDEX` | `financial-knowledge-pc` | Name of your Pinecone index |
| `REPORT_OUTPUT_DIR` | `./data/reports` | Directory for generated report files |

---

## 📚 Knowledge Base Pipeline

> The Pinecone index must be populated **once** before running the server.
> See [`create_vb/README.md`](./create_vb/README.md) for complete step-by-step instructions.

**Quick summary:**

```bash
cd create_vb

# Step 1: Extract text from financial PDFs
python data_extraction.py

# Step 2: Clean and normalize the text
python data_cleaning.py

# Step 3: Chunk, embed, and upload to Pinecone
#   Option A (static fixed-size chunking):
python data_embedding_static_chancking.py

#   Option B (semantic parent-child chunking — recommended):
python data_embedding_dynamic_chancking.py
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `PINECONE_API_KEY not found` | Ensure `.env` file exists and key is set correctly |
| `Cannot connect to Ollama` | Run `ollama run mistral` in a separate terminal first |
| `503 Agent not initialized` | Check startup logs — Pinecone connection or model loading may have failed |
| `BGE Reranker not loading` | Install `pip install FlagEmbedding`; ensure ~1 GB disk space for model download |
| Memory issues during embedding | Reduce `UPSERT_BATCH_SIZE` in embedding scripts; use GPU if available |
| Slow first query | Embedding model and BGE reranker are loaded on first use — this is expected |

---

## 🔐 Security Notes

- **Never commit** your `.env` file or API keys to version control
- `.env` is listed in `.gitignore` by default
- Rotate your Pinecone and OpenAI keys regularly
- For production deployments, use a secrets manager instead of `.env`

---

## 📄 License

This project is provided for educational and research purposes.

---

## 🙌 Acknowledgements

- [Pinecone](https://www.pinecone.io/) — Vector database
- [Ollama](https://ollama.ai/) — Local LLM inference
- [FinLang](https://huggingface.co/FinLang) — Finance-specific embedding model
- [BAAI/BGE](https://github.com/FlagOpen/FlagEmbedding) — State-of-the-art reranking
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance Python web framework
