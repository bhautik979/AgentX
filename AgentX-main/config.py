import os
from dotenv import load_dotenv

load_dotenv()

# ==================== API Configuration ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ==================== Pinecone Configuration ====================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "financial-knowledge-pc")

# ==================== LLM Configuration ====================
LLM_TYPE = os.getenv("LLM_TYPE", "ollama")  # ollama or openai
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ==================== Embedding Configuration ====================
EMBED_MODEL = os.getenv("EMBED_MODEL", "FinLang/finance-embeddings-investopedia")

# ==================== RAG Configuration ====================
TOP_K = int(os.getenv("TOP_K", 5))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", 0.55))

# ==================== Report Configuration ====================
REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "./data/reports")
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# ==================== Logging ====================
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

print("✅ Configuration loaded successfully")