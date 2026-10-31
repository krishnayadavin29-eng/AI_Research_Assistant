import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL: str = "gemini-2.0-flash"
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 2048

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "models/embedding-001"   # Gemini free embedding model

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_VECTOR: int = 6
TOP_K_BM25: int = 6
TOP_K_FINAL: int = 4          # after re-ranking
HYBRID_ALPHA: float = 0.5     # 0 = BM25 only, 1 = vector only

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data", "papers")
VECTOR_STORE_PATH: str = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
