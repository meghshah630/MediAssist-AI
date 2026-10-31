"""
config.py – App-wide constants and settings for MediAssist AI.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
SQLITE_DB_PATH = str(BASE_DIR / "data" / "mediassist.db")
SAMPLE_DOCS_PATH = str(BASE_DIR / "data" / "sample_docs")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "medical_docs"

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Ollama LLM ───────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "llama3.2"   # also support "phi3", "mistral"
SUPPORTED_MODELS = ["llama3.2", "phi3", "mistral"]

# ── Chunking ─────────────────────────────────────────────────────────────────
# Smaller chunks (200 words) provide more topically focused retrieval,
# reducing cross-topic contamination (e.g. hypertension meds retrieved for
# weight-loss questions). Overlap of 30 words preserves sentence continuity.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5

# ── Knowledge Base paths ─────────────────────────────────────────────────────
# Additional directories the ingestion pipeline should scan (in order)
EXTRA_DOCS_PATH: list = []  # users can add their own verified doc directories

# ── Chat ─────────────────────────────────────────────────────────────────────
MAX_HISTORY_MESSAGES = 20

# ── App metadata ─────────────────────────────────────────────────────────────
APP_NAME = "MediAssist AI"
APP_VERSION = "1.0.0"

# ── Disclaimer ───────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ DISCLAIMER: MediAssist AI is for educational purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider."
)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

