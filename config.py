import os
from dotenv import load_dotenv

load_dotenv()
PATH = "beir/scifact"
PATH_TEST = "beir/scifact/test"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_FULL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BM25_MODEL_NAME = "Qdrant/bm25"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 100
VECTOR_SIZE = 384

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "scifact"

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GEMINI_MODEL_NAME = "gemini-3.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
