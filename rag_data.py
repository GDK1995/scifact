import ir_datasets
import os
from dotenv import load_dotenv
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector, SparseVectorParams

load_dotenv()
PATH = "beir/scifact"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BM25_MODEL_NAME = "Qdrant/bm25"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
BATCH_SIZE = 100

dataset = ir_datasets.load(PATH)
docs = dataset.docs_iter()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

dense_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
bm25_embeddings = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

def split_doc(text: str) -> list[str]:
    chunks = text_splitter.split_text(text)
    return chunks

points = []
point_id = 0
for d in docs:
    chunks = split_doc(d.text)
    for chunk in chunks:
        dense_vector = dense_embeddings.embed_query(chunk)
        bm25_vector = list(bm25_embeddings.embed([chunk]))[0]
        points.append(PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vector,
                    "bm25": SparseVector(
                        indices=bm25_vector.indices.tolist(),
                        values=bm25_vector.values.tolist(),
                    ),
                },
                payload={"doc_id": d.doc_id, "doc_title": d.title, "text": chunk},
            )
        )
        point_id += 1

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

if client.collection_exists("scifact"):
    client.delete_collection("scifact")

client.create_collection(
    collection_name="scifact",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "bm25": SparseVectorParams()
    }
)

for i in range(0, len(points), BATCH_SIZE):
    client.upsert(collection_name="scifact", points=points[i:i + BATCH_SIZE])
