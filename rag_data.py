# команда запуска python rag_data.py
import ir_datasets
import config
from transformers import AutoTokenizer
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from qdrant_service import QdrantService
from qdrant_client.models import PointStruct, SparseVector

# Load datasets
dataset = ir_datasets.load(config.PATH)
docs = dataset.docs_iter()

# text splitter based on chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)

tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_FULL_NAME)

# init qdrant service
qdrant_client = QdrantService(
    qdrant_url=config.QDRANT_URL,
    qdrant_api_key=config.QDRANT_API_KEY,
    collection_name=config.COLLECTION_NAME,
    embed_hugging_model_name=config.EMBEDDING_MODEL_NAME,
    embed_bm25_model_name=config.BM25_MODEL_NAME
)

# def to split text into chunks
def split_doc(text: str) -> list[str]:
    chunks = text_splitter.split_text(text)
    return chunks

# create points for qdrant
points = []
point_id = 0
for d in docs:
    chunks = split_doc(d.text)
    for chunk in chunks:
        dense_vector = qdrant_client.embed_huggungface(chunk)
        bm25_vector = qdrant_client.embed_bm25(chunk)
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


# check if collection exists, if yes delete it and create a new one
qdrant_client.delete_collection()

# create collection with dense and sparse vectors
qdrant_client.create_collection(vector_size=config.VECTOR_SIZE)

# upsert points in batches
qdrant_client.upsert_points(points, batch_size=config.BATCH_SIZE)
