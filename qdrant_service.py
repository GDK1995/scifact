from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client.models import Prefetch, models, FusionQuery, Fusion, SparseVector

class QdrantService:

    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str, embed_hugging_model_name: str, embed_bm25_model_name: str):
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_hugging_model_name)
        self.bm25_embeddings = SparseTextEmbedding(model_name=embed_bm25_model_name)

    def embed_huggungface(self, text: str):
        return self.embeddings.embed_query(text)

    def embed_bm25(self, text: str):
        return list(self.bm25_embeddings.embed([text]))[0]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def query_dense(self, query_vector, limit: int = 5):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=limit
        )

    def query_dense_sparse(self, query_vector, query_bm25_vector, limit: int = 5, prefetch_limit: int = 20):
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=query_vector, using="dense", limit=prefetch_limit),
                Prefetch(query=SparseVector(
                    indices=query_bm25_vector.indices.tolist(),
                    values=query_bm25_vector.values.tolist()
                ), using="bm25", limit=prefetch_limit)],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit
        )

    def create_collection(self, vector_size: int = 384):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        )

    def delete_collection(self):
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

    def upsert_points(self, points, batch_size: int = 100):
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection_name, points=points[i:i + batch_size])
    
