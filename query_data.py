from dotenv import load_dotenv
import ir_datasets
import os
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from google import genai
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from sentence_transformers import CrossEncoder
from metrics import Metrics

load_dotenv()
PATH_TEST = "beir/scifact/test"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BM25_MODEL_NAME = "Qdrant/bm25"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-3.5-flash"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load datasets
query_dataset = ir_datasets.load(PATH_TEST)
query_docs = list(query_dataset.queries_iter())

# Load qrels for evaluation
qrel = ir_datasets.load(PATH_TEST)
qrel_list = list(qrel.qrels_iter())

# embeddings
dense_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
bm25_embeddings = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

# qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# cross encoder
cross_encoder = CrossEncoder(model_name_or_path=CROSS_ENCODER_MODEL_NAME)

metrics = Metrics()

def get_qrel_doc_ids(qrels: list, query: object) -> list:
    filtered = [qrel for qrel in qrels if qrel.query_id == query.query_id]
    return [qrel.doc_id for qrel in filtered]

def get_reranked_results(query: str, results: list, rerank_top_k: int = 5) -> list:
    pairs = [(query, result.payload["text"]) for result in results]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:rerank_top_k]]

def get_context(results: list) -> str:
    context_parts = []
    for j, result in enumerate(results):
        part = f"[Document {j}] (doc_id: {result.payload['doc_id']}, title: {result.payload['doc_title']})\n{result.payload['text']}"
        context_parts.append(part)
    return "\n\n".join(context_parts)

def get_response(query: str, context: str) -> str:
    prompt = f"""Answer the following question based on the provided context. If you don't know the answer, just say that you don't know. \n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = genai_client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
    return response.text

recall_dense_only_list = []
recall_dense_bm25_list = []
recall_reranked_list = []

reciprocal_rank_dense_only_list = []
reciprocal_rank_dense_bm25_list = []
reciprocal_rank_reranked_list = []

for i, query in enumerate(query_docs):
    embeded_question = dense_embeddings.embed_query(query.text)
    bm25_embed = list(bm25_embeddings.embed([query.text]))[0]

    # dense only
    result_dense_only = client.query_points(
        collection_name="scifact",
        query=embeded_question,
        using="dense",
        limit=5
    )
    # context_dense_only = get_context(result_dense_only.points)
    # response_dense_only = get_response(query.text, context_dense_only)

    filtered_qrels = get_qrel_doc_ids(qrel_list, query)
    filtered_doc_ids = [point.payload["doc_id"] for point in result_dense_only.points]

    recall_dense_only = metrics.recall_at_k(filtered_doc_ids, filtered_qrels, k=5)
    recall_dense_only_list.append(recall_dense_only)

    reciprocal_rank_dense_only = metrics.reciprocal_rank(filtered_doc_ids, filtered_qrels)
    reciprocal_rank_dense_only_list.append(reciprocal_rank_dense_only)

    # dense + bm25
    results = client.query_points(
        collection_name="scifact",
        prefetch=[
            Prefetch(query=embeded_question, using="dense", limit=20),
            Prefetch(query=SparseVector(
                indices=bm25_embed.indices.tolist(),
                values=bm25_embed.values.tolist()
            ), using="bm25", limit=20)],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5
    )
    # context = get_context(results.points)
    # response = get_response(query.text, context)

    filtered_dense_bm25_doc_ids = [point.payload["doc_id"] for point in results.points]

    recall_dense_bm25 = metrics.recall_at_k(filtered_dense_bm25_doc_ids, filtered_qrels, k=5)
    recall_dense_bm25_list.append(recall_dense_bm25)

    reciprocal_rank_dense_bm25 = metrics.reciprocal_rank(filtered_dense_bm25_doc_ids, filtered_qrels)
    reciprocal_rank_dense_bm25_list.append(reciprocal_rank_dense_bm25)

    # dense+bm25+reranks
    results_for_reranking = client.query_points(
        collection_name="scifact",
        prefetch=[
            Prefetch(query=embeded_question, using="dense", limit=40),
            Prefetch(query=SparseVector(
                indices=bm25_embed.indices.tolist(),
                values=bm25_embed.values.tolist()
            ), using="bm25", limit=40)],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=20
    )
    # context_reranked = get_context(results_for_reranking.points)
    # response_reranked = get_response(query.text, context_reranked)
    reranked_results = get_reranked_results(query.text, results_for_reranking.points, rerank_top_k=5)
    filtered_reranked_doc_ids = [point.payload["doc_id"] for point in reranked_results]
    
    recall_reranked = metrics.recall_at_k(filtered_reranked_doc_ids, filtered_qrels, k=5)
    recall_reranked_list.append(recall_reranked)

    reciprocal_rank_reranked = metrics.reciprocal_rank(filtered_reranked_doc_ids, filtered_qrels)
    reciprocal_rank_reranked_list.append(reciprocal_rank_reranked)

mean_recall_dense_only = sum(recall_dense_only_list) / len(recall_dense_only_list)
print(f"Mean Recall@5 (Dense Only): {mean_recall_dense_only:.3f}")
mrr_dense_only = sum(reciprocal_rank_dense_only_list) / len(reciprocal_rank_dense_only_list)
print(f"Mean Reciprocal Rank (MRR) (Dense Only): {mrr_dense_only:.3f}\n\n")

mean_recall_dense_bm25 = sum(recall_dense_bm25_list) / len(recall_dense_bm25_list)
print(f"Mean Recall@5 (Dense + BM25): {mean_recall_dense_bm25:.3f}")
mrr_dense_bm25 = sum(reciprocal_rank_dense_bm25_list) / len(reciprocal_rank_dense_bm25_list)
print(f"Mean Reciprocal Rank (MRR) (Dense + BM25): {mrr_dense_bm25:.3f}\n\n")

mean_recall_reranked = sum(recall_reranked_list) / len(recall_reranked_list)
print(f"Mean Recall@5 (Dense + BM25 + Reranked): {mean_recall_reranked:.3f}")
mrr_reranked = sum(reciprocal_rank_reranked_list) / len(reciprocal_rank_reranked_list)
print(f"Mean Reciprocal Rank (MRR) (Reranked): {mrr_reranked:.3f}")