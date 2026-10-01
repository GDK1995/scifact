# команда запуска python query_data.py
import ir_datasets
import config
from sentence_transformers import CrossEncoder
from metrics import Metrics
from qdrant_service import QdrantService

# Load datasets
query_dataset = ir_datasets.load(config.PATH_TEST)
query_docs = list(query_dataset.queries_iter())

# Load qrels for evaluation
qrel = ir_datasets.load(config.PATH_TEST)
qrel_list = list(qrel.qrels_iter())

# init qdrant service
qdrant_client = QdrantService(
    qdrant_url=config.QDRANT_URL,
    qdrant_api_key=config.QDRANT_API_KEY,
    collection_name=config.COLLECTION_NAME,
    embed_hugging_model_name=config.EMBEDDING_MODEL_NAME,
    embed_bm25_model_name=config.BM25_MODEL_NAME
)

# cross encoder
cross_encoder = CrossEncoder(model_name_or_path=config.CROSS_ENCODER_MODEL_NAME)

metrics = Metrics() #init the metrics class for evaluation

def get_qrel_doc_ids(qrels: list, query: object) -> list:
    filtered = [qrel for qrel in qrels if qrel.query_id == query.query_id]
    return [qrel.doc_id for qrel in filtered]

def get_reranked_results(query: str, results: list, rerank_top_k: int = 5) -> list:
    pairs = [(query, result.payload["text"]) for result in results]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:rerank_top_k]]

recall_dense_only_list = []
recall_dense_bm25_list = []
recall_reranked_list = []

reciprocal_rank_dense_only_list = []
reciprocal_rank_dense_bm25_list = []
reciprocal_rank_reranked_list = []

for i, query in enumerate(query_docs):
    embeded_question = qdrant_client.embed_huggungface(query.text) # get the dense embedding for the query
    bm25_embed = qdrant_client.embed_bm25(query.text) #get bm25 embedding for the query

    # dense only
    result_dense_only = qdrant_client.query_dense(query_vector=embeded_question, limit=5) # get the top 5 results from dense query

    filtered_qrels = get_qrel_doc_ids(qrel_list, query) # Filter qrels for the current query
    filtered_doc_ids = [point.payload["doc_id"] for point in result_dense_only.points] # get document IDs for the filtered results

    recall_dense_only = metrics.recall_at_k(filtered_doc_ids, filtered_qrels, k=5) # evaluate recall at k=5 for dense only results
    recall_dense_only_list.append(recall_dense_only) # Append the recall value to the list

    reciprocal_rank_dense_only = metrics.reciprocal_rank(filtered_doc_ids, filtered_qrels) # evaluate reciprocal rank for dense only results
    reciprocal_rank_dense_only_list.append(reciprocal_rank_dense_only) # append the reciprocal rank value to the list

    # dense + bm25
    results = qdrant_client.query_dense_sparse(query_vector=embeded_question, query_bm25_vector=bm25_embed, limit=5, prefetch_limit=20) # get the top 5 results from dense + bm25 query

    filtered_dense_bm25_doc_ids = [point.payload["doc_id"] for point in results.points] # get document IDs for the filtered results

    recall_dense_bm25 = metrics.recall_at_k(filtered_dense_bm25_doc_ids, filtered_qrels, k=5) # evaluate recall at k=5 for dense + bm25 results
    recall_dense_bm25_list.append(recall_dense_bm25) # append the recall value to the list

    reciprocal_rank_dense_bm25 = metrics.reciprocal_rank(filtered_dense_bm25_doc_ids, filtered_qrels) # evaluate reciprocal rank for dense + bm25 results
    reciprocal_rank_dense_bm25_list.append(reciprocal_rank_dense_bm25) # append the reciprocal rank value to the list

    # dense+bm25+reranks
    results_for_reranking = qdrant_client.query_dense_sparse(query_vector=embeded_question, query_bm25_vector=bm25_embed, limit=20, prefetch_limit=40) # get the top 20 results from dense + bm25 query for reranking

    reranked_results = get_reranked_results(query.text, results_for_reranking.points, rerank_top_k=5) # rerank the top 20 results and get the top 5 reranked results
    filtered_reranked_doc_ids = [point.payload["doc_id"] for point in reranked_results] # get document IDs for the filtered reranked results
    
    recall_reranked = metrics.recall_at_k(filtered_reranked_doc_ids, filtered_qrels, k=5) # evaluate recall at k=5 for reranked results
    recall_reranked_list.append(recall_reranked) # append the recall value to the list

    reciprocal_rank_reranked = metrics.reciprocal_rank(filtered_reranked_doc_ids, filtered_qrels) # evaluate reciprocal rank for reranked results
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