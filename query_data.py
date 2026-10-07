# команда запуска python query_data.py
import ir_datasets
import config
import csv
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
    pairs = [(query, result.payload['text']) for result in results]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:rerank_top_k]]

def get_doc_ids(results: list, limit: int) -> list:
    seen = set()
    filtered_doc_ids = []
    for result in results:
        doc_id = result.payload["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            filtered_doc_ids.append(doc_id)
        if len(filtered_doc_ids) == limit:
            break

    return filtered_doc_ids

def save_results_to_csv(results: list, filename: str):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "query_id", "query_text", "relevant_qrels",
            "dense_doc_ids", "dense_recall_at_5", "dense_reciprocal_rank",
            "dense_bm25_doc_ids", "dense_bm25_recall_at_5", "dense_bm25_reciprocal_rank", "dense_bm25_results",
            "reranked_doc_ids", "reranked_recall_at_5", "reranked_reciprocal_rank", "reranked_results"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

recall_dense_only_list = []
recall_dense_bm25_list = []
recall_reranked_list = []

reciprocal_rank_dense_only_list = []
reciprocal_rank_dense_bm25_list = []
reciprocal_rank_reranked_list = []

score_list = []
for i, query in enumerate(query_docs):
    filtered_qrels = get_qrel_doc_ids(qrel_list, query) # Filter qrels for the current query
    embeded_question = qdrant_client.embed_huggungface(query.text) # get the dense embedding for the query
    bm25_embed = qdrant_client.embed_bm25(query.text) #get bm25 embedding for the query

    # dense only
    result_dense_only = qdrant_client.query_dense(query_vector=embeded_question, limit=30) # get the top 30 results from dense query
    filtered_doc_ids = get_doc_ids(result_dense_only.points, limit=5) # get 5 document IDs for the filtered results

    recall_dense_only = metrics.recall_at_k(filtered_doc_ids, filtered_qrels, k=5) # evaluate recall at k=5 for dense only results
    recall_dense_only_list.append(recall_dense_only) # Append the recall value to the list

    reciprocal_rank_dense_only = metrics.reciprocal_rank(filtered_doc_ids, filtered_qrels) # evaluate reciprocal rank for dense only results
    reciprocal_rank_dense_only_list.append(reciprocal_rank_dense_only) # append the reciprocal rank value to the list

    # dense + bm25
    results = qdrant_client.query_dense_sparse(query_vector=embeded_question, query_bm25_vector=bm25_embed, limit=30, prefetch_limit=40) # get the top 30 results from dense + bm25 query
    filtered_dense_bm25_doc_ids = get_doc_ids(results.points, limit=5) # get 5 document IDs for the filtered results

    recall_dense_bm25 = metrics.recall_at_k(filtered_dense_bm25_doc_ids, filtered_qrels, k=5) # evaluate recall at k=5 for dense + bm25 results
    recall_dense_bm25_list.append(recall_dense_bm25) # append the recall value to the list

    reciprocal_rank_dense_bm25 = metrics.reciprocal_rank(filtered_dense_bm25_doc_ids, filtered_qrels) # evaluate reciprocal rank for dense + bm25 results
    reciprocal_rank_dense_bm25_list.append(reciprocal_rank_dense_bm25) # append the reciprocal rank value to the list

    # dense+bm25+reranks
    reranked_results = get_reranked_results(query.text, results.points, rerank_top_k=30) # rerank the top 30 results and get the top 30 reranked results
    filtered_reranked_doc_ids = get_doc_ids(reranked_results, limit=5) # get 5 document IDs for the filtered reranked results

    recall_reranked = metrics.recall_at_k(filtered_reranked_doc_ids, filtered_qrels, k=5) # evaluate recall at k=5 for reranked results
    recall_reranked_list.append(recall_reranked) # append the recall value to the list

    reciprocal_rank_reranked = metrics.reciprocal_rank(filtered_reranked_doc_ids, filtered_qrels) # evaluate reciprocal rank for reranked results
    reciprocal_rank_reranked_list.append(reciprocal_rank_reranked)

    # append the scores to the list
    score_list.append({
        "query_id": query.query_id,
        "query_text": query.text,
        "relevant_qrels": filtered_qrels,
        "dense_doc_ids": filtered_doc_ids,
        "dense_recall_at_5": recall_dense_only,
        "dense_reciprocal_rank": reciprocal_rank_dense_only,
        "dense_bm25_doc_ids": filtered_dense_bm25_doc_ids,
        "dense_bm25_recall_at_5": recall_dense_bm25,
        "dense_bm25_reciprocal_rank": reciprocal_rank_dense_bm25,
        "dense_bm25_results": results,
        "reranked_doc_ids": filtered_reranked_doc_ids,
        "reranked_recall_at_5": recall_reranked,
        "reranked_reciprocal_rank": reciprocal_rank_reranked,
        "reranked_results": reranked_results
    })
save_results_to_csv(score_list, "results.csv") # save the results to a csv file

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