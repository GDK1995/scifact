# команда запска uvicorn ask:app --reload
from fastapi import FastAPI
from google import genai
import config
from qdrant_service import QdrantService
from sentence_transformers import CrossEncoder

# init qdrant service
qdrant_client = QdrantService(
    qdrant_url=config.QDRANT_URL,
    qdrant_api_key=config.QDRANT_API_KEY,
    collection_name=config.COLLECTION_NAME,
    embed_hugging_model_name=config.EMBEDDING_MODEL_NAME,
    embed_bm25_model_name=config.BM25_MODEL_NAME
)

# gemini client
genai_client = genai.Client(api_key=config.GEMINI_API_KEY)

# init FastAPI
app = FastAPI(title="SciFact RAG")

# cross encoder
cross_encoder = CrossEncoder(model_name_or_path=config.CROSS_ENCODER_MODEL_NAME)

def get_context(results: list) -> str:
    context_parts = []
    for j, result in enumerate(results):
        part = f"[Document {j}] (doc_id: {result.payload['doc_id']}, title: {result.payload['doc_title']})\n{result.payload['text']}"
        context_parts.append(part)
    return "\n\n".join(context_parts)

def get_response(query: str, context: str) -> str:
    prompt = f"""Answer the following question based on the provided context. If you don't know the answer, just say that you don't know. \n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = genai_client.models.generate_content(model=config.GEMINI_MODEL_NAME, contents=prompt)
    return response.text

def get_reranked_results(query: str, results: list, rerank_top_k: int = 5) -> list:
    pairs = [(query, result.payload["text"]) for result in results]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:rerank_top_k]]

def get_source_from_reranked_result(results: list) -> list:
    sources = []
    for result in results:
        doc_id = result.payload["doc_id"]
        doc_title = result.payload["doc_title"]
        doc_text = result.payload["text"]
        sources.append(f"Document ID: {doc_id}, Title: {doc_title}, Text: {doc_text}")
    return sources

@app.post("/ask")
async def ask_question(query: str):
    embeded_question = qdrant_client.embed_huggungface(query) # get the dense embedding for the query
    bm25_embed = qdrant_client.embed_bm25(query) #get bm25 embedding for the query

    results_for_reranking = qdrant_client.query_dense_sparse(query_vector=embeded_question, query_bm25_vector=bm25_embed, limit=20, prefetch_limit=40) # get the top 20 results from dense + bm25 query for reranking
    reranked_results = get_reranked_results(query, results_for_reranking.points, rerank_top_k=5)

    context = get_context(reranked_results) # get the context from the top 20 results
    response = get_response(query, context) # get the response from the gemini model

    sources = get_source_from_reranked_result(reranked_results) # get the source from the reranked results
    return {"query": query, "response": response, "sources": sources}

@app.get("/")
def root():
    return {"status": "ok", "collection_count": qdrant_client.count()}
