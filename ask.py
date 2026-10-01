from fastapi import FastAPI
from google import genai
import config
from qdrant_service import QdrantService

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

@app.post("/ask")
async def ask_question(query: str):
    embeded_question = qdrant_client.embed_huggungface(query) # get the dense embedding for the query
    bm25_embed = qdrant_client.embed_bm25(query) #get bm25 embedding for the query

    results_for_reranking = qdrant_client.query_dense_sparse(query_vector=embeded_question, query_bm25_vector=bm25_embed, limit=20, prefetch_limit=40) # get the top 20 results from dense + bm25 query for reranking

    context = get_context(results_for_reranking.points) # get the context from the top 20 results
    response = get_response(query, context) # get the response from the gemini model
    return {"query": query, "response": response}

@app.get("/")
def root():
    return {"status": "ok", "collection_count": qdrant_client.count()}
