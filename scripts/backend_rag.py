from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
import requests

# Config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "medical_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3 # Return fewer, more relevant docs to the LLM
LLM_SERVER_URL = "http://127.0.0.1:8001/completion"

# FastAPI app
app = FastAPI(title="Medical RAG Backend")

# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:5173", # Default Vite dev server port
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model and Qdrant client at startup
print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)
print("Connecting to Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("Startup complete.")

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K

class DocResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict

class ChatResponse(BaseModel):
    query: str
    llm_answer: str
    results: List[DocResult]

PROMPT_TEMPLATE = """
You are a helpful, respectful and honest medical assistant. Answer the user's question based ONLY on the provided context.
If the context does not contain enough information to answer the question, state that you cannot answer based on the provided information.
Do not use any prior knowledge.

Context:
---
{context}
---

Question: {question}

Answer:
"""

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 1. Embed the query
    query_vec = model.encode(request.query, convert_to_numpy=True)
    
    # 2. Search Qdrant for relevant documents
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=request.top_k,
        with_payload=True
    )
    
    # 3. Format context for the LLM
    context = "\n\n---\n\n".join([hit.payload.get("text", "") for hit in search_result])
    
    # 4. Create the prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=request.query)

    # 5. Send prompt to the local LLM
    try:
        llm_response = requests.post(
            LLM_SERVER_URL,
            json={"prompt": prompt, "n_predict": 512, "stream": False, "temperature": 0.1}
        )
        llm_response.raise_for_status()
        llm_answer = llm_response.json().get("content", "").strip()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM server is unavailable: {e}")

    # 6. Format and return the final response
    results = [
        DocResult(
            id=str(hit.id),
            text=hit.payload.get("text", ""),
            score=hit.score,
            metadata=hit.payload
        )
        for hit in search_result
    ]
    return ChatResponse(query=request.query, llm_answer=llm_answer, results=results)

@app.get("/")
def root():
    return {"message": "Medical RAG Backend is running."} 