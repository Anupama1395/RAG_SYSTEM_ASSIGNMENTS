import os
import uuid
import math
import requests
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Assignment 3 - RAG Q&A Service")

# -----------------------------
# Hugging Face / Mistral config
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Export it before running the app.")

HF_MODEL = os.environ.get("HF_MODEL", "gpt-4-turbo")
HF_API_URL = "https://api.openai.com/v1/chat/completions"

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = (
    "You are a helpful CS teaching assistant. "
    "Answer using only the retrieved context. "
    "If the answer is not in the context, say: "
    "'I could not find that in the ingested documents.' "
    "Be concise and accurate."
)

# -----------------------------
# Embedding model

# Free local embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# In-memory stores

# sessions[session_id] = [{"role": "...", "content": "..."}]
sessions: Dict[str, List[Dict[str, str]]] = {}

# chunk_store: list of dicts
# each item:
# {
#   "chunk_id": "doc1#0",
#   "doc_id": "doc1",
#   "text": "...",
#   "embedding": [float, float, ...]
# }
chunk_store: List[Dict[str, Any]] = []



# Pydantic models

class SessionResponse(BaseModel):
    session_id: str


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("doc_id cannot be empty")
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text cannot be empty")
        return v


class QARequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty")
        return v


# -----------------------------
# Utility functions
# -----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Fixed-length chunking with overlap.
    This uses character-based chunking, which is fine for the assignment.
    """
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += chunk_size - overlap

    return chunks


def embed_text(text: str) -> List[float]:
    """
    Return embedding as a Python list.
    """
    embedding = embedding_model.encode(text, normalize_embeddings=False)
    return embedding.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Manual cosine similarity implementation, as required.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


def retrieve_top_k(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Embed query, compute cosine similarity against all stored chunks,
    return top-k chunks sorted by descending score.
    """
    if not chunk_store:
        return []

    query_embedding = embed_text(query)
    scored = []

    for chunk in chunk_store:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "score": round(float(score), 6),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def call_hf_inference(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 350,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json=payload,
            timeout=60
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Hugging Face request failed: {e}")

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=response.status_code,
            detail="HF auth failed or model is gated."
        )
    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="HF quota/rate limit exceeded.")
    if response.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is loading. Retry in a minute.")
    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"HF error: {response.status_code} {response.text}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=500, detail=f"Unexpected HF response format: {data}")


def build_grounded_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        context_block = "No relevant context was retrieved."
    else:
        context_parts = []
        for item in retrieved_chunks:
            context_parts.append(
                f"[{item['chunk_id']}] (score={item['score']})\n{item['text']}"
            )
        context_block = "\n\n".join(context_parts)

    prompt = (
        "Use the retrieved context below to answer the question.\n\n"
        "Rules:\n"
        "1. Answer only from the retrieved context.\n"
        "2. If the answer is not present, say you could not find it in the ingested documents.\n"
        "3. Do not invent facts.\n"
        "4. When useful, mention the chunk IDs you relied on.\n\n"
        f"Retrieved Context:\n{context_block}\n\n"
        f"Question: {question}"
    )
    return prompt


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def read_root():
    return {
        "message": "Assignment 3 RAG Q&A Service is running",
        "model": HF_MODEL,
        "chunks_in_memory": len(chunk_store),
        "sessions_in_memory": len(sessions),
    }


@app.post("/session", response_model=SessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return {"session_id": session_id}


@app.post("/ingest")
def ingest_document(request: IngestRequest):
    # Optional behavior: remove old chunks for same doc_id before re-ingesting
    global chunk_store
    chunk_store = [c for c in chunk_store if c["doc_id"] != request.doc_id]

    chunks = chunk_text(request.text, chunk_size=800, overlap=150)

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks could be created from the document text.")

    for i, chunk in enumerate(chunks):
        chunk_id = f"{request.doc_id}#{i}"
        embedding = embed_text(chunk)

        chunk_store.append({
            "chunk_id": chunk_id,
            "doc_id": request.doc_id,
            "text": chunk,
            "embedding": embedding,
        })

    return {
        "doc_id": request.doc_id,
        "chunks_added": len(chunks),
    }


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, max_length=2000),
    k: int = Query(default=3, ge=1, le=10)
):
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not chunk_store:
        return {
            "query": query,
            "results": []
        }

    results = retrieve_top_k(query=query, k=k)

    return {
        "query": query,
        "results": [
            {
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "text": item["text"],
            }
            for item in results
        ]
    }


@app.post("/qa")
def qa(request: QARequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if not chunk_store:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    retrieved = retrieve_top_k(request.question, k=request.k)
    grounded_prompt = build_grounded_prompt(request.question, retrieved)

    # Keep short history for practicality
    history = sessions[request.session_id][-6:]
    messages = history + [{"role": "user", "content": grounded_prompt}]

    try:
        answer = call_hf_inference(messages)
        if not answer:
            raise HTTPException(status_code=500, detail="Empty response from model.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")

    # Store only the natural question + final answer in session history
    sessions[request.session_id].append({"role": "user", "content": request.question})
    sessions[request.session_id].append({"role": "assistant", "content": answer})

    turn_count = len([m for m in sessions[request.session_id] if m["role"] == "user"])

    return {
        "answer": answer,
        "citations": [
            {
                "chunk_id": item["chunk_id"],
                "score": item["score"]
            }
            for item in retrieved
        ],
        "turn_count": turn_count
    }