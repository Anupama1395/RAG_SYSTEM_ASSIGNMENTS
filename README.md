# Assignment 3: RAG Q&A Service
# Overview

In this assignment, I built a Retrieval-Augmented Generation (RAG) Q&A system using FastAPI.

The goal of this project is to:

## Ingest documents

Break them into smaller chunks

Retrieve the most relevant chunks using semantic similarity

Generate answers based only on the retrieved content

Provide citations for transparency

This helps reduce hallucinations and makes the system more reliable.

 ## What the System Does

The system works in three main steps:

Ingestion
Takes a document, splits it into chunks, and stores embeddings.

Search
Finds the most relevant chunks using cosine similarity.

Question Answering (QA)
Uses retrieved chunks + LLM to generate grounded answers with citations.

## Tech Stack

FastAPI

Pydantic

SentenceTransformers (for embeddings)

## OpenAI API (for answer generation)

Python

 How to Run the Project
1. Activate environment
source /Users/anupama/llm_api/venv/bin/activate
2. Set API key
export HF_TOKEN= openai_api_key
3. Run the server
cd /Users/anupama
uvicorn app:app
# API Usage
1. Create a session
curl -X POST "http://127.0.0.1:8000/session"
2. Ingest a document
curl -X POST "http://127.0.0.1:8000/ingest" \
-H "Content-Type: application/json" \
-d '{"doc_id":"doc1","text":"This is a test document."}'
3. Search (debugging step)
curl "http://127.0.0.1:8000/search?query=test&k=3"
4. Ask a question
curl -X POST "http://127.0.0.1:8000/qa" \
-H "Content-Type: application/json" \
-d '{"session_id":"<6c20d616-d892-4a75-8701-445e5d75f928>", "question":"What is the document about?", "k":1}'
 
 # Key Features

Chunking with overlap for better context retention

Embedding generation using SentenceTransformers

Manual cosine similarity implementation

Top-k retrieval of relevant chunks

Grounded prompt construction

Session-based conversation handling

Answers include citations for transparency

# Testing
## Manual Testing

Verified that ingestion correctly creates chunks

Checked that /search returns relevant results

Ensured QA answers are based only on ingested content

Tested behavior for out-of-scope questions

Suggested Automated Tests

Chunking correctness

Cosine similarity ranking

Endpoint integration tests

 # Mock LLM responses

 Example Output
{
  "answer": "This document is about a test document.",
  "citations": [
    {"chunk_id": "doc1#0", "score": 0.54}
  ],
  "turn_count": 1
}
 # Reflection
7. Why does grounding reduce hallucinations?

Grounding reduces hallucinations because the model is forced to answer only using the retrieved context instead of relying on its own general knowledge.

8. How do chunk size and overlap affect retrieval quality?

Smaller chunks give more precise matches but may lose context. Larger chunks preserve context but may include irrelevant information. Overlap helps maintain continuity between chunks.

9. What is the difference between semantic search and keyword search?

Keyword search looks for exact word matches, while semantic search understands the meaning of the query and retrieves conceptually similar results.

10. What are common failure modes of RAG systems?

Some common issues include:

Poor chunking strategy

Irrelevant retrieval

Model ignoring the provided context

No relevant chunks found

11. Why are citations important in AI systems?

Citations improve trust and transparency. They allow users to verify where the answer came from and understand how the model arrived at its response.

 # Final Thoughts

This assignment helped me understand how retrieval and generation can be combined to build more reliable AI systems. It also highlighted the importance of grounding, evaluation, and transparency in LLM-based applications.