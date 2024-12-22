from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
from database import VectorDB
from utils import OpenAIClient

app = FastAPI()
vector_db = VectorDB()
ai_client = OpenAIClient()

class Document(BaseModel):
    content: str
    metadata: Optional[str] = ""

class Query(BaseModel):
    question: str

@app.post("/add-documents")
async def add_documents(documents: List[Document]):
    try:
        # Convert documents to the format expected by VectorDB
        docs = [{"content": doc.content, "metadata": doc.metadata} for doc in documents]
        
        # Get embeddings for all documents
        embeddings = [ai_client.get_embedding(doc.content) for doc in documents]
        
        # Add documents and embeddings to vector database
        vector_db.add_documents(docs, embeddings)
        
        return {"message": f"Successfully added {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(query: Query):
    try:
        # Get embedding for the query
        query_embedding = ai_client.get_embedding(query.question)
        
        # Search for similar documents
        similar_docs = vector_db.search_similar(query_embedding)
        
        # Create context from similar documents
        context = "\n".join([doc["content"] for doc in similar_docs])
        
        # Get completion from OpenAI
        answer = ai_client.get_completion(query.question, context)
        
        return {
            "answer": answer,
            "similar_documents": similar_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 