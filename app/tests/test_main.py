import pytest
from fastapi.testclient import TestClient
import json
from main import app, vector_db, ai_client

client = TestClient(app)

# Sample test data
sample_documents = [
    {
        "content": "Test document 1",
        "metadata": "test1"
    },
    {
        "content": "Test document 2",
        "metadata": "test2"
    }
]

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Setup before and cleanup after each test"""
    # Setup - ensure we have a clean database
    try:
        vector_db.client.schema.delete_class("Document")
    except:
        pass
    vector_db._create_schema()
    
    yield
    
    # Cleanup after test
    try:
        vector_db.client.schema.delete_class("Document")
    except:
        pass

def test_add_documents_success():
    response = client.post(
        "/add-documents",
        json=[{"content": doc["content"], "metadata": doc["metadata"]} 
              for doc in sample_documents]
    )
    
    assert response.status_code == 200
    assert response.json() == {"message": "Successfully added 2 documents"}
    
    # Verify documents were actually added by querying one
    query_embedding = ai_client.get_embedding(sample_documents[0]["content"])
    similar_docs = vector_db.search_similar(query_embedding, limit=2)
    assert len(similar_docs) > 0
    assert any(doc["content"] == sample_documents[0]["content"] for doc in similar_docs)

def test_add_documents_with_invalid_data():
    response = client.post(
        "/add-documents",
        json=[{"content": "", "metadata": "test"}]  # Empty content should fail
    )
    
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_query_flow():
    # First add some documents
    client.post(
        "/add-documents",
        json=[{"content": doc["content"], "metadata": doc["metadata"]} 
              for doc in sample_documents]
    )
    
    # Then query
    response = client.post(
        "/query",
        json={"question": "What is document 1 about?"}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "answer" in result
    assert "similar_documents" in result
    assert len(result["similar_documents"]) > 0

def test_query_with_empty_database():
    response = client.post(
        "/query",
        json={"question": "test question"}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "answer" in result
    assert isinstance(result["similar_documents"], list) 