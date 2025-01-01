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
    # First add some documents with more meaningful content
    test_documents = [
        {
            "content": "The capital city of France is Paris. It is known for the Eiffel Tower.",
            "metadata": "test1"
        },
        {
            "content": "The capital city of Japan is Tokyo. It is known for its modern technology.",
            "metadata": "test2"
        }
    ]
    
    client.post(
        "/add-documents",
        json=[{"content": doc["content"], "metadata": doc["metadata"]} 
              for doc in test_documents]
    )
    
    # Then query with a more specific question
    response = client.post(
        "/query",
        json={"question": "What is the capital city of France and what is it known for?"}
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

def test_query_personal_information():
    # Add documents with personal/biographical information
    personal_documents = [
        {
            "content": "John Smith is a software engineer who graduated from MIT in 2015. He specializes in Python and machine learning.",
            "metadata": "bio1"
        },
        {
            "content": "John's current project involves developing an AI-powered chatbot for customer service. He has been working on this since 2022.",
            "metadata": "bio2"
        }
    ]
    
    # Add the documents to the vector database
    client.post(
        "/add-documents",
        json=[{"content": doc["content"], "metadata": doc["metadata"]} 
              for doc in personal_documents]
    )
    
    # Test different types of personal queries
    queries = [
        {
            "question": "What is John's educational background?",
            "expected_content": ["MIT", "2015"]
        },
        {
            "question": "What is John currently working on?",
            "expected_content": ["chatbot", "customer service"]
        }
    ]
    
    for query in queries:
        response = client.post("/query", json={"question": query["question"]})
        
        assert response.status_code == 200
        result = response.json()
        assert "answer" in result
        assert "similar_documents" in result
        assert len(result["similar_documents"]) > 0
        
        # Verify that the response contains relevant information
        answer = result["answer"].lower()
        assert any(expected.lower() in answer for expected in query["expected_content"]), \
            f"Expected content not found in answer: {answer}" 