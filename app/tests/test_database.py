import pytest
import weaviate
from database import VectorDB
import os

@pytest.fixture
def vector_db():
    """Fixture to create a fresh VectorDB instance for each test"""
    db = VectorDB()
    
    # Clean up any existing schema
    try:
        db.client.schema.delete_class("Document")
    except:
        pass
    
    # Create fresh schema
    db._create_schema()
    
    yield db
    
    # Cleanup after test
    try:
        db.client.schema.delete_class("Document")
    except:
        pass

def test_create_schema(vector_db):
    """Test schema creation"""
    # Check if schema exists
    schema = vector_db.client.schema.get()
    classes = [class_obj['class'] for class_obj in schema['classes']]
    
    assert "Document" in classes

def test_add_documents(vector_db):
    """Test adding documents to the database"""
    documents = [
        {"content": "Test document 1", "metadata": "test1"},
        {"content": "Test document 2", "metadata": "test2"}
    ]
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    vector_db.add_documents(documents, embeddings)
    
    # Query to verify documents were added
    result = (
        vector_db.client.query
        .get("Document", ["content", "metadata"])
        .do()
    )
    
    added_docs = result["data"]["Get"]["Document"]
    assert len(added_docs) == 2
    assert any(doc["content"] == "Test document 1" for doc in added_docs)
    assert any(doc["content"] == "Test document 2" for doc in added_docs)

def test_search_similar(vector_db):
    """Test searching for similar documents"""
    # Add test documents
    documents = [
        {"content": "The cat sat on the mat", "metadata": "doc1"},
        {"content": "The dog played in the yard", "metadata": "doc2"}
    ]
    embeddings = [
        [1.0, 0.0, 0.0],  # First document embedding
        [0.0, 1.0, 0.0]   # Second document embedding
    ]
    
    vector_db.add_documents(documents, embeddings)
    
    # Search using first document's embedding
    query_embedding = [1.0, 0.0, 0.0]
    results = vector_db.search_similar(query_embedding, limit=1)
    
    assert len(results) == 1
    assert results[0]["content"] == "The cat sat on the mat"
    assert results[0]["metadata"] == "doc1"

def test_search_similar_with_limit(vector_db):
    """Test searching with different limit values"""
    documents = [
        {"content": "Doc 1", "metadata": "1"},
        {"content": "Doc 2", "metadata": "2"},
        {"content": "Doc 3", "metadata": "3"}
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0]
    ]
    
    vector_db.add_documents(documents, embeddings)
    
    # Search with limit=2
    results = vector_db.search_similar([1.0, 0.0, 0.0], limit=2)
    assert len(results) == 2

def test_add_documents_with_missing_metadata(vector_db):
    """Test adding documents with missing metadata"""
    documents = [
        {"content": "Test document"}  # No metadata
    ]
    embeddings = [[0.1, 0.2, 0.3]]
    
    vector_db.add_documents(documents, embeddings)
    
    result = (
        vector_db.client.query
        .get("Document", ["content", "metadata"])
        .do()
    )
    
    added_docs = result["data"]["Get"]["Document"]
    assert len(added_docs) == 1
    assert added_docs[0]["metadata"] == ""  # Default empty string

@pytest.mark.skip(reason="Only run when testing error handling")
def test_invalid_weaviate_url():
    """Test handling of invalid Weaviate URL"""
    os.environ["WEAVIATE_URL"] = "http://invalid-url:8080"
    
    with pytest.raises(weaviate.exceptions.WeaviateConnectionError):
        VectorDB() 