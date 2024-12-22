import weaviate
import os
from typing import List, Dict

class VectorDB:
    def __init__(self):
        self.client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL"),
        )
        self._create_schema()

    def _create_schema(self):
        """Create the schema if it doesn't exist"""
        schema = {
            "class": "Document",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                },
                {
                    "name": "embedding",
                    "dataType": ["number[]"],
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],
                }
            ]
        }
        
        try:
            self.client.schema.create_class(schema)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            pass

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """Add documents and their embeddings to the vector store"""
        with self.client.batch as batch:
            for doc, embedding in zip(documents, embeddings):
                batch.add_data_object(
                    data_object={
                        "content": doc["content"],
                        "metadata": doc.get("metadata", ""),
                        "embedding": embedding
                    },
                    class_name="Document"
                )

    def search_similar(self, query_embedding: List[float], limit: int = 3):
        """Search for similar documents using vector similarity"""
        result = (
            self.client.query
            .get("Document", ["content", "metadata"])
            .with_near_vector({
                "vector": query_embedding,
            })
            .with_limit(limit)
            .do()
        )
        
        return result["data"]["Get"]["Document"] 