import openai
import os
from typing import List, Dict

class OpenAIClient:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's embedding model"""
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def get_completion(self, prompt: str, context: str) -> str:
        """Get completion from OpenAI with context"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content 