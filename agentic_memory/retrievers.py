from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
try:
    from .query_shaper import QueryShaper
except Exception:
    QueryShaper = None  # type: ignore

def simple_tokenize(text):
    return word_tokenize(text)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./memory_db", use_query_shaper: bool = True):
        """Initialize ChromaDB retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the embedding model
            persist_directory: Directory to persist ChromaDB data
        """
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            # Use ChromaDB's default embedding function instead of SentenceTransformer
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            # If there's an issue with the persistent client (e.g., corrupted data),
            # fall back to creating a new client
            # Issue with persistent ChromaDB client, recreating client
            
            # Remove corrupted data if it exists
            if os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
            
            # Create fresh client and collection
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Store configuration for later use
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        # Optional local query shaping (non-LLM)
        self.use_query_shaper = use_query_shaper
        self.query_shaper = (
            QueryShaper() if (use_query_shaper and 'QueryShaper' in globals() and QueryShaper is not None) else None
        )
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
                
        self.collection.add(
            documents=[document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5, where: dict = None):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            where: Optional filter conditions for metadata
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        # Optionally shape the query locally to improve recall for story continuation
        query_text = query
        if getattr(self, "query_shaper", None) is not None:
            try:
                shaped = self.query_shaper.shape(query)
                semantic_query = shaped.get("semantic_query") if isinstance(shaped, dict) else None
                if semantic_query:
                    query_text = str(semantic_query)
            except Exception:
                # Fallback to original query if shaping fails
                pass

        query_params = {
            "query_texts": [query_text],
            "n_results": k
        }
        
        if where:
            query_params["where"] = where
            
        results = self.collection.query(**query_params)
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass
                        
        return results
