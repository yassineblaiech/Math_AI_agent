from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import FAISS
from typing import List, Dict, Optional
import numpy as np
import torch
from langchain.base_language import BaseLanguageModel
from langchain.llms import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, ChatMessage, HumanMessage, SystemMessage

class CustomChatModel(BaseChatModel):
    """Custom chat model that forwards requests to local FastAPI endpoint."""
    
    def __init__(self, base_url: str):
        """Initialize with base URL for API endpoint."""
        super().__init__()
        self._base_url = base_url

    @property
    def _llm_type(self) -> str:
        """Return identifier for the LLM type"""
        return "custom_chat_model"
        
    def _generate(self, messages, stop=None, run_manager=None):
        """Generate response from messages using FastAPI endpoint."""
        import requests
        
        # Extract the text from the last human message
        text = messages[-1].content if messages else ""
        
        # Forward to local API using /ask endpoint instead of /generate
        response = requests.post(
            f"{self._base_url}/ask",
            json={"text": text, "max_tokens": 512}
        )
        response.raise_for_status()
        
        # Return AIMessage with the generated text
        return AIMessage(content=response.json()["answer"])

class MathRetriever:
    def __init__(self, vector_store: FAISS, model_url: str = "http://localhost:8000"):
        self.vector_store = vector_store
        
        # Use similarity_search to get all documents instead of get_all_documents
        all_docs = self.vector_store.similarity_search("", k=10000)  # Get all docs
        
        # Initialize BM25 retriever with documents
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        
        # Use custom chat model instead of ChatOpenAI
        self.llm = CustomChatModel(base_url=model_url)
        
        # Initialize other components
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        
    def hybrid_search(self, query: str, k: int = 5, 
                     vector_weight: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search using both vector and BM25 similarity.
        """
        vector_docs = self.vector_store.similarity_search_with_score(query, k=k)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # Normalize scores
        vector_scores = {doc.id: score for doc, score in vector_docs}
        bm25_scores = {doc.id: score for doc, score in bm25_docs}
        
        # Combine scores
        combined_docs = {}
        for doc, score in vector_docs + bm25_docs:
            if doc.id in combined_docs:
                continue
            combined_score = (
                vector_weight * vector_scores.get(doc.id, 0) +
                (1 - vector_weight) * bm25_scores.get(doc.id, 0)
            )
            combined_docs[doc.id] = (doc, combined_score)
            
        # Sort by combined score
        sorted_docs = sorted(
            combined_docs.values(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        return [doc for doc, _ in sorted_docs]
    
    def math_aware_retrieval(self, query: str, k: int = 5) -> List[Dict]:
        """
        Perform math-aware retrieval prioritizing documents with relevant formulas.
        """
        docs = self.hybrid_search(query, k=k*2)  # Get more docs initially
        
        # Filter and re-rank based on mathematical content
        math_scores = []
        for doc in docs:
            score = 0
            if doc.metadata.get("has_equations", False):
                score += 1
            score += min(doc.metadata.get("formula_count", 0) * 0.2, 2)
            math_scores.append(score)
            
        # Combine with original scores
        final_scores = np.array(math_scores)
        ranked_indices = np.argsort(-final_scores)
        
        return [docs[i] for i in ranked_indices[:k]]
    
    def retrieve_with_compression(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve documents and compress them to extract relevant passages.
        """
        docs = self.math_aware_retrieval(query, k=k)
        compressed_docs = self.compressor.compress_documents(docs, query)
        return compressed_docs
    
    def retrieve(self, query: str, k: int = 5, 
                use_compression: bool = False) -> List[Dict]:
        """
        Main retrieval method with options for different retrieval strategies.
        """
        if use_compression:
            return self.retrieve_with_compression(query, k)
        return self.math_aware_retrieval(query, k)
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Retrieve documents with their relevance scores.
        """
        return self.vector_store.similarity_search_with_scores(query, k=k)