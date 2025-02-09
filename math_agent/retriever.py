from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Optional
import numpy as np

class MathRetriever:
    def __init__(self, vector_store: FAISS, model_url: str = "http://localhost:8000"):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            openai_api_base=model_url,
            model_name="mistralai/Mistral-7B-Instruct-v0.1"
        )
        
        # Initialize BM25 retriever for hybrid search
        self.bm25_retriever = BM25Retriever.from_documents(
            self.vector_store.get_all_documents()
        )
        
        # Initialize LLM chain extractor for document compression
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