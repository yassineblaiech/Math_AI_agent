from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Optional, Any
import numpy as np
from langchain.base_language import BaseLanguageModel
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, ChatMessage, HumanMessage, SystemMessage, ChatResult, ChatGeneration, BaseMessage
from openai import OpenAI
from pydantic import Field
import os
from langchain.callbacks.manager import CallbackManagerForLLMRun

def get_clean_env_var(var_name: str) -> str:
    """Get environment variable with proper encoding handling."""
    if var := os.getenv(var_name):
        return var.strip(' \ufeff')  # Remove BOM and whitespace
    
    # Fallback to manual environment check
    for key, value in os.environ.items():
        if key.strip(' \ufeff') == var_name:
            return value.strip(' \ufeff')
    return None

class CustomChatModel(BaseChatModel):
    client: OpenAI = Field(default=None)
    max_tokens: Optional[int] = Field(default=512)

    def __init__(self, api_key: str, max_tokens: int = 512):
        """Initialize with API key for the service."""
        super().__init__()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "custom_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion with proper result structure."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        formatted_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                role = "system"
            formatted_messages.append({"role": role, "content": m.content})

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=formatted_messages,
                max_tokens=self.max_tokens,
                **kwargs
            )

            message = AIMessage(content=response.choices[0].message.content)
            generation = ChatGeneration(
                message=message,
                generation_info={
                    "finish_reason": response.choices[0].finish_reason,
                    "logprobs": None
                }
            )

            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"Error in chat completion: {str(e)}")
            raise

class MathRetriever:
    def __init__(self, vector_store: FAISS, model_url: str = "http://localhost:8000"):
        self.vector_store = vector_store
        
        # Use similarity_search to get all documents instead of get_all_documents
        all_docs = self.vector_store.similarity_search("", k=10000)  # Get all docs
        
        # Initialize BM25 retriever with documents
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        
        # Get API key properly
        deepseek_api_key = get_clean_env_var("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        # Initialize custom chat model with API key and max_tokens
        self.llm = CustomChatModel(api_key=deepseek_api_key, max_tokens=512)
        
        # Initialize compressor
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        
    def hybrid_search(self, query: str, k: int = 5, vector_weight: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search using both vector and BM25 similarity.
        """
        # Get vector search results with scores
        vector_docs = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Get BM25 results using invoke instead of deprecated get_relevant_documents
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # Create score dictionaries with fallback IDs
        vector_scores = {doc.metadata.get('id', f'v_{i}'): score 
                        for i, (doc, score) in enumerate(vector_docs)}
        bm25_scores = {doc.metadata.get('id', f'b_{i}'): 1.0 
                    for i, doc in enumerate(bm25_docs)}
        
        # Get unique document IDs
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        # Combine scores with safe document retrieval
        combined_scores = []
        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            
            # Normalize and combine scores
            combined_score = (vector_weight * v_score + 
                            (1 - vector_weight) * b_score)
            
            # Get the document with safe fallback
            doc = None
            if doc_id.startswith('v_'):
                doc = next((d for d, _ in vector_docs if d.metadata.get('id', f'v_{doc_id}') == doc_id), None)
            else:
                doc = next((d for d in bm25_docs if d.metadata.get('id', f'b_{doc_id}') == doc_id), None)
                
            if doc:
                combined_scores.append((doc, combined_score))
        
        # Sort by score and return top k documents
        sorted_docs = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:k]
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
    
    def retrieve(self, query: str, k: int = 5, use_compression: bool = False) -> List[Dict]:
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
        return self.vector_store.similarity_search_with_score(query, k=k)
