import logging
from datetime import datetime
import json
from typing import Dict, List, Optional
from .monitoring import MathAgentMonitor
import time

# Configure logging
logger = logging.getLogger('math_agent.rag')

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from .retriever import MathRetriever
from .validation import MathValidator

# System prompt to set the context and rules
SYSTEM_PROMPT = """You are an expert mathematics professor who excels at explaining complex concepts clearly.
Your responses should:
1. Begin with a clear, concise definition or overview
2. Present mathematical formulas using LaTeX notation ($...$ for inline, $$...$$for display)
3. Provide step-by-step explanations with clear reasoning
4. Include relevant examples when helpful
5. Cite sources when referencing theorems or proofs

If a question is unclear or lacks necessary information, explain what additional details are needed."""

# Main QA prompt with enhanced context handling
MATH_QA_PROMPT = """{system_prompt}

Below are relevant excerpts from mathematical texts:
{context}

Question: {question}

Please provide a complete answer following this structure:
1. Definition/Overview
2. Detailed Explanation
3. Key Formulas (in LaTeX)
4. Example (if applicable)
5. References

Answer:"""

# Followup prompt for additional context if needed
FOLLOWUP_PROMPT = """Based on the previous answer, would you like me to:
1. Provide more examples
2. Explain any specific step in more detail
3. Show an alternative approach
4. Provide related theorems or proofs"""

class MathRAGAgent:
    def __init__(self, retriever: MathRetriever, model_url: str = "http://localhost:8000"):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            openai_api_base=model_url,
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            temperature=0.3  # Lower temperature for more precise mathematical responses
        )
        self.prompt = PromptTemplate(
            template=MATH_QA_PROMPT,
            input_variables=["context", "question", "system_prompt"]
        )
        self.monitor = MathAgentMonitor()
        self.validator = MathValidator()
        logger.info("Initialized MathRAGAgent with %s model", model_url)
        
    def answer_question(self, 
                       question: str, 
                       use_hybrid_search: bool = True,
                       use_compression: bool = False,
                       k: int = 5) -> Dict:
        """Enhanced answer generation with monitoring and logging."""
        # Wrap the actual processing in the timer
        @self.monitor.time_query_processing
        def process_question():
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("Processing question [%s]: %s", session_id, question)
            
            try:
                start_time = time.time()
                
                # Log retrieval strategy
                logger.debug("Using retrieval strategy - hybrid: %s, compression: %s, k: %d",
                           use_hybrid_search, use_compression, k)
                
                # Use enhanced retrieval methods
                docs = self.retriever.hybrid_search(question, k=k)
                logger.debug("Retrieved %d documents using hybrid search", len(docs))
                    
                # Create QA chain with enhanced prompting
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=lambda x: docs,  # Use retrieved docs directly
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": self.prompt,
                        "verbose": True
                    }
                )
                
                # Get answer with enhanced context and system prompt
                logger.debug("Generating answer with QA chain")
                result = qa_chain({
                    "query": question,
                    "system_prompt": SYSTEM_PROMPT
                })
                
                # Add metadata about mathematical content
                math_metadata = self._extract_math_metadata(result["source_documents"])
                result["math_metadata"] = math_metadata
                logger.debug("Math metadata: %s", json.dumps(math_metadata))
                
                # Validate LaTeX syntax in the answer
                validation = self.validator.validate_latex_syntax(result["answer"])
                result["validation"] = validation
                
                # Log query and processing time
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                
                logger.info(
                    "Completed processing [%s] in %.2f seconds. Stats: %s",
                    session_id,
                    processing_time,
                    json.dumps({
                        "total_formulas": math_metadata["total_formulas"],
                        "has_equations": math_metadata["has_equations"],
                        "num_docs": len(result["source_documents"])
                    })
                )
                
                self.monitor.log_query(question, result)
                return result
                
            except Exception as e:
                logger.error(
                    "Error processing question [%s]: %s", 
                    session_id, 
                    str(e),
                    exc_info=True
                )
                self.monitor.log_error(e, {
                    "session_id": session_id,
                    "question": question,
                    "use_hybrid_search": use_hybrid_search,
                    "use_compression": use_compression
                })
                raise

        return process_question()
    
    def _extract_math_metadata(self, docs: List[Dict]) -> Dict:
        """Extract mathematical metadata from retrieved documents."""
        logger.debug("Extracting math metadata from %d documents", len(docs))
        
        total_formulas = 0
        has_equations = False
        
        for doc in docs:
            if doc.metadata.get("has_equations", False):
                has_equations = True
            total_formulas += doc.metadata.get("formula_count", 0)
            
        metadata = {
            "total_formulas": total_formulas,
            "has_equations": has_equations,
            "num_docs_with_equations": sum(
                1 for doc in docs if doc.metadata.get("has_equations", False)
            )
        }
        
        logger.debug("Extracted math metadata: %s", json.dumps(metadata))
        return metadata