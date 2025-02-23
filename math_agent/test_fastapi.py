from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login, HfFolder
from typing import List, Optional
from math_agent.utils.test import rag_agent  # Updated import
from math_agent.data_processing import MathCorpusProcessor
from math_agent.retriever import MathRetriever
import os
from dotenv import load_dotenv
from pathlib import Path
import logging



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = ROOT_DIR / "test_data"
if not TEST_DATA_DIR.exists():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created test data directory at {TEST_DATA_DIR}")
    
# Get the project root directory and .env file path
ROOT_DIR = Path(__file__).resolve().parent.parent
print ('root dir is',ROOT_DIR)
env_path = ROOT_DIR / ".env"
print('env path is',env_path)

# Load environment variables
if not env_path.exists():
    raise FileNotFoundError(f".env file not found at {env_path}")

# Load the environment variables
load_dotenv(dotenv_path=env_path)


app = FastAPI(title="Math RAG Agent API")

class MathQuery(BaseModel):
    text: str
    max_tokens: int = 512
    include_sources: bool = True

class MathResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]]
    confidence: float

# Initialize models
hf_token = "your_hf_token"


model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name,token=hf_token, device_map="cpu",  # Force CPU usage
    torch_dtype=torch.float32)


sample_content = """
# Basic Mathematical Concepts

## Quadratic Formula
The quadratic formula is used to solve quadratic equations of the form $ax^2 + bx + c = 0$
The solution is given by:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

## Pythagorean Theorem
For a right triangle with sides a, b, and c, where c is the hypotenuse:
$$a^2 + b^2 = c^2$$
"""

# Write the content
with open(TEST_DATA_DIR / "sample_math.txt", "w", encoding="utf-8") as f:
    f.write(sample_content)
    
# Initialize RAG components
processor = MathCorpusProcessor(str(TEST_DATA_DIR))
vector_store = processor.process_documents()
retriever = MathRetriever(vector_store)



@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "Math RAG Agent API",
        "version": "1.0.0",
        "usage": {
            "POST /ask": "Send a math question in JSON format",
            "GET /ask": "Test endpoint with query parameter",
            "GET /docs": "API documentation"
        }
    }

@app.get("/ask")
async def ask_question_get(question: str):
    """
    Test endpoint for asking math questions using GET request.
    
    Args:
        question (str): The math question (as query parameter)
    """
    try:
        answer, sources = rag_agent(question)
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in sources],
            "confidence": 0.95
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=MathResponse)
async def ask_question(query: MathQuery):
    try:
        # Call rag_agent as a function
        result = rag_agent(query.text)
        
        # Unpack the tuple return value
        answer, sources = result
        
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in sources],
            "confidence": 0.95
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))