from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from math_agent.utils.test import rag_agent  
from math_agent.data_processing import MathCorpusProcessor
from math_agent.retriever import MathRetriever
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = ROOT_DIR / "test_data"
if not TEST_DATA_DIR.exists():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created test data directory at {TEST_DATA_DIR}")

# Get the project root directory and .env file path
print ('root dir is',ROOT_DIR)

def get_clean_env_var(var_name: str) -> str:
    """Get environment variable with proper encoding handling."""
    if var := os.getenv(var_name):
        return var.strip(' \ufeff')  # Remove BOM and whitespace
    
    # Fallback to manual environment check
    for key, value in os.environ.items():
        if key.strip(' \ufeff') == var_name:
            return value.strip(' \ufeff')
    return None

# Get environment path
env_path = get_clean_env_var("RAGENVPATH")
if not env_path:
    raise ValueError("RAGENVPATH environment variable not set")

env_path = Path(env_path)
# Load environment variables
if not env_path.exists():
    raise FileNotFoundError(f".env file not found at {env_path}")

# Load the environment variables
get_env = load_dotenv(dotenv_path=env_path)

if not get_env:
    raise ValueError("Failed to load environment variables from .env file")

# Get API keys
deepseek_api_key = get_clean_env_var("DEEPSEEK_API_KEY")
hf_api_key = get_clean_env_var("HF_API_KEY")

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
if not hf_api_key:
    raise ValueError("HF_API_KEY not found in environment variables")

# Initialize Deepseek client
client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)

# Login to Hugging Face with the API token
if hf_api_key:
    login(token=hf_api_key)
else:
    raise ValueError("HF_API_KEY not found in environment variables")

# Before initializing the model, unset HF_TOKEN if it exists
if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]

# Then initialize the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', 
                           token=hf_api_key)  # Pass your Hugging Face token here

app = FastAPI(title="Math RAG Agent API")

class MathQuery(BaseModel):
    text: str
    max_tokens: int = 512
    include_sources: bool = True

class MathResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]]
    confidence: float
    reasoning: Optional[str]


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
processor = MathCorpusProcessor(str(TEST_DATA_DIR),hf_api_key)
vector_store = processor.process_documents()
retriever = MathRetriever(vector_store)

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "Math RAG Agent API with Deepseek Reasoner",
        "version": "1.0.0",
        "usage": {
            "POST /ask": "Send a math question in JSON format",
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
        # Get context from RAG
        docs = retriever.retrieve(query.text, k=5)
        context = "\n".join(doc.page_content for doc in docs)
        
        # Create message with context
        messages = [{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query.text}"
        }]
        
        # Call Deepseek Reasoner
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        
        # Debug the response
        print("Response type:", type(response))
        print("Response structure:", response)
        
        # Access response as dictionary
        if isinstance(response, dict):
            answer = response['choices'][0]['message']['content']
            reasoning = response['choices'][0]['message'].get('reasoning_content')
        else:
            # Access as object
            answer = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            
        return {
            "answer": answer,
            "reasoning": reasoning,
            "sources": [doc.metadata for doc in docs],
            "confidence": 0.95
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))