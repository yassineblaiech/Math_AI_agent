# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Optional
from math_agent import rag_agent  # Now this will work

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
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

@app.post("/ask", response_model=MathResponse)
async def ask_question(query: MathQuery):
    try:
        answer, sources = rag_agent(query.text)
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in sources],
            "confidence": 0.95
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))