# Math RAG Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized Retrieval-Augmented Generation system for mathematical reasoning, combining DeepSeek Reasoner with hybrid search techniques for enhanced formula understanding and problem solving.

## Key Features

- **Hybrid Search Engine**  
    Combines vector similarity (FAISS) and BM25 text search with math-aware scoring
- **Formula-Priority Retrieval**  
    Specialized weighting for documents containing mathematical equations
- **Structured Responses**  
    Organized output with definitions, explanations, LaTeX formulas, and examples
- **Conversational Follow-ups**  
    Maintains context for iterative questioning
- **Document Compression**  
    Focused context extraction from relevant passages

## Architecture Overview

### Core Components

| Component | Description | Key Features |
|-----------|-------------|--------------|
| `MathRAGAgent` ([rag_agent.py](math_agent/rag_agent.py)) | Main query handler | Structured responses, follow-up support, response validation |
| `MathRetriever` ([retriever.py](math_agent/retriever.py)) | Hybrid search system | Vector+BM25 fusion, formula scoring, document compression |

### Model Integration

| Component | Technology | Description |
|-----------|-------------|-------------|
| Primary LLM | DeepSeek Reasoner | Optimized for mathematical reasoning |
| Base Model | Mistral-7B-Instruct-v0.1 | Foundation for custom implementations |
| Embeddings | sentence-transformers | Text vectorization for semantic search |

### Search Algorithm

```python
# Hybrid scoring formula
final_score = vector_weight * vector_score + (1 - vector_weight) * bm25_score
# With math-boost adjustment for formula-containing documents
```

## Installation

### Requirements
- Python 3.9+
- FAISS (CPU/GPU version)
- CUDA 11.7+ (for GPU acceleration)

```bash
# Create virtual environment
python -m venv math_rag_env
source math_rag_env/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install specialized packages
pip install faiss-cpu sentence-transformers transformers
```

## Environment Setup
```bash
# .env.example
MODEL_PATH="deepseek-ai/deepseek-math-7b-r"
EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
FAISS_INDEX_PATH="/path/to/faiss_index"
BM25_INDEX_PATH="/path/to/bm25_index"
```

## Usage Examples

### Basic Query
```python
from math_agent.ask_rag_agent import rag_agent

response, sources = rag_agent("Explain Euler's formula with examples")
```

### Sample Response Structure:
```json
{
    "definition": "Fundamental relationship between trigonometric functions...",
    "formula": "e^{i\\theta} = \\cos\\theta + i\\sin\\theta",
    "examples": [
        {"input": "θ = π", "calculation": "e^{iπ} = -1"},
        {"input": "θ = 0", "result": "e^{0} = 1"}
    ],
    "sources": ["advanced_calculus.pdf", "complex_analysis.md"]
}
```

## API Endpoints

```bash
# Start FastAPI server
uvicorn math_agent.test_fastapi:app --reload --port 8000
```

| Endpoint | Method | Parameters | Description |
|----------|---------|------------|-------------|
| /ask | GET | query: str | Text-based question answering |
| /ask | POST | JSON: {"question": "...", "history": []} | Context-aware queries |

### Example API Request:
```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Derive the quadratic formula", "history": []}'
```

## Dependencies

| Category | Packages |
|----------|----------|
| Core ML | transformers, torch, sentence-transformers |
| Search | faiss-cpu, rank-bm25 |
| API | fastapi, uvicorn, pydantic |
| Utilities | python-dotenv, loguru, tiktoken |

## Contributing

1. Fork the repository
2. Create feature branch (git checkout -b feature/improvement)
3. Commit changes (git commit -am 'Add new feature')
4. Push to branch (git push origin feature/improvement)
5. Open Pull Request