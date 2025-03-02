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
- Git
- Visual Studio Build Tools (for FAISS installation)
- CUDA 11.7+ (optional, for GPU acceleration)

### Setup Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/math_agent.git
cd math_agent
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate
```

3. **Install Dependencies**
```bash
# Install PyTorch first (CPU version)
pip install torch torchvision torchaudio

# Install core requirements
pip install -e .

```

4. **Environment Configuration**
```bash
# Create .env file
copy .env.example .env

# Edit .env with your settings
notepad .env
```

Example `.env` configuration:
```ini
DEEPSEEK_API_KEY=your_deepseek_api_key
HF_API_KEY=your_huggingface_api_key
MODEL_PATH=deepseek-ai/deepseek-math-7b-r
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
FAISS_INDEX_PATH=./data/faiss_index
BM25_INDEX_PATH=./data/bm25_index
```

5. **Verify Installation**
```bash
# Run tests
python -m pytest tests/

# Start the API server
uvicorn math_agent.test_fastapi:app --reload --host 0.0.0.0 --port 8000
```

### Troubleshooting

- If FAISS installation fails:
  ```bash
  # Try installing wheel directly
  pip install --no-cache-dir faiss-cpu
  ```

- If you encounter DLL load errors:
  1. Install Visual C++ Redistributable
  2. Add Python Scripts directory to PATH

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
````
