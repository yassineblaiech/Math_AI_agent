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
```

⚠️ **IMPORTANT:** Add .env to your Environment variables with the name RAGENVPATH. This step is crucial for the application to function properly.
```ini
RAGENVPATH=path_to_your_.env_file
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

### Method 1: Direct Agent Interaction

#### Basic Query
```python
from math_agent.ask_rag_agent import rag_agent

response, sources = rag_agent("Explain Euler's formula with examples")
```

#### Sample Response Structure:
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

### Method 2: Command Line Interface

Use the CLI for quick mathematical queries directly from your terminal:

```bash
python -m math_agent.ask_rag_agent "What is the quadratic formula?"
```

#### Sample Terminal Output:
```
2025-03-02 01:23:23,003 - httpx - INFO - HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"
Generated search query: "Quadratic formula"
Found 10 search results
Loaded document: Quadratic formula
Loaded document: Quadratic equation
Loaded document: Solution in radicals

Question: What is the quadratic formula?

Answer: ### Quadratic Formula: Complete Explanation

**1. Definition/Overview**
The quadratic formula provides the solutions to any quadratic equation of the form
$$ax^2 + bx + c = 0 \quad (a \neq 0).$$

**2. Mathematical Formula**
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

**3. Step-by-Step Derivation**
1. Start with standard form: $ax^2 + bx + c = 0$
2. Divide by $a$ (≠ 0): $x^2 + \frac{b}{a}x + \frac{c}{a} = 0$
3. Complete the square...
[...output truncated for brevity...]
```

This method is particularly useful for:
- Quick mathematical lookups
- Terminal-based workflows
- Scripting and automation
- Teaching and demonstration purposes

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
`````
