# Math RAG Agent

A self-hosted RAG (Retrieval-Augmented Generation) AI agent for answering mathematics questions.

## Features

- Support for K-12 to university-level mathematics
- LaTeX formula processing and validation
- Hybrid search combining vector and keyword matching
- Math-aware document retrieval
- Self-hosted infrastructure

## Installation

```bash
# Clone the repository
git clone https://github.com/yassineblaiech/Math_AI_agent
cd math_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .