from setuptools import setup, find_packages

setup(
    name="math_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "pydantic>=1.8.0",
        "uvicorn>=0.15.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-huggingface>=0.0.1",  # Add this
        "antlr4-python3-runtime",        # Add this for sympy
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "prometheus-client>=0.16.0",
        "python-multipart>=0.0.5",
        "sympy>=1.12",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "python-dotenv>=0.19.0",
        "accelerate>=0.26.0",
        "openai>=1.0.0",
        "wikipedia>=1.4.0",
        "bitsandbytes>=0.41.0",
        'rank_bm25>=0.2.2',
        "scipy>=1.9.0",
        "numpy>=1.21.0",  
        "rank_bm25>=0.2.2",  
        "tiktoken>=0.5.0", 
        "langchain-text-splitters>=0.0.1"
    ]
)