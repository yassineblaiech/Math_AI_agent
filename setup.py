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
        "langchain-community>=0.0.10",  # Add this
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "prometheus-client>=0.16.0",
        "python-multipart>=0.0.5",
        "sympy>=1.12",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0"
    ]
)