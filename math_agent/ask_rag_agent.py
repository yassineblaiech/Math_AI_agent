import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from .utils.test import rag_agent
from .data_processing import MathCorpusProcessor
from .retriever import MathRetriever, CustomChatModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_clean_env_var(var_name: str) -> str:
    """Get environment variable with proper encoding handling."""
    if var := os.getenv(var_name):
        return var.strip(' \ufeff')  # Remove BOM and whitespace
    
    # Fallback to manual environment check
    for key, value in os.environ.items():
        if key.strip(' \ufeff') == var_name:
            return value.strip(' \ufeff')
    return None

def load_environment():
    """Load environment variables from .env file"""

    env_path = get_clean_env_var("RAGENVPATH")
    if not env_path:
        raise ValueError("RAGENVPATH environment variable not set")
    
    env_path = Path(env_path)
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    if not load_dotenv(dotenv_path=env_path):
        raise ValueError("Failed to load environment variables")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Ask a question to the Math RAG Agent')
    parser.add_argument('question', type=str, help='The question you want to ask')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maximum tokens in response')
    parser.add_argument('--show-sources', action='store_true', help='Show source documents')
    
    args = parser.parse_args()
    
    try:
        # Load environment variables
        load_environment()
        
        # Get API keys using get_clean_env_var instead of os.getenv
        deepseek_api_key = get_clean_env_var("DEEPSEEK_API_KEY")
        hf_api_key = get_clean_env_var("HF_API_KEY")
        
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        if not hf_api_key:
            raise ValueError("HF_API_KEY not found in environment variables")
        
        # Get answer from RAG agent
        answer, sources = rag_agent(
            question=args.question, 
            max_tokens=args.max_tokens
        )
        
        # Print results
        print("\nQuestion:", args.question)
        print("\nAnswer:", answer)
        
        if args.show_sources and sources:
            print("\nSources:")
            for i, source in enumerate(sources, 1):
                print(f"\n{i}. {source}")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())