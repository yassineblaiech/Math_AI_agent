import os
import pytest
from dotenv import load_dotenv
from pathlib import Path
from math_agent.retriever import get_clean_env_var

@pytest.fixture(autouse=True)
def env_setup():
    """Load environment variables before each test"""
    # Get .env path from RAGENVPATH
    env_path = get_clean_env_var("RAGENVPATH")
    if not env_path:
        pytest.skip("RAGENVPATH environment variable not set")
    
    env_path = Path(env_path)
    if not env_path.exists():
        pytest.skip(f".env file not found at {env_path}")
    
    # Load environment variables from .env file
    if not load_dotenv(dotenv_path=env_path):
        pytest.skip("Failed to load environment variables")
    
    # Verify required env vars using get_clean_env_var
    deepseek_api_key = get_clean_env_var("DEEPSEEK_API_KEY")
    hf_api_key = get_clean_env_var("HF_API_KEY")
    
    if not deepseek_api_key or not hf_api_key:
        pytest.skip("Required API keys not found in environment")

    return {"deepseek_api_key": deepseek_api_key, "hf_api_key": hf_api_key}