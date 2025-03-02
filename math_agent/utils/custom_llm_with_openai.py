from openai import OpenAI
import os

def get_clean_env_var(var_name: str) -> str:
    if var := os.getenv(var_name):
        return var.strip(' \ufeff')
    for key, value in os.environ.items():
        if key.strip(' \ufeff') == var_name:
            return value.strip(' \ufeff')
    return None

def llm_agent():
    deepseek_api_key = get_clean_env_var("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    return client