from langchain.llms.base import LLM
from typing import Optional, List, Dict, Any
import requests

class CustomLLM(LLM):
    endpoint: str = "http://localhost:8000/generate"
    max_length: int = 512

    @property
    def _llm_type(self) -> str:
        return "custom-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            self.endpoint,
            json={"prompt": prompt, "max_length": self.max_length}
        )
        return response.json()["response"]

# Initialize the LLM
custom_llm = CustomLLM()