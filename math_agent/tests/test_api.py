from fastapi.testclient import TestClient
from math_agent.test_fastapi import app
import pytest

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Math RAG Agent API with Deepseek Reasoner"

@pytest.mark.usefixtures("env_setup")
def test_ask_get_endpoint():
    response = client.get("/ask", params={"question": "What is the quadratic formula?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

@pytest.mark.usefixtures("env_setup")
def test_ask_post_endpoint():
    response = client.post(
        "/ask",
        json={"text": "What is the quadratic formula?", "max_tokens": 512}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()