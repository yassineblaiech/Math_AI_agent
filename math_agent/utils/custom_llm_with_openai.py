from langchain.chat_models import ChatOpenAI

def llm_agent():
    
    llm = ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:8000/v1",
        model_name="mistralai/Mistral-7B-Instruct-v0.1"
    )
    return llm