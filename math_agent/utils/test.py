from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from ..retriever import CustomChatModel
import wikipedia
import os
from openai import OpenAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv


def get_clean_env_var(var_name: str) -> str:
    """Get environment variable with proper encoding handling."""
    if var := os.getenv(var_name):
        return var.strip(' \ufeff')  # Remove BOM and whitespace
    
    # Fallback to manual environment check
    for key, value in os.environ.items():
        if key.strip(' \ufeff') == var_name:
            return value.strip(' \ufeff')
    return None

def process_docs(question, model_url: str = "http://localhost:8000"):
    """Process documents for RAG retrieval"""
    # Load environment variables first
    env_path = get_clean_env_var("RAGENVPATH")
    if not env_path:
        raise ValueError("RAGENVPATH environment variable not set")
    
    env_path = Path(env_path)
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    # Load the environment variables from .env
    if not load_dotenv(dotenv_path=env_path):
        raise ValueError("Failed to load environment variables from .env file")
    
    # Now get API keys after loading .env
    hf_api_key = get_clean_env_var("HF_API_KEY")
    if not hf_api_key:
        raise ValueError("HF_API_KEY not found in environment variables")
    
    deepseek_api_key = get_clean_env_var("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    # Initialize custom model with API key
    llm = CustomChatModel(api_key=deepseek_api_key)
    
    # Create prompt and generate search query using new pattern
    prompt = PromptTemplate.from_template(
        """Given the following question, generate a concise search query to find relevant Wikipedia articles.
        Question: {question}
        Search Query:"""
    )
    
    # Use the new invoke pattern instead of deprecated run
    messages = [HumanMessage(content=prompt.format(question=question))]
    response = llm.invoke(messages)
    search_query = response.content.strip()
    # Clean up the search query if it contains "Search Query:"
    search_query = search_query.replace("Search Query:", "").strip()
    
    # Add debug logging
    print(f"Generated search query: {search_query}")
    
    try:
        search_results = wikipedia.search(search_query)
        print(f"Found {len(search_results)} search results")
    except wikipedia.exceptions.DisambiguationError as e:
        search_results = e.options[:3]
        print(f"Disambiguated to {len(search_results)} options")
    except Exception as e:
        print(f"Error in Wikipedia search: {e}")
        search_results = []
    
    # In process_docs function
    if len(search_results) == 0:
        # Try with a more direct search
        try:
            search_results = [question]  # Try the question itself
            print("No results found, trying direct question as query")
        except Exception as e:
            print(f"Direct query failed: {e}")
            search_results = []
    
    docs = []
    for title in search_results[:3]:
        try:
            loader = WikipediaLoader(query=title, load_max_docs=1)
            new_docs = loader.load()
            docs.extend(new_docs)
            print(f"Loaded document: {title}")
        except Exception as e:
            print(f"Error loading {title}: {e}")
    
    if not docs:
        print("No documents found. Using fallback content.")
        docs = [{
            "page_content": f"""Mathematics concepts and formulas related to {question}:
            Common mathematical concepts include:
            - Quadratic equations and the quadratic formula
            - Pythagorean theorem and trigonometry
            - Calculus fundamentals (derivatives and integrals)
            - Algebraic manipulations and factoring
            - Geometric principles and theorems""",
            "metadata": {
                "source": "fallback",
                "title": "Mathematical Concepts Overview",
                "relevance": "general"
            }
        }]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    
    from ..data_processing import MathCorpusProcessor
    
    # Create a temporary directory for processing
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_docs")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # Initialize with both required arguments
    processor = MathCorpusProcessor(data_dir=temp_dir, hf_api_key=hf_api_key)
    embeddings = processor.embeddings
    db = FAISS.from_documents(texts, embeddings)
    
    # At the end of process_docs function
    try:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")
    
    return db.as_retriever()

def rag_agent(question: str, max_tokens: int = 512, model_url: str = "https://api.deepseek.com"):
    """Use Deepseek Reasoner with configurable max tokens"""
    retriever = process_docs(question, model_url)
    deepseek_api_key = get_clean_env_var("DEEPSEEK_API_KEY")
    
    # Initialize custom model with API key and max_tokens
    llm = CustomChatModel(api_key=deepseek_api_key, max_tokens=max_tokens)
    
    # Create prompt with system and user content combined
    system_prompt = """You are an expert mathematics professor who excels at explaining complex concepts clearly.
    Your responses should:
    1. Begin with a clear, concise definition or overview
    2. Present mathematical formulas using LaTeX notation ($...$ for inline, $$...$$for display)
    3. Provide step-by-step explanations with clear reasoning
    4. Include relevant examples when helpful
    5. Cite sources when referencing theorems or proofs"""
    
    prompt = PromptTemplate.from_template(f"""
    {system_prompt}
    
    Below are relevant excerpts from mathematical texts:
    {{context}}
    
    Question: {{question}}
    
    Please provide a complete answer following the structure above.
    """)
    
    # Create QA chain with the prompt
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )
    
    # Get result using invoke instead of __call__
    result = qa.invoke({"query": question})
    return result["result"], result["source_documents"]

if __name__ == "__main__":
    question = "What is the Riemann Hypothesis?"
    answer, sources = rag_agent(question)
    print("Answer:", answer)
    print("\nSources:")
    for doc in sources:
        print(f"- {doc.metadata['title']}")