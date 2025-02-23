from langchain.chains import RetrievalQA, LLMChain
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from ..retriever import CustomChatModel
import wikipedia
import os


def process_docs(question, model_url: str = "http://localhost:8000"):
    # Initialize custom model
    llm = CustomChatModel(base_url=model_url)
    
    # Generate search query from the question
    prompt_template = """Given the following question, generate a concise search query to find relevant Wikipedia articles.
    Question: {question}
    Search Query:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    search_query = chain.run(question).strip()
    
    # Search Wikipedia for relevant page titles
    try:
        search_results = wikipedia.search(search_query)
    except wikipedia.exceptions.DisambiguationError as e:
        search_results = e.options[:3]  # Handle disambiguation by taking top 3 options
    except:
        search_results = []
    
    # Load the top 3 Wikipedia pages
    docs = []
    for title in search_results[:3]:
        try:
            loader = WikipediaLoader(query=title, load_max_docs=1)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {title}: {e}")
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    
    # Create a vector store for similarity search
    from ..data_processing import MathCorpusProcessor
    embeddings = MathCorpusProcessor("").embeddings
    db = FAISS.from_documents(texts, embeddings)
    return db.as_retriever()

def rag_agent(question, model_url: str = "http://localhost:8000"):
    """Use CustomChatModel instead of OpenAI"""
    retriever = process_docs(question, model_url)
    llm = CustomChatModel(base_url=model_url)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa({"query": question})
    return result["result"], result["source_documents"]

if __name__ == "__main__":
    question = "What is the Riemann Hypothesis?"
    answer, sources = rag_agent(question)
    print("Answer:", answer)
    print("\nSources:")
    for doc in sources:
        print(f"- {doc.metadata['title']}")