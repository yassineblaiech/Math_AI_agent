from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import wikipedia
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def process_docs(question):
    # Generate search query from the question
    prompt_template = """Given the following question, generate a concise search query to find relevant Wikipedia articles.
    Question: {question}
    Search Query:"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm = OpenAI(temperature=0)
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
    top_titles = search_results[:3]
    docs = []
    for title in top_titles:
        try:
            loader = WikipediaLoader(query=title, load_max_docs=1)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {title}: {e}")
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    
    # Create a vector store for similarity search
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db.as_retriever()


def rag_agent(question):
    retriever = process_docs(question)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa({"query": question})
    return result["result"], result["source_documents"]

def rag_agent_with_other_llm(question):
    retriever = process_docs(question)
    
    qa = RetrievalQA.from_chain_type(
        llm=vllm_llm,  # or custom_llm
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