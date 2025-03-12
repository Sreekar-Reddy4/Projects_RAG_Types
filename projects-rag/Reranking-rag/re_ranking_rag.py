from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.chains import RetrievalQA
import os

# Ensure API keys are set properly
#os.environ["GROQ_API_KEY"] = "your_groq_api_key"  # Use environment variable instead of hardcoding
os.environ["HF_TOKEN"] = "your_huggingface_api_key"

# Helper function to print documents
def pretty_print_docs(docs):
    for i, d in enumerate(docs):
        print(f"{'-' * 50}\nDocument {i + 1}:")
        print(f"Content:\n{d.page_content}\n")
        print("Metadata:")
        for key, value in d.metadata.items():
            print(f"  {key}: {value}")
    print(f"{'-' * 50}")

# Step 1: Load and process data
def load_and_process_data(url):
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["id"] = idx

    return chunks

# Step 2: Create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={'trust_remote_code': True})
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

# Step 3: Re-ranking Retrieval-Augmented Generation
def reranking_rag(query, vectorstore, llm):
    # Retrieve top-k documents using the initial retriever
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(query)  # FIXED: Use invoke()

    # Ensure FlashRank is properly initialized
    FlashrankRerank.model_rebuild()  # FIXED: Ensure the model is defined before use
    reranker = FlashrankRerank()

    # Apply FlashRank to re-rank retrieved documents
    reranked_docs = reranker.compress_documents(retrieved_docs, query=query)

    # Initialize RAG pipeline
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}))

    # Generate response using the re-ranked documents
    response = chain.invoke(query)

    return {
        "query": query,
        "final_answer": response["result"],
        "retrieval_method": "Re-ranking with FlashRank",
        "reranked_docs": reranked_docs
    }

# Step 4: Initialize LLM & Process Data
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key="gsk_FhF1pzKD2UaYidsjD42oWGdyb3FYrpNwb4zbitHKP0bWzsqzdny",
    temperature=0.5
)

# Load and process data
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
chunks = load_and_process_data(url)

# Create vector store
vectorstore = create_vector_store(chunks)

# Example queries
queries = [
    "What are the main applications of artificial intelligence in healthcare?",
    "Explain the concept of machine learning and its relationship to AI.",
    "Discuss the ethical implications of AI in decision-making processes."
]

# Run Re-ranking RAG for each query
for query in queries:
    print(f"\nQuery: {query}")
    result = reranking_rag(query, vectorstore, llm)
    print("Final Answer:")
    print(result["final_answer"])
    print("\nRetrieval Method:")
    print(result["retrieval_method"])

    # Display re-ranked documents
    print("\nDocuments after re-ranking:")
    pretty_print_docs(result["reranked_docs"])
