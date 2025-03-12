import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import getpass

# Securely retrieve API keys
# os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")
os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN")

class AdaptiveRAG:
    def __init__(self):
        """Initializes the AdaptiveRAG system."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key="AIzaSyBKoqnMGUgyAjSlEHJHSCTG96CrIMpYxlY",
            temperature=0.3)

        self.embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={"trust_remote_code": True})
        self.vectorstore = None
        self.retriever = None

    # def load_documents(self, documents: List[str]):
    #     """Processes and loads documents into the system."""
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #     texts = text_splitter.create_documents(documents)
    #     self.vectorstore = Chroma.from_documents(texts, self.embeddings)
    #     self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
    
   
    def load_pdf(self, pdf_path: str):
        """Processes and loads a PDF document into the system."""
        # Extract text from PDF
        def extract_text_from_pdf(pdf_path):
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])
            return text

        # Read and split text
        raw_text = extract_text_from_pdf(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents([raw_text])
    
        # Load into vector store
        self.vectorstore = Chroma.from_documents(texts, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def query_reformulation(self, query: str, context: str) -> str:
        """Reformulates a query based on context."""
        prompt = f"""Given the original query and the current context, reformulate the query to be more specific:

        Original query: {query}
        Current context: {context}

        Reformulated query:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def generate_answer(self, query: str, context: str) -> str:
        """Generates an answer based on query and context."""
        prompt = f"""Answer the following query using the given context:

        Query: {query}
        Context: {context}

        Answer:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def run(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Runs the adaptive RAG process."""
        original_query = query
        context = ""
        iteration_results = []

        for i in range(max_iterations):
            if i > 0:
                new_query = self.query_reformulation(original_query, context)
                if new_query == query:
                    break
                query = new_query
            docs = self.retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in docs])

            answer = self.generate_answer(query, context)

            iteration_results.append({
                "iteration": i + 1,
                "query": query,
                "answer": answer,
                "source_documents": docs
            })

            if self.is_answer_satisfactory(answer):
                break

        return {"final_answer": answer, "iterations": iteration_results}

    def is_answer_satisfactory(self, answer: str) -> bool:
        """Checks if the answer is satisfactory based on length."""
        return len(answer) > 500

# Example Usage
adaptive_rag = AdaptiveRAG()

# documents = [
#     "Paris is the capital of France.",
#     "London is the capital of the United Kingdom.",
#     "Berlin is the capital of Germany."
# ]
pdf_path = r"C:\Users\vahin\OneDrive\Desktop\projects-rag\adaptive-rag\climate.pdf"
adaptive_rag.load_pdf(pdf_path)

#query = "Why Consider Pre-Industrial Climate Change?"
query = "Spatial and Temporal Patterns of the Response to Different Forcings and their Uncertainties"
result = adaptive_rag.run(query)

print(f"Final answer: {result['final_answer']}")
print(f"Number of iterations: {len(result['iterations'])}")
for i, iteration in enumerate(result['iterations']):
    print(f"\nIteration {i + 1}:")
    print(f"Query: {iteration['query']}")
    print(f"Answer: {iteration['answer']}")
