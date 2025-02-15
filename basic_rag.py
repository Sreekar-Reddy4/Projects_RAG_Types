# -*- coding: utf-8 -*-
"""basic_rag.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cA3yryEXdhIHwWRfAMleWPj0pv6izRXl

<a href="https://github.com/genieincodebottle/generative-ai/blob/main/genai-usecases/advance-rag/basic-rag/basic-rag.ipynb" target="_parent"><img src="https://img.shields.io/badge/Open in GitHub-181717?style=flat&logo=github&logoColor=white" alt="Open In GitHub"></a>

# 📚🔍🤖 What is Basic RAG?

Basic RAG is the standard, straightforward implementation of **Retrieval-Augmented Generation**. It involves retrieving relevant information from a knowledge base in response to a query, then using this information to generate an answer using a language model.

# ❓ Why we need RAG?

1. Combines the broad knowledge of language models with specific, up-to-date information
2. Improves accuracy of responses by grounding them in retrieved facts
3. Reduces hallucinations common in standalone language models
4. Allows for easy updates to the knowledge base without retraining the entire model

# Install required libraries
"""

# !pip install -q -U \
#      Sentence-transformers==3.0.1 \
#      langchain==0.2.11 \
#      langchain-google-genai==1.0.7 \
#      langchain-community==0.2.10 \
#      langchain-huggingface==0.0.3 \
#      einops==0.8.0 \
#      faiss_cpu==1.8.0.post1

"""# Import related libraries related to Langchain, HuggingfaceEmbedding"""

# Import Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import getpass
import os

"""# Provide Google API Key. You can create Google API key at following lin

[Google Gemini-Pro API Creation Link](https://console.cloud.google.com/apis/credentials)

[YouTube Video](https://www.youtube.com/watch?v=ZHX7zxvDfoc)


"""
# a good approach to securely entering the api key
os.environ["GOOGLE_API_KEY"] = getpass.getpass()

"""# Provide Huggingface API Key. You can create Huggingface API key at following lin

[Huggingface API Creation Link](https://huggingface.co/settings/tokens)



"""

os.environ["HF_TOKEN"] = getpass.getpass()

# Helper function for printing docs
def pretty_print_docs(docs):
    # Iterate through each document and format the output
    for i, d in enumerate(docs):
        print(f"{'-' * 50}\nDocument {i + 1}:")
        print(f"Content:\n{d.page_content}\n")
        print("Metadata:")
        for key, value in d.metadata.items():
            print(f"  {key}: {value}")
    print(f"{'-' * 50}")  # Final separator for clarity

# Example usage
# Assuming `docs` is a list of Document objects

"""# Basic RAG in Action"""

# Import necessary libraries
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents from a web URL
#documents = WebBaseLoader("https://github.com/hwchase17/chroma-langchain/blob/master/state_of_the_union.txt").load()
# load from local storage
file_path = r'D:\Projects_RAG_Types\state_of_the_union.txt'

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()
# Split documents into chunks of 500 characters with 100 characters overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Add unique IDs to each text chunk
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

# Create embeddings for the text chunks
embedding = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs = {'trust_remote_code': True})

# Initialize a FAISS retriever with the text embeddings
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

# Define a query and retrieve relevant documents
query1 = "What did the president say about Ketanji Brown Jackson"
query = "who is intel's ceo and what did he tell"
query2 = "what is Q99 about"
docs = retriever.invoke(query)
# Print the retrieved documents
pretty_print_docs(docs)

# Import the RetrievalQA chain for question-answering tasks
from langchain.chains import RetrievalQA
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="Your API KEY here")
# Create a RetrievalQA chain using the language model and the retriever
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Invoke the chain with a specific query to get a response
reponse = chain.invoke(query)

# Print the result of the response
print(reponse["result"])