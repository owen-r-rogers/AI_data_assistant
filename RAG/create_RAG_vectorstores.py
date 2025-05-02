import os
import numpy as np
import pickle

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# load openAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# load .pdf files
files = [f for f in os.listdir() if f.endswith('.pdf')]


# instantiate splitter and embedding model
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_model = OpenAIEmbeddings()


# collect all split documents
all_split_docs = []


for file in files:
    pdf_loader = PyMuPDFLoader(os.path.join(os.getcwd(), file))
    pdf_doc = pdf_loader.load()
    split_doc = splitter.split_documents(pdf_doc)
    all_split_docs.extend(split_doc)


# create one FAISS vector store from all documents
vectorstore = FAISS.from_documents(all_split_docs, embedding=embedding_model)


# save results
os.makedirs('vectorstore/faiss_index', exist_ok=True)
vectorstore.save_local('vectorstore/faiss_index')


# save raw numpy embeddings
faiss_index = vectorstore.index
vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
np.save(os.path.join('vectorstore/faiss_index', 'rag_embeddings.npy'), vectors)
