import os
import numpy as np
import pickle

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# load .pdf files
files = [f for f in os.listdir() if f.endswith('.pdf')]

# Embed the contents of each file
for file in files:

    # read and load pdf file in
    pdf_loader = PyMuPDFLoader(os.path.join(os.getcwd(), file))
    pdf_docs = pdf_loader.load()








