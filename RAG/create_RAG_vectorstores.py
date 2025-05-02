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


# Embed the contents of each file
for file in files:

    file_name = file.split('.')[0]

    # read and load pdf file in
    pdf_loader = PyMuPDFLoader(os.path.join(os.getcwd(), file))
    pdf_doc = pdf_loader.load()

    # split documents
    split_docs = splitter.split_documents(pdf_doc)

    # create vector store
    vectorstore = FAISS.from_documents(pdf_doc, embedding=embedding_model)

    # save vectorstore and metadata
    vectorstore.save_local(f'./vectorstore/faiss_index/{file_name}')
    with open(f'./vectorstore/faiss_index/{file_name}.pkl', 'wb') as f:
        pickle.dump(vectorstore, f)

    # get the FAISS index object
    faiss_index = vectorstore.index

    # save embeddings to a numpy array
    vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    np.save(f'vectorstore/rag_embeddings_{file_name}.npy', vectors)

    # add a reference section from documents to the vector store
    ids = vectorstore.add_documents(documents=pdf_doc)









