import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import streamlit as st
import datetime
from dotenv import load_dotenv
from Bio import Blast
from Bio import Entrez
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import SeqIO
from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


# Streamlit
st.set_page_config(page_title='data_assistant.py', layout='wide')
st.title('Automated clustering and analysis of BLAST search')

ncbi_acc_known = st.checkbox('I know the accession number I want to BLAST.')

if ncbi_acc_known:
    accession_input = st.text_input('Enter an NCBI accession number.')
    st.write(f'You entered {accession_input}')

    handle = nucleotide_blast(accession_input)
    blast_results = process_stream(handle, save=False)
    st.write(blast_results)
else:
    ask_for_acc = st.text_input('')

# Create a box at the bottom of the page for input
prompt = st.chat_input('Ask a question')






if prompt:
    accession = get_accession('You are an expert in telling people NCBI accession codes.', prompt)
    st.write(f"Predicted accession number: {accession}")

