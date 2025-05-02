import sys
import os

import langchain.document_loaders
import langchain_community.vectorstores
import langchain_openai

from utils.data_assistant_utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# RAG pipeline

# load and embed documents
load = langchain.document_loaders.TextLoader()



st.set_page_config(page_title='data_assistant.py', layout='wide')
st.title('Automated clustering and analysis of BLAST search')

ncbi_acc_known = st.checkbox('I know the accession number I want to BLAST.')
ncbi_acc_unknown = st.checkbox('I do NOT know the accession number I want to BLAST.')

if ncbi_acc_known:
    accession_input = st.text_input('Enter an NCBI accession number.')

    if accession_input:
        st.write(f'You entered {accession_input}')
        handle = nucleotide_blast(accession_input)
        blast_results = process_stream(handle, save=False)
        st.write(blast_results)

if ncbi_acc_unknown:
    inq_input = st.text_input('What accession number are you interested in finding?')

    if inq_input:
        acc = get_accession(system_prompt='You are an expert in the NCBI accession number formatting, and you take the specimen of interest from the user prompt and search through publicly available databases to find the corresponding NCBI accession number, with an emphasis on finding mRNA and DNA sequences.',
                            user_prompt=inq_input)

        st.write('Retrieving the NCBI accession number for your inquery.')
        st.write(f'The accession number for your prompt - {inq_input} - is:')
        st.write(acc)
