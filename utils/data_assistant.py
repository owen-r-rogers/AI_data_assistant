from utils.data_assistant_utils import *


st.set_page_config(page_title='data_assistant.py', layout='wide')
st.title('Automated clustering and analysis of BLAST search')

ncbi_acc_known = st.checkbox('I know the accession number I want to BLAST.')

if ncbi_acc_known:
    accession_input = st.text_input('Enter an NCBI accession number.')
    st.write(f'You entered {accession_input}')

    handle = nucleotide_blast(accession_input)
    blast_results = process_stream(handle, save=False)
    st.write(blast_results)

# Create a box at the bottom of the page for input
prompt = st.chat_input('Ask a question')


if prompt:
    accession = get_accession('You are an expert in telling people NCBI accession codes.', prompt)
    st.write(f"Predicted accession number: {accession}")

