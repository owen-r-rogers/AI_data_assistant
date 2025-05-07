import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_assistant_utils import *

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key


st.set_page_config(page_title='data_assistant.py', layout='wide')
st.title('Automated clustering and analysis of BLAST search')


acc_option = st.radio(
    'How do you want to provide the accession number to BLAST?',

    [
        'I know the accession number',
        'I do NOT know the accession number',
        'I want to upload a file'
    ]
)


if acc_option == 'I know the accession number':
    accession_input = st.text_input('Enter an NCBI accession number.')

    if accession_input:
        st.write(f'You entered {accession_input}')

        # insert slider for how many hits to display
        hitsize = st.text_input('Enter how many hits to return')

        if hitsize:

            handle = nucleotide_blast(accession_input, hitlist_size=hitsize)
            blast_results = process_stream(handle, save=False)

            # st.write(type(blast_results['sequence'][0]))
            st.write(blast_results)

            embed = st.checkbox('Embed BLAST results?')

            if embed:

                with st.status('Preparing data for embedding...'):

                    blast_df, matrix = prepare_for_embedding(blast_results)

                with st.status('Displaying embeddings...'):

                    tsne_fig, tsne_ax = plot_tsne(matrix)
                    pca_fig, pca_ax = plot_pca(matrix)

                tab1, tab2, tab3 = st.tabs(['t-SNE plot', 'PCA plot', 'Data'])

                with tab1:

                    st.pyplot(tsne_fig, use_container_width=True)

                with tab2:

                    st.pyplot(pca_fig, use_container_width=True)

                with tab3:
                    st.header('Raw embeddings of the BLAST results.')
                    st.write(matrix)




if acc_option == 'I do NOT know the accession number':
    inq_input = st.text_input('What accession number are you interested in finding?')

    if inq_input:
        acc = ask_ai(system_prompt='You are an expert in the NCBI accession number formatting, and you take the specimen of interest from the user prompt and search through publicly available databases to find the corresponding NCBI accession number, with an emphasis on finding mRNA and DNA sequences.',
                            user_prompt=inq_input)

        st.write('Retrieving the NCBI accession number for your inquery.')
        st.write(f'The accession number for your prompt - {inq_input} - is:')
        st.write(acc)

        hitsize = st.text_input('Enter how many hits to return')

        if hitsize:

            handle = nucleotide_blast(acc, hitlist_size=hitsize)
            blast_results = process_stream(handle, save=False)

            st.write(blast_results)

            embed = st.checkbox('Embed BLAST results?')

            if embed:

                with st.status('Preparing data for embedding...'):

                    blast_df, matrix = prepare_for_embedding(blast_results)

                with st.status('Displaying embeddings...'):

                    tsne_fig, tsne_ax = plot_tsne(matrix)
                    pca_fig, pca_ax = plot_pca(matrix)

                tab1, tab2, tab3 = st.tabs(['t-SNE plot', 'PCA plot', 'Data'])

                with tab1:

                    st.pyplot(tsne_fig, use_container_width=True)

                with tab2:

                    st.pyplot(pca_fig, use_container_width=True)

                with tab3:
                    st.header('Raw embeddings of the BLAST results.')
                    st.write(matrix)

    # st.write('Hello')

if acc_option == 'I want to upload a file':

    uploaded_file = st.file_uploader('Upload your file containing BLAST results')

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.text_area('File contents', df.head(), height=200)



