import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import streamlit as st
import seaborn as sns
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

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

from sklearn.manifold import TSNE

class Draft:
    def __init__(self,
                 max_tokens=None,
                 temperature=None,
                 model='gpt-4o-mini',
                 embedding_model='text-embedding-ada-002',
                 provider='openai',
                 rag=True):

        # load openAI API key
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key

        # initialize parameters for the openai prompt
        self.max_tokens = max_tokens if max_tokens is not None else 100
        self.temperature = temperature if temperature is not None else 0.7
        self.model = model if model is not None else "gpt-4o-mini"
        self.provider = provider if provider is not None else "openai"
        self.langchain_model = None
        self.embedding_model = embedding_model

        # initialize RAG retriever
        self.vectorstore = FAISS.load_local('/Users/owenrogers/Desktop/projects/github/school/AI_data_assistant/RAG/vectorstore/faiss_index',
                                            embeddings=OpenAIEmbeddings(model=embedding_model),
                                            allow_dangerous_deserialization=True)

    def assign_tokens(self, tokens):
        self.max_tokens = tokens

    def assign_temperature(self, temperature):
        self.temperature = temperature

    def assign_model(self, model):
        self.model = model

    def assign_provider(self, provider):
        self.provider = provider

    def prime(self):
        # ititialize the langchain model after specifying parameters

        self.langchain_model = init_chat_model(self.model,
                                               model_provider=self.provider,
                                               temperature=self.temperature,
                                               max_tokens=self.max_tokens)

    def generate(self, system_prompt, user_prompt):
        # generate the response

        if self.langchain_model is None:
            self.prime()

        # create a prompt
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt),
        ]

        # generate the response
        message = self.langchain_model.invoke(messages)

        return message.content


class Critique:

    # cri

    def __init__(self,
                 response,
                 user_prompt=None,
                 max_tokens=None,
                 temperature=None,
                 model=None):

        # load openAI API key
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key


        # take in the response from the Draft()
        self.response = response
        self.user_prompt = user_prompt if user_prompt is not None else f'Evaluate the accuracy of: "{response}"'
        self.max_tokens = max_tokens if max_tokens is not None else 100
        self.temperature = temperature if temperature is not None else 0.7
        self.model = model if model is not None else "gpt-4o-mini"
        self.langchain_model = None

    def prime(self):
        self.langchain_model = init_chat_model(
            self.model,
            model_provider='openai',
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


    def generate(self, system_prompt, user_prompt):

        if self.langchain_model is None:
            self.prime()

        # create a prompt
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt)
        ]

        # generate the response
        critique = self.langchain_model.invoke(messages)

        return critique.content


class Report():
    def __init__(self, draft, critique, max_tokens=None, temperature=None, model=None, provider=None):
        self.draft = draft          # output of Draft
        self.critique = critique    # output of Critique
        self.max_tokens = max_tokens if max_tokens is not None else 100
        self.temperature = temperature if temperature is not None else 0.7
        self.model = model if model is not None else "gpt-4o-mini"
        self.provider = provider if provider is not None else "openai"
        self.langchain_model = None

    def prime(self):
        self.langchain_model = init_chat_model(self.model, model_provider=self.provider, temperature=self.temperature, max_tokens=self.max_tokens)

    def report(self, user_prompt=None, system_prompt=None):

        if self.langchain_model is None:
            self.prime()

        user_prompt = user_prompt if user_prompt is not None else f'Can you give me a final answer integrating these two responses: "{self.draft}", and the critique of that draft: "{self.critique}" '
        system_prompt = system_prompt if system_prompt is not None else f'You are an AI agent that generates a response integrating the draft: {self.draft} and the critique of that draft: {self.critique}.'

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt)
        ]

        report = self.langchain_model.invoke(messages)

        return report.content


def ask_ai(system_prompt, user_prompt):
    """
    Expand functinoality to make it more generalizable. Include arguments for up and sp for all classes.
    """

    # load OpenAI API key
    try:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        print(e)

    # initialize Draft
    draft = Draft(max_tokens=600, temperature=0.7)
    draft.prime()
    guess = draft.generate(system_prompt, user_prompt)

    # initialize Critique
    critique = Critique(guess)
    critique.prime()
    critique_sp = 'You are an expert on accesion number formating for the NCBI database.'
    critique_up = f'Please evaluate the accuracy of {guess} and provide feedback, including the correct accession ID if applicable.'
    evaluation = critique.generate(critique_sp, critique_up)

    # initialize report
    report = Report(guess, evaluation)
    report_sp = 'You are the final reporter for taking in an initial guess and a criticial evaluation.md of that guess, reporting the final correct answer to the user.'
    report_up = f'Please integrate the critique: "{evaluation}", of the initial guess: "{guess}".'
    final_report = report.report(user_prompt=report_up, system_prompt=report_sp)

    # report only the accesion number
    prompt = f'Can you take this string and return ONLY the NCBI accession number, if it is deemed correct by the model: {final_report}. Again, ONLY return a string of the NCBI accession number.'
    accession = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
        max_tokens=100
    )

    return accession.choices[0].message.content


def fetch_sequence(accession_no,
                   db='nucleotide',
                   rettype='gb',
                   retmode='text',
                   email='your_email_here',
                   verbose=True):

    Entrez.email = email
    handle = Entrez.efetch(db=db, id=accession_no, rettype=rettype, retmode=retmode)

    record = SeqIO.read(handle, rettype if rettype != 'gb' else 'genbank')

    if verbose:
        print(f'The sequence in {rettype.upper()} format for ID {record.id}:')
        print(record.seq[:50])
        print('')
        print(record.description)

    handle.close()
    return record


def nucleotide_blast(sequence, database='nt', hitlist_size=50, entrez_query=None):

    try:
        # Perform BLAST search

        if entrez_query is not None:
            result_handle = NCBIWWW.qblast('blastn', database, sequence, hitlist_size=hitlist_size,entrez_query=entrez_query)
        else:
            result_handle = NCBIWWW.qblast('blastn', database, sequence,hitlist_size=hitlist_size)

        return result_handle

    except Exception as e:
        print(f"An error occurred during BLAST search: {e}")
        return None


def process_stream(stream, save=True, save_name='BLAST_results'):
    """
    Function to process a stream object into a dataframe and save it as a .csv file.

    """

    record = NCBIXML.parse(stream)

    rows = []

    for br in record:
        for al in br.alignments:
            hsp = al.hsps[0]
            row = {
                'title': al.title,
                'length': al.length,
                'e_value': hsp.expect,
                'score': hsp.score,
                'query_start': hsp.query_start,
                'query_end': hsp.query_end
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    df['title'] = df['title'].apply(lambda x: x.split('|')[3])

    # create and fill a dictionary stroing accession numbers, sequence, and summary for clustering
    blast_accessions = {'title': [],
                        'sequence': [],
                        'summary': []}

    for record in df['title']:

        blast_accessions['title'].append(record)

        rec = fetch_sequence(record, email='orogers@wesleyan.edu')

        sequence = rec.seq

        seq = str(sequence)

        blast_accessions['sequence'].append(seq)

        try:
            blast_accessions['summary'].append(rec.annotations['comment'])

        except:

            print(f'Accession number {record} does not have a summary')
            blast_accessions['summary'].append(f'There is no summary provided for accession no. {record}')

    ext_df = pd.DataFrame(blast_accessions)

    assert (ext_df['title'].values == df['title'].values).all(), 'Something went wrong in the way you generated the dictionary'

    all_data = ext_df.merge(df, 'left', on='title')

    if save:
        all_data.to_csv(f'{save_name}.csv', index=False)

    return all_data


def batch_embed(documents, batch_size=20, delay=1.0, model='text-embedding-ada-002'):
    import time

    embeddings = OpenAIEmbeddings(
        model=model
    )

    results = []

    for i in range(0, len(documents), batch_size):
        chunk = documents[i:i + batch_size]

        try:
            vectors = embeddings.embed_documents(chunk)
            results.extend(vectors)

            print(f'Successfully embedded and stored chunk {i}')

        except Exception as e:
            print(e)

        time.sleep(delay)

    return results


def prepare_for_embedding(processed_stream):

    # process data from processed_stream df that will be necessary for embedding
    for_clustering = processed_stream[['title', 'sequence', 'summary', 'e_value', 'score']]
    for_clustering.loc[:, 'sequence'] = for_clustering['sequence'].apply(lambda x: str(x))
    for_clustering.loc[:, 'text'] = for_clustering.loc[:, 'summary'] + ' ' + for_clustering.loc[:, 'sequence']
    for_clustering = for_clustering[['text']]

    # create a list of lists for embedding
    documents = for_clustering['text'].tolist()

    # batch embed
    embedded_seqs = batch_embed(documents)

    df = pd.DataFrame(embedded_seqs)
    matrix = df.values
    assert len(df) == len(processed_stream), 'The two dataframes are not equal in length'
    df['title'] = processed_stream['title']

    return df, matrix


def plot_tsne(matrix):
    """
    Take output of prepare_for_embedding() and plot t-SNE
    """

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate='auto')

    vis_dims = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims]
    y = [y for x, y in vis_dims]

    # create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=x, y=y, ax=ax, s=200, alpha=0.6, color='red', marker='o', edgecolor='black')

    ax.set_xlabel('Dimension 1', fontsize=16)
    ax.set_ylabel('Dimension 2', fontsize=16)
    ax.set_title('t-SNE embeddings', fontsize=28)

    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    return fig, ax
