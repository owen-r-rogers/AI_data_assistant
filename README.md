# AI_data_assistant

## The purpose of this application is to analyze biomedical data.

### Right now:

This application only does BLAST on different nucleotide sequences. BLAST = Basic Local Alignment Search Tool. BLAST finds regions of similarity between biological sequences (https://blast.ncbi.nlm.nih.gov/Blast.cgi)

This has the nice advantage of not needing any input files. There is a LOT of streamlining I still need to do, but right now it's set up to be able to take in a query, return the NCBI accession number (gene identifier) that can then be used to perform a BLAST run to find other similar sequences. 

Right now, the biggest next steps are to write a function to actually carry out the analysis part of this, because right now I only have the pipeline setup.

### To perform a BLAST search:

To get the accession number if you know the organism or biological structure of interest:

```angular2html
accession_number = get_accession('You are an expert in NCBI accession number formatting', 'What is the NCBI accession number of the spike protein on SARS-CoV-2?')
```

To get the sequence of a known accession number (either known before or output by get_accession):

```angular2html
sequence = fetch_sequence(accession_number, 
                          db='nucleotide',
                          rettype='gb',
                          retmode='text',
                          email=your_email,
                          verbose=True)
```

Now to perform a BLAST:

```angular2html
blast_result = nucleotide_blast(sequence)

# process it into a dataframe
df = process_stream(blast_result, save=True, save_name='BLAST_results')
```

### 2May - RAG pipeline
I used the following articles to create a vector store for RAG:  
https://www.nature.com/articles/s41579-024-01036-y  
https://arxiv.org/abs/2306.15006  
https://www.biorxiv.org/content/10.1101/2024.04.30.591806v1  
https://www.nature.com/articles/s41467-023-39856-w  
https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00852-x  
