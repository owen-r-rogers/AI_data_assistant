# RAG log

### For RAG, I used four primary research articles and one review I found after a quick literature search.
### These are mostly about different ways to embed genetic sequences, although two of them are more focused on drug development. One of them is about the general principle of drug design based on the sequence of the genetic material, and another is on COVID treatment (since I've been developing this method only testing on proteins related to COVID).

#### General steps I took for making a RAG pipeline:

---
I. I made a separate python script to carry out the generation of the vector embeddings.

II. In this file I made a list of files (ending with .pdf since the script was in that directory), and then I went through the process of splitting the files using the RecursiveCharacterTextSplitter. I appended these to a list and then iterated through it create FAISS vector stores.

III. The main reason I first concatenated all of the data was so that I could save all of the data as a single file type. In the output for example, there is only one file for the raw numpy embeddings, etc. This seemed more practical given that I want to retrieve all of this information at the same time when I use my AI assistant.