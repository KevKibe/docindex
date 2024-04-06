<h1 align="center">DocIndex: Fast Document Storage for RAG</h1>
<p align="center">

  <a href="https://github.com/KevKibe/docindex/commits/">
    <img src="https://img.shields.io/github/last-commit/KevKibe/docindex?" alt="Last commit">
  </a>
  <a href="https://github.com/KevKibe/docindex/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/KevKibe/docindex?" alt="License">
  </a>

*Efficiently store multiple documents and their metadata, whether they're offline or online, in a Pinecone Vector Database optimized for Retrieval Augmented Generation (RAG) models Fast* 

## Features

- ⚡️ **Rapid Indexing**: Quickly index multiple documents along with their metadata, including source, page details, and content, into Pinecone DB.<br>
- 📚 **Document Flexibility**: Index documents from your local storage or online sources with ease.<br>
- 📂 **Format Support**: Seamlessly handle various document formats, including PDF, docx(in-development), etc.<br>
- 🔁 **Embedding Services Integration**: Enjoy support for multiple embedding services such as OpenAIEmbeddings, GoogleGenerativeAIEmbeddings and more in development.<br>
- 🛠️ **Configurable Vectorstore**: Configure a vectorstore directly from the index to facilitate RAG pipelines effortlessly.

## Setup

```python
pip install docindex
```

## Usage
```python
from _openai.index import OpenaiPineconeIndexer

# Replace these values with your actual Pinecone API key, index name, OpenAI API key, and environment
pinecone_api_key = "pinecone-api-key"
index_name = "pinecone-index-name"
openai_api_key = "openai-api-key"
environment = "pinecone-index-environment"

# Define the batch limit for indexing, how many pages per pass.
batch_limit = 20

# List of URLs of the documents to be indexed. (offline on your computer or an online)
urls = [
 "your-document-1.pdf",
 "your-document-2.pdf"
]

# Initialize the Pinecone indexer
pinecone_index = OpenaiPineconeIndexer(index_name, pinecone_api_key, environment, openai_api_key)

# Index the documents with the specified URLs and batch limit
pinecone_index.index_documents(urls,batch_limit)
```

## Initialize Vectorstore

```python
from pinecone import Pinecone as IndexPinecone
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the Pinecone index
index_pc = IndexPinecone(api_key=pinecone_api_key)
index = index_pc.Index(index_name)
        
# Initialize OpenAI embeddings if you're using OpenAI embeddings
embed = OpenAIEmbeddings(
        model = 'text-embedding-ada-002',
        openai_api_key = openai_api_key
        )

# Define the text field
text_field = "text"

# Initialize the Vectorstore with the Pinecone index and OpenAI embeddings
vectorstore = VectorStorePinecone(index, embed.embed_query, text_field)
```

## Using the CLI

- Clone the Repository: Clone or download the application code to your local machine.
```bash
git clone https://github.com/KevKibe/docindex.git
```

- Create a virtual environment for the project and activate it.
```bash
cd docindex

python -m venv venv

source venv/bin/activate
```
- Install dependencies by running this command
```bash
pip install -r requirements.txt
```

- Navigate to src and run this command to index documents
```bash
cd src

python -m _openai.doc_index  --pinecone_api_key "your_pinecone_api_key" --index_name "your_index_name" --openai_api_key "your_openai_api_key" --environment "your_environment" --batch_limit 10 --docs  "doc-1.pdf" "doc-2.pdf'

```
