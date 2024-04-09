<h1 align="center">DocIndex: Fast Persistent Document Embeddings Storage for RAG</h1>
<p align="center">

  <a href="https://github.com/KevKibe/docindex/commits/">
    <img src="https://img.shields.io/github/last-commit/KevKibe/docindex?" alt="Last commit">
  </a>
  <a href="https://github.com/KevKibe/docindex/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/KevKibe/docindex?" alt="License">
  </a>
<br>

![Diagram](image.png)

*Efficiently store multiple document embeddings and their metadata, whether they're offline or online, in a persistent Pinecone Vector Database optimized for Retrieval Augmented Generation (RAG) applications fast* 

## Features

- ⚡️ **Rapid Indexing**: Quickly index multiple documents along with their metadata, including source, page details, and content, into Pinecone DB.<br>
- 📚 **Document Flexibility**: Index documents from your local storage or online sources with ease.<br>
- 📂 **Format Support**: Seamlessly handle various document formats, including PDF, docx(in-development), etc.<br>
- 🔁 **Embedding Services Integration**: Enjoy support for multiple embedding services such as OpenAI Embeddings, Google Generative AI Embeddings and more in development.<br>
- 🛠️ **Configurable Vectorstore**: Configure a vectorstore directly from the index to facilitate RAG pipelines effortlessly.

## Setup

```python
pip install docindex
```

## Getting Started
- Sign up to [Pinecone](https://www.pinecone.io/) and get an API key.
## Using OpenAI 
```python
from _openai.docindex import OpenaiPineconeIndexer

# Replace these values with your actual Pinecone API key, index name, OpenAI API key
pinecone_api_key = "pinecone-api-key"
index_name = "index-name"               # e.g index-1
openai_api_key = "openai-api-key"
batch_limit = 20                        # Batch limit for upserting documents
chunk_size = 256                        # Optional: size of texts per chunk. 

# List of URLs of the documents to be indexed. (offline on your computer or online)
urls = [
 "your-document-1.pdf",
 "your-document-2.pdf"
]

# Initialize the Pinecone indexer
pinecone_indexer = OpenaiPineconeIndexer(index_name, pinecone_api_key, openai_api_key)

# To create a new Index
pinecone_indexer.create_index()

# Store the document embeddings with the specified URLs and batch limit
pinecone_indexer.index_documents(urls,batch_limit,chunk_size)

# Initialize the Vectorstore
vectorstore = pinecone_indexer.initialize_vectorstore(index_name)
```
```python
# To delete the created Index
pinecone_indexer.delete_index()
```


## Using Google Generative AI  

```python
from _google.docindex import GooglePineconeIndexer

# Replace these values with your actual Pinecone API key, index name, Google API key
pinecone_api_key = "pinecone-api-key"
index_name = "index-name"                # e.g index-1
google_api_key = "google-api-key"
batch_limit = 20                         # Batch limit for upserting documents
chunk_size = 256                         # Optional: size of texts per chunk. 

# List of URLs of the documents to be indexed. (offline on your computer or an online)
urls = [
 "your-document-1.pdf",
 "your-document-2.pdf"
]

pinecone_indexer = GooglePineconeIndexer(index_name, pinecone_api_key, google_api_key)

# To create a new Index
pinecone_indexer.create_index()

# Store the document embeddings with the specified URLs and batch limit
pinecone_indexer.index_documents(urls,batch_limit,chunk_size)

# Initialize the Vectorstore
vectorstore = pinecone_indexer.initialize_vectorstore(index_name)
```
```python
# To delete the created Index
pinecone_indexer.delete_index()
```


## Using the CLI

- Clone the Repository: Clone or download the application code to your local machine.
```bash
git clone https://github.com/KevKibe/docindex.git
```

- Create a virtual environment for the project and activate it.
```bash
# Navigate to project repository
cd docindex

# create virtual environment
python -m venv venv

# activate virtual environment
source venv/bin/activate
```
- Install dependencies by running this command
```bash
pip install -r requirements.txt
```

- Navigate to src 
```bash
cd src
```
- To create an index

```bash
# Using OpenAI 
python -m  _openai.create_index --pinecone_api_key "your_pinecone_api_key" --index_name "your_index_name" --openai_api_key "your_openai_api_key"

# Using Google Generative AI
python -m  _google.create_index --pinecone_api_key "your_pinecone_api_key" --index_name "your_index_name" --google_api_key "your_google_api_key"
```

- Run the command to start indexing the documents

```bash
# Using OpenAI 
python -m _openai.doc_index  --pinecone_api_key "your_pinecone_api_key" --index_name "your_index_name" --openai_api_key "your_openai_api_key" --batch_limit 10 --docs  "doc-1.pdf" "doc-2.pdf' --chunk_size 256 
```
```bash
# Using Google Generative AI 
python -m _google.doc_index  --pinecone_api_key "your_pinecone_api_key" --index_name "your_index_name" --google_api_key "your_google_api_key" --batch_limit 10 --docs  "doc-1.pdf" "doc-2.pdf' --chunk_size 256 
```
- To delete an index

```bash
# Using OpenAI 
python -m  _openai.delete_index --pinecone_api_key "your_pinecone_api_key" --index_name "your_index_name" --openai_api_key "your_openai_api_key"

# Using Google Generative AI
python -m  _google.delete_index --pinecone_api_key "your_pinecone_api_key" --index_name "your_index_name" --google_api_key "your_google_api_key"
```

## Contributing 
Contributions are welcome and encouraged.

Before contributing, please take a moment to review our [Contribution Guidelines](https://github.com/KevKibe/docindex/blob/master/DOCS/CONTRIBUTING.md) for important information on how to contribute to this project.

If you're unsure about anything or need assistance, don't hesitate to reach out to us or open an issue to discuss your ideas.

We look forward to your contributions!

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/KevKibe/docindex/blob/master/LICENSE) file for details.

## Contact
For any enquiries, please reach out to me through keviinkibe@gmail.com