from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, PodSpec
import tiktoken
from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
from langchain_openai import OpenAIEmbeddings
from tqdm.auto import tqdm
from langchain_pinecone import PineconeVectorStore 
from docindex.doc_model import Page
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4

class PineconeIndexer:
    def __init__(
        self,
        index_name: str,
        pinecone_api_key: str = None,
        cohere_api_key: str = None,
        openai_api_key: str = None,
        google_api_key: str = None
        ):
        self.cohere_api_key = cohere_api_key
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.tokenizer = tiktoken.get_encoding('p50k_base')

    def create_index(self, environment: str = "us-west1-gcp" ):
        """
        Creates an index with the specified parameters.

        Args:
            environment (str, optional): The environment where the index will be created. Defaults to "us-west1-gcp".

        Returns:
            None
        """
        print(f"Creating index {self.index_name}")
        self.pc.create_index(
            name=self.index_name,
            dimension=768,
            metric="cosine",
            spec=PodSpec(
                environment=environment,
                pod_type="p1.x1",
                pods=1
            )
            )
        return print(f"Index {self.index_name} created successfully!")
    
    def delete_index(self):
        """
        Deletes the created index.

        Returns:
            None
        """
        print(f"Deleting index {self.index_name}")
        self.pc.delete_index(self.index_name)
        return print(f"Index {self.index_name} deleted successfully!")
    
    def load_document(self, file_url: str) -> List[str]:
        """
        Load a document from a given file URL and split it into pages.

        This method supports loading documents in various formats including PDF, DOCX, DOC, Markdown, and HTML.
        It uses the appropriate loader for each file type to load the document and split it into pages.

        Args:
            file_url (str): The URL of the file to be loaded.

        Returns:
            List[str]: A list of strings, where each string represents a page from the loaded document.

        Raises:
            ValueError: If the file type is not supported or recognized.
        """
        pages = []
        file_path = Path(file_url)

        file_extension = file_path.suffix

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_url)
            pages = loader.load_and_split()

        elif file_extension in ('.docx', '.doc'):
            loader = UnstructuredWordDocumentLoader(file_url)
            pages = loader.load_and_split()

        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(file_url)
            pages = loader.load_and_split()

        elif file_extension == '.html':
            loader = UnstructuredHTMLLoader(file_url)
            pages = loader.load_and_split()

        return pages
    
    def tiktoken_len(self, text: str) -> int:
        """
        Calculate length of text in tokens.

        Parameters:
            text (str): Input text.

        Returns:
            int: Length of text in tokens.
        """
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    
    def embed(self, sample_text: str):
        """
        Generates embeddings for the provided sample text using either Google's Generative AI or OpenAI.

        Args:
            sample_text (str): The input text to generate embeddings for.

        Returns:
            GoogleGenerativeAIEmbeddings or OpenAIEmbeddings: The embeddings object depending on the API key available.

        Raises:
            ValueError: If no valid API key is provided.
        """
        # Google Generative AI
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            return genai.embed_content(
                model='models/embedding-001',
                content=sample_text,
                task_type="retrieval_document"
            )

        # OpenAI Embeddings
        elif self.openai_api_key:
            return OpenAIEmbeddings(
                openai_api_key=self.openai_api_key
            )
        elif self.cohere_api_key:
            return CohereEmbeddings(model_name = "embed-english-light-v3.0",
                                    cohere_api_key=self.cohere_api_key)
        else:
            raise ValueError("A valid API key for either Google, Cohere or OpenAI must be provided to generate embeddings.")
    
    def upsert_documents(self, documents: List[Page], batch_limit: int, chunk_size: int = 256) -> None:
        """
        Upsert documents into the Pinecone index.

        Args:
            documents (List[Page]): List of documents to upsert.
            batch_limit (int): Maximum batch size for upsert operation.
            chunks_size(int): size of texts per chunk.

        Returns:
            None
        """
        texts = []
        metadatas = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )
        for i, record in enumerate(tqdm(documents)):
            metadata = {
                'content': record.page_content,
                'source': record.page,
                'title': record.source
            }
            record_texts = text_splitter.split_text(record.page_content)  
            record_metadatas = [{
                "chunk": j, "text": text, **metadata
            } for j, text in enumerate(record_texts)]
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)
            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeddings = None
                if self.google_api_key:
                    embeddings = self.embed(texts)['embedding']
                elif self.openai_api_key:
                    embed = self.embed()  
                    embeddings = embed.embed_documents(texts)
                if embeddings is not None:
                    index = self.pc.Index(self.index_name)  
                    index.upsert(vectors=zip(ids, embeddings , metadatas), async_req=True)
                    texts = []
                    metadatas = []
                else:
                    print("No API key provided for embedding generation.")


    def index_documents(self, urls: List[str], batch_limit: int, chunk_size: int = 256) -> None:
        """
        Process a list of URLs and upsert documents to a Pinecone index.

        Args:
            urls (List[str]): List of URLs to process.
            batch_limit (int): Batch limit for upserting documents.
            chunks_size(int): size of texts per chunk.

        Returns:
            None
        """
        for url in tqdm(urls, desc="Processing URLs"):
            print(f"Processing URL: {url}")
            pages = self.load_document(url)
            print(f"Found {len(pages)} pages in the PDF.")
            pages_data = [
                Page(
                    page_content=page.page_content,
                    metadata=page.metadata,
                    page=page.metadata.get("page", 0),
                    source=page.metadata.get("source")
                )
                for page in pages
            ]

            print(f"Upserting {len(pages_data)} pages to the Pinecone index...")
            self.upsert_documents(pages_data, batch_limit, chunk_size)  
            print("Finished upserting documents for this URL.")
        index = self.pc.Index(self.index_name)
        print(index.describe_index_stats())
        print("Indexing complete.")
        return index
        
    def initialize_vectorstore(self, index_name):
        index = self.pc.Index(index_name)
        embed = OpenAIEmbeddings(
                model = 'text-embedding-ada-002',
                openai_api_key = self.openai_api_key
                )
        vectorstore = PineconeVectorStore(index, embed, "text")
        return vectorstore

