from docindex.doc_indexing import PineconeIndexer
import argparse

def parse_args():
    """
    Parse and return command-line arguments for indexing documents on Pinecone.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Index documents on Pinecone using specified embeddings.")
    parser.add_argument("--pinecone_api_key", type=str, required=True, help="Pinecone API key to access the Pinecone index.")
    parser.add_argument("--google_api_key", type=str, help="Google API key for Generative AI embeddings.")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key for OpenAI embeddings.")
    parser.add_argument("--cohere_api_key", type=str, help="Cohere API key for Cohere embeddings.")

    parser.add_argument("--index_name", type=str, required=True, help="Name of the Pinecone index to use.")
    parser.add_argument("--docs", nargs="+", required=True, help="List of URLs for the documents to be indexed.")

    parser.add_argument("--batch_limit", type=int, default=20, help="Maximum batch size for indexing (default: 100).")
    parser.add_argument("--chunk_size", type=int, default=256, help="Size of texts per chunk (default: 1000 characters).")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    pinecone_indexer = PineconeIndexer(args.index_name, 
                                       args.pinecone_api_key, 
                                       args.google_api_key, 
                                       args.openai_api_key, 
                                       args.cohere_api_key)
    pinecone_indexer.index_documents(args.docs, args.batch_limit, args.chunk_size)
