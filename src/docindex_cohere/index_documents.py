from .doc_index import CoherePineconeIndexer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Index documents on Pinecone using OpenAI embeddings.")
    parser.add_argument("--pinecone_api_key", type=str, help="Pinecone API key")
    parser.add_argument("--index_name", type=str, help="Name of the Pinecone index")
    parser.add_argument("--cohere_api_key", type=str, help="OpenAI API key")
    parser.add_argument("--docs", nargs="+", help="URLs of the documents to be indexed")
    
    parser.add_argument("--batch_limit", type=int, default=32, help="Maximum batch size for indexing (default: 100).")
    parser.add_argument("--chunk_size", type=int, default=256, help="Size of texts per chunk (default: 1000 characters).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pinecone_indexer = CoherePineconeIndexer(args.index_name, args.pinecone_api_key, args.cohere_api_key)
    pinecone_indexer.index_documents(args.docs, args.batch_limit, args.chunk_size)
    pinecone_indexer.initialize_vectorstore(args.index_name)
