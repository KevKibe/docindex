from .doc_index import GooglePineconeIndexer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Creates an Index on Pinecone.")
    parser.add_argument("--pinecone_api_key", type=str, help="Pinecone API key")
    parser.add_argument("--index_name", type=str, help="Name of the Pinecone index")
    parser.add_argument("--google_api_key", type=str, help="Google API key")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pinecone_indexer = GooglePineconeIndexer(args.index_name, args.pinecone_api_key, args.google_api_key)
    pinecone_indexer.create_index()