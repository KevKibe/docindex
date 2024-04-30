from .doc_index import CoherePineconeIndexer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Deletes an Index on Pinecone.")
    parser.add_argument("--pinecone_api_key", type=str, help="Pinecone API key")
    parser.add_argument("--index_name", type=str, help="Name of the Pinecone index")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pinecone_indexer = CoherePineconeIndexer(args.index_name, args.pinecone_api_key)
    pinecone_indexer.delete_index()
