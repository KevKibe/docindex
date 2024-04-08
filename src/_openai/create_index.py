from .docindex import OpenaiPineconeIndexer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Index documents on Pinecone using OpenAI embeddings.")
    parser.add_argument("--pinecone_api_key", type=str, help="Pinecone API key")
    parser.add_argument("--index_name", type=str, help="Name of the Pinecone index")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pinecone_indexer = OpenaiPineconeIndexer(args.index_name, args.pinecone_api_key, args.openai_api_key)
    pinecone_indexer.create_index()
