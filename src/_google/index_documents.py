from .doc_index import GooglePineconeIndexer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Index documents on Pinecone using OpenAI embeddings.")
    parser.add_argument("--pinecone_api_key", type=str, help="Pinecone API key")
    parser.add_argument("--index_name", type=str, help="Name of the Pinecone index")
    parser.add_argument("--google_api_key", type=str, help="OpenAI API key")
    parser.add_argument("--batch_limit", default = 32, type=int,  help="Maximum batch size for indexing")
    parser.add_argument("--docs", nargs="+", help="URLs of the documents to be indexed")
    parser.add_argument("--chunk_size", default = 256, help="size of texts per chunk")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pinecone_indexer = GooglePineconeIndexer(args.index_name, args.pinecone_api_key, args.google_api_key)
    pinecone_indexer.index_documents(args.docs, args.batch_limit, args.chunk_size)
