from index import OpenAIPineconeIndexer


pinecone_api_key = " "
index_name = " "
openai_api_key = " "
environment = " "


batch_limit = 10

urls = [
    "https://akinsure.com/content/uploads/insurance-guidebook.pdf",
    "https://www.ira.go.ke/images/docs/insure/INSURANCE_ACT_UPDATED_2022-1.pdf",
]


pinecone_index = OpenAIPineconeIndexer(index_name, pinecone_api_key, environment, openai_api_key)
pinecone_index.index_documents(urls,batch_limit)


    