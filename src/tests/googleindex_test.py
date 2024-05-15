import unittest
from _google.doc_index import GooglePineconeIndexer
import os 
from io import StringIO
from unittest.mock import patch
import pinecone
from langchain_pinecone import PineconeVectorStore 
from dotenv import load_dotenv
load_dotenv()


class TestGooglePineconeIndexer(unittest.TestCase):
    """
    Test case class for the GooglePineconeIndexer.
    """

    def setUp(self):
        """
        Set up the test case with common attributes.
        """
        self.index_name = "new-index-1"
        self.pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        self.google_api_key = os.environ.get('GOOGLE_API_KEY')
        self.indexer = GooglePineconeIndexer(self.index_name, self.pinecone_api_key, self.google_api_key)
        return super().setUp()
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_01_create_index(self, mock_stdout):
        """
        Test creating an index and assert the output.
        """
        self.indexer.create_index()
        printed_output = mock_stdout.getvalue().strip()  
        lines = printed_output.split('\n')
        index_created_message_0 = lines[0] 
        self.assertEqual(index_created_message_0, f"Creating index {self.index_name}")
        index_created_message_1 = lines[1]
        self.assertEqual(index_created_message_1, f"Index {self.index_name} created successfully!")

    @patch('builtins.print')
    def test_02_index_documents(self, mock_print):
        """
        Test indexing documents and assert the type of the index.
        """
        urls = [
            "https://arxiv.org/pdf/1706.03762.pdf",
            "src/tests/DOCX_TestPage.docx", 
            "src/tests/TEST.md",
            "src/tests/test.html"
            ]
        self.indexer.index_documents(urls, batch_limit=10, chunk_size=256)
        index = self.indexer.pc.Index(self.index_name)
        self.assertIsInstance(index, pinecone.data.index.Index)
        
    def test_03_initialize_vectorstore(self):
        """
        Test initializing the vector store and assert its type.
        """
        vectorstore = self.indexer.initialize_vectorstore(self.index_name)
        self.assertIsInstance(vectorstore, PineconeVectorStore)

    # def test_04_retrieve_and_generate(self):
    #     """
    #     Test initializing the vector store and assert its type.
    #     """
    #     vector_store = self.indexer.initialize_vectorstore(self.index_name)
    #     response = self.indexer.retrieve_and_generate(
    #         query = "tell me something from the context texts",
    #         vector_store = vector_store,
    #         top_k = 3,
    #         rerank_model = "t5"
    #         )
    #     print(response)
    #     self.assertIsNotNone(response, "The retriever response should not be None.")

    @patch('sys.stdout', new_callable=StringIO)
    def test_05_delete_index(self, mock_stdout):
        """
        Test deleting an index and assert the output.
        """
        self.indexer.delete_index()
        printed_output = mock_stdout.getvalue().strip()  
        lines = printed_output.split('\n')
        index_deleted_message_0 = lines[0] 
        self.assertEqual(index_deleted_message_0, f"Deleting index {self.index_name}")
        index_deleted_message_1 = lines[1]
        self.assertEqual(index_deleted_message_1, f"Index {self.index_name} deleted successfully!")
    
    @classmethod
    def sort_test_methods(cls, testCaseClass, testCaseNames):
        """
        Sort test methods for better readability.
        """
        return sorted(testCaseNames)

if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = TestGooglePineconeIndexer.sort_test_methods
    unittest.main()
