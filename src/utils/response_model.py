from typing import List, Union
from pydantic import BaseModel, Field

class Document(BaseModel):
    page_content: str = Field(..., description="The content of the page from the source document.")
    source: Union[float, int] = Field(..., description="The page number of the page_content in the document")
    title: str = Field(..., description="The title or URL of the source document.")


class QueryResult(BaseModel):
    query: str  = Field(..., description="The query that was submitted.")
    result: str = Field(..., description="The result of the query, including any retrieved information.")
    page: Union[float, int] = Field(..., description="The page number of the final result of the query.")
    source_documents: List[Document] = Field(..., description="A list of source documents related to the query.")

    @property
    def sources(self) -> List[Union[float, int]]:
        """
        Returns a list of the sources (page numbers) from the source documents.
        """
        return [doc.source for doc in self.source_documents]

    @property
    def titles(self) -> List[str]:
        """
        Returns a list of the titles from the source documents.
        """
        return [doc.title for doc in self.source_documents]

    @property
    def page_contents(self) -> List[str]:
        """
        Returns a list of the page contents from the source documents.
        """
        return [doc.page_content for doc in self.source_documents]

# Example 
# data = {
#     'query': 'how did RAG come up?',
#     'result': 'RAG came up as a language model that is more strongly grounded in '
#               'than BART and has been effective in Jeopardy question generation.\n'
#               '\n'
#               'Sources:\n'
#               '- https://arxiv.org/pdf/2005.11401.pdf (page 5.0)',
#     'page': 5.0,  # A
#     'source_documents': [
#         {
#             'page_content': 'page-content-where-the-response-is-from.\n10',
#             'source': 9.0,
#             'title': 'document-title'
#         },
#         {
#             'page_content': 'page-content-where-the-response-is-from.\n17',
#             'source': 5.0,
#             'title': 'document-title'
#         }
#     ]
# }

