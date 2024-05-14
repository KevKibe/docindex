from pydantic import BaseModel, Field
from typing import Dict, Union

class Page(BaseModel):
    page_content: str = Field(..., description="The content of the page")
    metadata: Dict[str, Union[str, int]] = Field(..., description="Metadata about the document")
    page: int = Field(..., description="The page of the content")
    source: Union[str, int] = Field(..., description="The source url of the document")