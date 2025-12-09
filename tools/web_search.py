from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 1. Define Schema
class SearchInput(BaseModel):
    query: str = Field(description="The search query for lists, rankings, or admission guides.")

search_tool_instance = DuckDuckGoSearchRun(region="pk-en")

# 2. Define Tool with Schema
@tool("web_search", args_schema=SearchInput)
def web_search(query: str):
    """
    Search the web for general advice, university lists, or admission deadlines
    when the database does not have specific info.
    """
    try:
        return search_tool_instance.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"