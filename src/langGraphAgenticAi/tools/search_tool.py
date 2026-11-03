from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient


def get_tools(api_key: str | None = None) -> list[BaseTool]:
    """Return Tavily search when a key is configured; otherwise no tools."""
    if not api_key or not api_key.strip():
        return []

    client = TavilyClient(api_key=api_key.strip())

    @tool
    def search_web(query: str) -> dict[str, Any]:
        """Search the web for current information and return linked sources."""
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
        return {
            "query": query,
            "answer": response.get("answer"),
            "results": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
                for item in response.get("results", [])
            ],
        }

    return [search_web]


def create_tool_node(tools) -> ToolNode:
    if not tools:
        raise ValueError("At least one tool is required.")
    return ToolNode(tools=tools)
