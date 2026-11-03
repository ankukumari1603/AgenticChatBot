from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from src.langGraphAgenticAi.nodes.basic_chatbot_node import BasicChatbotNode
from src.langGraphAgenticAi.nodes.chatbot_with_Tool_node import (
    ChatbotWithToolNode,
)
from src.langGraphAgenticAi.state.state import State
from src.langGraphAgenticAi.tools.search_tool import (
    create_tool_node,
    get_tools,
)


class GraphBuilder:
    """Build a chat graph with optional Tavily search."""

    def __init__(
        self,
        model: BaseChatModel,
        tavily_api_key: str | None = None,
    ) -> None:
        if model is None:
            raise ValueError("A chat model is required to build the graph.")
        self.model = model
        self.tavily_api_key = tavily_api_key

    def setup_graph(self):
        graph = StateGraph(State)
        tools = get_tools(self.tavily_api_key)

        if tools:
            graph.add_node(
                "chatbot",
                ChatbotWithToolNode(self.model).create_chatbot(tools),
            )
            graph.add_node("tools", create_tool_node(tools))
            graph.add_edge(START, "chatbot")
            graph.add_conditional_edges("chatbot", tools_condition)
            graph.add_edge("tools", "chatbot")
        else:
            graph.add_node("chatbot", BasicChatbotNode(self.model).process)
            graph.add_edge(START, "chatbot")
            graph.add_edge("chatbot", END)

        return graph.compile()
