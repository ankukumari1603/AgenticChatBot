from collections.abc import Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool

from src.langGraphAgenticAi.nodes.basic_chatbot_node import with_system_prompt
from src.langGraphAgenticAi.state.state import State


class ChatbotWithToolNode:
    """Create a model node that can request the configured tools."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def create_chatbot(
        self, tools: Sequence[BaseTool]
    ) -> Callable[[State], dict[str, list[AnyMessage]]]:
        model_with_tools = self.model.bind_tools(tools)

        def chatbot_node(state: State) -> dict[str, list[AnyMessage]]:
            messages = state.get("messages", [])
            if not messages:
                raise ValueError("The conversation contains no messages.")
            response = model_with_tools.invoke(with_system_prompt(messages))
            return {"messages": [response]}

        return chatbot_node
