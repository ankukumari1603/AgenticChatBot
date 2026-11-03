from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage

from src.langGraphAgenticAi.state.state import State

SYSTEM_PROMPT = """You are a helpful, accurate assistant.
Be clear when you are uncertain, and do not invent facts or sources.
When a web search tool is available, use it for current facts and include
relevant source links in your response.
"""


def with_system_prompt(messages: list[AnyMessage]) -> list[AnyMessage]:
    if messages and getattr(messages[0], "type", None) == "system":
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT), *messages]


class BasicChatbotNode:
    """Process the conversation and return a model response."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def process(self, state: State) -> dict[str, list[AnyMessage]]:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("The conversation contains no messages.")
        return {"messages": [self.model.invoke(with_system_prompt(messages))]}
