import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.langGraphAgenticAi.LLMS.GroqLLM import (
    DEFAULT_GROQ_MODEL,
    GroqLLM,
    get_groq_model,
)
from src.langGraphAgenticAi.graph.graphbuilder import GraphBuilder
from src.langGraphAgenticAi.state.state import State
from src.langGraphAgenticAi.tools.search_tool import get_tools
from src.langGraphAgenticAi.ui.streamlit.chat_ui import _redact_secrets
from src.langGraphAgenticAi.ui.streamlit.text_summarizer import (
    STYLE_INSTRUCTIONS,
    summarize_text,
)


class AgentGraphTests(unittest.TestCase):
    def test_graph_runs_without_search_credentials(self) -> None:
        model = FakeListChatModel(responses=["Hello"])
        graph = GraphBuilder(model).setup_graph()
        result = graph.invoke({"messages": [HumanMessage(content="Hi")]})
        self.assertEqual(result["messages"][-1].content, "Hello")

    def test_graph_routes_tool_calls_and_returns_to_the_agent(self) -> None:
        @tool
        def fake_search(query: str) -> dict[str, str]:
            """Return a test search result."""
            return {"query": query, "result": "A test result."}

        model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "fake_search",
                            "args": {"query": "test"},
                            "id": "call-test",
                        }
                    ],
                ),
                AIMessage(content="The search found a test result."),
            ]
        )
        with (
            patch.object(
                FakeMessagesListChatModel,
                "bind_tools",
                new=lambda self, tools, **kwargs: self,
            ),
            patch(
                "src.langGraphAgenticAi.graph.graphbuilder.get_tools",
                return_value=[fake_search],
            ),
        ):
            graph = GraphBuilder(model, tavily_api_key="test-key").setup_graph()
            result = graph.invoke(
                {"messages": [HumanMessage(content="Search for a test result.")]},
                config={"recursion_limit": 10},
            )

        self.assertTrue(
            any(isinstance(message, ToolMessage) for message in result["messages"])
        )
        self.assertEqual(
            result["messages"][-1].content,
            "The search found a test result.",
        )

    def test_search_is_optional_without_a_tavily_key(self) -> None:
        self.assertEqual(get_tools(), [])

    def test_tavily_search_tool_is_created_with_a_key(self) -> None:
        tools = get_tools("test-key")
        self.assertEqual([item.name for item in tools], ["search_web"])

    def test_state_uses_a_typed_message_history(self) -> None:
        self.assertIn("messages", State.__annotations__)

    def test_groq_model_defaults_to_qwen_and_requires_a_key(self) -> None:
        self.assertEqual(DEFAULT_GROQ_MODEL, "qwen/qwen3.8-27b")
        with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
            GroqLLM("")

    def test_chat_and_summarizer_can_reuse_one_model_instance(self) -> None:
        fake_model = FakeListChatModel(responses=["Unused"])
        get_groq_model.clear()
        try:
            with patch.object(
                GroqLLM,
                "get_llm_model",
                return_value=fake_model,
            ) as create_model:
                first = get_groq_model("test-key", "test-model")
                second = get_groq_model("test-key", "test-model")

            self.assertIs(first, second)
            create_model.assert_called_once()
        finally:
            get_groq_model.clear()

    def test_error_log_message_redacts_credentials(self) -> None:
        safe_message = _redact_secrets(
            "request failed with groq-secret and tavily-secret",
            "groq-secret",
            "tavily-secret",
        )
        self.assertEqual(
            safe_message,
            "request failed with [REDACTED] and [REDACTED]",
        )

    def test_summarizer_supports_all_styles(self) -> None:
        for style in ("Short", "Detailed", "Bullet Points"):
            model = FakeListChatModel(responses=["Key ideas and conclusion."])
            self.assertEqual(
                summarize_text(model, "A long source text.", style),
                "Key ideas and conclusion.",
            )
        self.assertEqual(len(STYLE_INSTRUCTIONS), 3)

    def test_summarizer_rejects_empty_text_and_unknown_styles(self) -> None:
        model = FakeListChatModel(responses=["Unused"])
        with self.assertRaisesRegex(ValueError, "Enter text"):
            summarize_text(model, "  ", "Short")
        with self.assertRaisesRegex(ValueError, "summary styles"):
            summarize_text(model, "Some text.", "Unknown")


if __name__ == "__main__":
    unittest.main()
