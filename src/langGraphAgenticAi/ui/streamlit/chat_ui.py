import logging
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.langGraphAgenticAi.graph.graphbuilder import GraphBuilder
from src.langGraphAgenticAi.LLMS.GroqLLM import (
    DEFAULT_GROQ_MODEL,
    get_groq_model,
)
from src.langGraphAgenticAi.ui.streamlit.text_summarizer import (
    render_summary_result,
    render_text_summarizer,
)
from src.langGraphAgenticAi.ui.streamlit.ai_news_ui import (
    render_ai_news_report,
    render_ai_news_sidebar,
)

logger = logging.getLogger(__name__)


def _redact_secrets(message: str, *secrets: str | None) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _to_langchain_messages(
    history: list[dict[str, str]],
) -> list[HumanMessage | AIMessage]:
    messages: list[HumanMessage | AIMessage] = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return str(content)


def _clear_chat() -> None:
    st.session_state["messages"] = []


def render_chat_ui(
    groq_api_key: str | None,
    tavily_api_key: str | None,
    default_model: str = DEFAULT_GROQ_MODEL,
) -> None:
    st.session_state.setdefault("messages", [])

    with st.sidebar:
        st.subheader("Configuration")
        model_name = st.text_input(
            "Groq Qwen model",
            value=default_model,
            help=f"Groq model ID. The default is {DEFAULT_GROQ_MODEL}.",
        ).strip()
        st.caption(
            "Web search is enabled when TAVILY_API_KEY is configured."
            if tavily_api_key
            else "Add TAVILY_API_KEY to enable web search. Chat works without it."
        )
        st.button(
            "Clear chat",
            icon=":material/delete_sweep:",
            on_click=_clear_chat,
            width="stretch",
        )
        render_text_summarizer(
            groq_api_key,
            model_name or DEFAULT_GROQ_MODEL,
            get_groq_model,
        )
        render_ai_news_sidebar(
            groq_api_key,
            tavily_api_key,
            model_name or DEFAULT_GROQ_MODEL,
            get_groq_model,
        )

        if groq_api_key:
            st.success("Groq API key configured", icon=":material/lock:")
        else:
            st.warning(
                "Configure GROQ_API_KEY in your environment or Streamlit secrets to chat.",
                icon=":material/key_off:",
            )

    st.title("🤖 Agentic AI Chatbot")
    st.caption("Your intelligent AI assistant powered by Agentic AI.")
    render_ai_news_report()
    render_summary_result()

    for item in st.session_state["messages"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    with st.form("chat_form", clear_on_submit=True):
        input_column, send_column = st.columns(
            [7, 1],
            vertical_alignment="bottom",
            wrap=True,
        )
        prompt = input_column.text_input(
            "Message",
            placeholder="Ask me anything...",
            label_visibility="collapsed",
            disabled=not groq_api_key,
            key="chat_prompt",
        )
        submitted = send_column.form_submit_button(
            "Send",
            icon=":material/send:",
            type="primary",
            width="stretch",
            disabled=not groq_api_key,
            key="send_message",
        )

    if not submitted or not prompt.strip():
        return

    prompt = prompt.strip()
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                graph_builder = GraphBuilder(
                    get_groq_model(
                        groq_api_key,
                        model_name or DEFAULT_GROQ_MODEL,
                    ),
                    tavily_api_key=tavily_api_key,
                )
                graph = graph_builder.setup_graph()
                result = graph.invoke(
                    {"messages": _to_langchain_messages(st.session_state["messages"])}
                )
            response = _message_text(result["messages"][-1].content).strip()
            if not response:
                raise RuntimeError("The model returned an empty response.")
            st.markdown(response)
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )
        except Exception as exc:
            safe_message = _redact_secrets(
                str(exc),
                groq_api_key,
                tavily_api_key,
            )
            logger.error(
                "Agent request failed (%s): %s",
                type(exc).__name__,
                safe_message,
            )
            error_message = (
                "The assistant could not complete this request. Verify the Groq "
                "key and selected model, then try again."
            )
            st.error(error_message)
            st.caption(f"Error type: {type(exc).__name__}")
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_message}
            )
