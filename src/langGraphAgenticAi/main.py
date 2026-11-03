import os

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from src.langGraphAgenticAi.LLMS.GroqLLM import DEFAULT_GROQ_MODEL
from src.langGraphAgenticAi.ui.streamlit.chat_ui import render_chat_ui

load_dotenv(override=False)


def _get_credential(name: str) -> str | None:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_langgraph_agenticai_app() -> None:
    st.set_page_config(
        page_title="Agentic AI Chatbot",
        page_icon=":material/auto_awesome:",
        layout="centered",
    )
    render_chat_ui(
        groq_api_key=_get_credential("GROQ_API_KEY"),
        tavily_api_key=_get_credential("TAVILY_API_KEY"),
        default_model=_get_credential("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
    )
