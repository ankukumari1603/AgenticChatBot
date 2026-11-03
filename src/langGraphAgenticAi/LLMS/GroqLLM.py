import streamlit as st
from langchain_groq import ChatGroq

DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"


class GroqLLM:
    """Create the configured Qwen chat model hosted by Groq."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_GROQ_MODEL) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("GROQ_API_KEY is required to start a conversation.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Enter a valid Groq model name.")

        self.api_key = api_key.strip()
        self.model_name = model_name.strip()

    def get_llm_model(self) -> ChatGroq:
        return ChatGroq(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.7,
        )


@st.cache_resource(show_spinner=False)
def get_groq_model(api_key: str, model_name: str) -> ChatGroq:
    """Return the shared Groq model used by chat and summarization."""
    return GroqLLM(api_key, model_name).get_llm_model()
