import logging
from collections.abc import Callable

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SUMMARY_STYLES = ("Short", "Detailed", "Bullet Points")

STYLE_INSTRUCTIONS = {
    "Short": "Write a concise summary in 2–4 sentences.",
    "Detailed": (
        "Write a thorough, well-organized summary that covers the main ideas, "
        "important supporting points, and conclusions."
    ),
    "Bullet Points": (
        "Summarize the main idea and important key points as clear, concise "
        "bullet points."
    ),
}


def summarize_text(model: BaseChatModel, text: str, style: str) -> str:
    """Summarize text with the supplied, already-configured chat model."""
    if not text.strip():
        raise ValueError("Enter text to summarize.")
    if style not in STYLE_INSTRUCTIONS:
        raise ValueError(f"Choose one of these summary styles: {SUMMARY_STYLES}.")

    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Summarize only the source text. Preserve important names, "
                    "facts, and conclusions. Do not invent details. "
                    f"{STYLE_INSTRUCTIONS[style]}"
                )
            ),
            HumanMessage(content=text),
        ]
    )
    content = response.content
    if isinstance(content, str):
        summary = content.strip()
    elif isinstance(content, list):
        summary = "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
    else:
        summary = str(content).strip()

    if not summary:
        raise RuntimeError("The model returned an empty summary.")
    return summary


def _redact_secrets(message: str, *secrets: str | None) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def render_text_summarizer(
    api_key: str | None,
    model_name: str,
    model_factory: Callable[[str, str], BaseChatModel],
) -> None:
    """Render the sidebar controls and retain the generated summary."""
    with st.sidebar.expander("Text Summarizer", icon=":material/summarize:"):
        text = st.text_area(
            "Text to summarize",
            height=180,
            placeholder="Paste or type the text you want summarized.",
            key="summarizer_input",
        )
        style = st.segmented_control(
            "Summary style",
            options=SUMMARY_STYLES,
            default="Short",
            key="summarizer_style",
        )
        summarize_clicked = st.button(
            "Summarize",
            icon=":material/auto_awesome:",
            width="stretch",
            key="summarize_text",
        )

        if summarize_clicked:
            st.session_state.pop("text_summary", None)
            if not text.strip():
                st.warning("Enter some text before summarizing.")
            elif not api_key:
                st.error("Configure GROQ_API_KEY before summarizing.")
            elif style is None:
                st.warning("Choose a summary style.")
            else:
                try:
                    with st.spinner("Summarizing text…"):
                        summary = summarize_text(
                            model_factory(api_key, model_name),
                            text,
                            style,
                        )
                    st.session_state["text_summary"] = summary
                    st.session_state["text_summary_style"] = style
                except Exception as exc:
                    safe_message = _redact_secrets(
                        str(exc),
                        api_key,
                    )
                    logger.error(
                        "Text summarization failed (%s): %s",
                        type(exc).__name__,
                        safe_message,
                    )
                    st.error(
                        "The text could not be summarized. Check your Groq "
                        "configuration and try again."
                    )


def render_summary_result() -> None:
    summary = st.session_state.get("text_summary")
    if summary:
        style = st.session_state.get("text_summary_style", "Short")
        with st.container(border=True):
            st.subheader(f"{style} summary")
            st.markdown(summary)
