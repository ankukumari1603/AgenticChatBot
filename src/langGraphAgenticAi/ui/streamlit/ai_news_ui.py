import logging
from collections.abc import Callable
from pathlib import Path

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel
from tavily import TavilyClient

from src.langGraphAgenticAi.news.ai_news_summarizer import (
    REPORTS,
    NewsArticle,
    generate_news_report,
    save_news_report,
)

logger = logging.getLogger(__name__)
REPORT_TYPES = tuple(REPORTS)


def _redact_secrets(message: str, *secrets: str | None) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def render_ai_news_sidebar(
    groq_api_key: str | None,
    tavily_api_key: str | None,
    model_name: str,
    model_factory: Callable[[str, str], BaseChatModel],
) -> None:
    with st.sidebar.expander("AI News", icon=":material/newspaper:"):
        report_type = st.selectbox(
            "Report type",
            REPORT_TYPES,
            key="ai_news_report_type",
        )
        st.caption("Reports cover the recent day, week, or month.")
        if not tavily_api_key:
            st.info("Configure TAVILY_API_KEY to collect news articles.")
        if not groq_api_key:
            st.info("Configure GROQ_API_KEY to summarize articles.")

        if st.button(
            "Generate AI news report",
            icon=":material/auto_awesome:",
            width="stretch",
            key="generate_ai_news",
        ):
            if not tavily_api_key:
                st.warning("Add TAVILY_API_KEY before generating a news report.")
            elif not groq_api_key:
                st.warning("Add GROQ_API_KEY before generating a news report.")
            else:
                try:
                    with st.spinner(f"Collecting and summarizing {report_type.lower()}…"):
                        articles = generate_news_report(
                            report_type,
                            TavilyClient(api_key=tavily_api_key),
                            model_factory(groq_api_key, model_name),
                        )
                        report_path = save_news_report(report_type, articles)
                    st.session_state["ai_news_report"] = [
                        article.as_row() for article in articles
                    ]
                    st.session_state["ai_news_generated_type"] = report_type
                    st.session_state["ai_news_report_path"] = str(report_path)
                    st.success(f"Saved report to `{report_path}`.")
                except Exception as exc:
                    safe_message = _redact_secrets(
                        str(exc),
                        groq_api_key,
                        tavily_api_key,
                    )
                    logger.error(
                        "AI news report generation failed (%s): %s",
                        type(exc).__name__,
                        safe_message,
                    )
                    st.error(
                        "The AI news report could not be generated. Check your "
                        "Groq and Tavily configuration, then try again."
                    )


def render_ai_news_report() -> None:
    rows = st.session_state.get("ai_news_report")
    if rows is None:
        return
    report_type = st.session_state.get(
        "ai_news_generated_type", "AI News Summary"
    )
    st.subheader(report_type)
    if not rows:
        st.info("No usable articles were found for this report period.")
        return
    st.dataframe(
        rows,
        column_config={
            "Article title": st.column_config.TextColumn("Article title"),
            "Publication date": st.column_config.TextColumn("Publication date"),
            "Short summary": st.column_config.TextColumn("Short summary"),
            "Source": st.column_config.LinkColumn(
                "Original source",
                display_text="Open article",
            ),
        },
        hide_index=True,
        width="stretch",
        alt="AI news articles with publication dates, summaries, and original source links.",
    )
    report_path = st.session_state.get("ai_news_report_path")
    if report_path:
        st.caption(f"Markdown report saved to `{Path(report_path).as_posix()}`.")
