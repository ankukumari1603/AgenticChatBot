# Agentic AI Chatbot

A modular Streamlit chatbot that uses Qwen hosted on Groq and LangGraph for
agent orchestration. Configure a Tavily key to let the agent search the web.

## Features

- Dark Streamlit chat interface with separate user and assistant messages,
  session conversation history, a loading indicator, a Send button, and Clear
  chat control.
- Groq Qwen model (`qwen/qwen3.8-27b` by default).
- LangGraph graph with typed message state, chatbot nodes, conditional tool
  routing, and a tool execution node.
- Optional Tavily web search. The chatbot still works without a Tavily key.
- Sidebar text summarizer with Short, Detailed, and Bullet Points styles,
  powered by the same configured Groq Qwen model as chat.
- AI News reports for recent day, week, and month, collected with Tavily,
  deduplicated, summarized with Groq, displayed as a table, and saved as Markdown.
- Environment-variable and Streamlit Secrets support for credentials.
- Ready for local use and Streamlit Community Cloud.

## Architecture

The root `app.py` is a small Streamlit entry point. The application is split
into the requested modules:

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── runtime.txt
├── .streamlit/
│   └── config.toml
├── src/
│   ├── __init__.py
│   └── langGraphAgenticAi/
│       ├── __init__.py
│       ├── main.py
│       ├── LLMS/
│       │   ├── __init__.py
│       │   └── GroqLLM.py
│       ├── graph/
│       │   ├── __init__.py
│       │   └── graphbuilder.py
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── basic_chatbot_node.py
│       │   └── chatbot_with_Tool_node.py
│       ├── state/
│       │   ├── __init__.py
│       │   └── state.py
│       ├── tools/
│       │   ├── __init__.py
│       │   └── search_tool.py
│       ├── news/
│       │   ├── __init__.py
│       │   └── ai_news_summarizer.py
│       └── ui/
│           ├── __init__.py
│           └── streamlit/
│               ├── __init__.py
│               ├── ai_news_ui.py
│               ├── chat_ui.py
│               └── text_summarizer.py
├── AInews/
│   └── (Markdown reports are created when generated)
└── tests/
    ├── test_agent.py
    └── test_ai_news.py
```

`main.py` loads local environment values and Streamlit secrets, then launches
the UI. `GraphBuilder` uses a basic chatbot node when Tavily is not configured.
When Tavily is available, the graph binds the search tool to the model and
routes tool calls through LangGraph's `ToolNode`; the model then uses the tool
results to finish its response.

The Text Summarizer appears separately in the sidebar. Paste text, choose a
style, and select **Summarize**. Its result is displayed in the main interface.
Chat and summarization share the configured Groq model instance; no additional
provider or dependency is required.

The **AI News** sidebar section generates Daily, Weekly, and Monthly AI news
reports. Tavily's news search scope limits results to the selected recent
period. Duplicate source URLs/titles are removed; article publication dates
come only from Tavily metadata and are labeled `Not provided by source` when
missing. Each generated report is displayed as a linked table and written to
`AInews/daily_summary.md`, `AInews/weekly_summary.md`, or
`AInews/monthly_summary.md`. Both `GROQ_API_KEY` and `TAVILY_API_KEY` are
required to generate reports.

## Requirements

- Python 3.11 or newer
- A Groq API key (required for chat)
- A Tavily API key (optional; required only for web search)

## Install

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure keys

Copy `.env.example` to `.env`, then edit the local copy and add your own
credentials. `.env` is ignored by Git.

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b
TAVILY_API_KEY=your_tavily_api_key
```

PowerShell setup:

```powershell
Copy-Item .env.example .env
notepad .env
```

`GROQ_API_KEY` is required. `TAVILY_API_KEY` is optional. The model can also be
changed in the app sidebar. Do not commit keys or `.streamlit/secrets.toml`.
Leave `TAVILY_API_KEY` blank to run without web search.

## Run locally

```powershell
streamlit run app.py
```

Without an activated environment, use:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Tests

```powershell
python -m unittest discover -s tests -v
```

The graph and news tests use fake models and mocked Tavily results; they do not
make live Groq or Tavily API calls.

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub with `app.py` as the entry point and
   `requirements.txt` at the repository root.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from the repository and choose `app.py` as the main file.
3. In the app's Secrets settings, add your own keys:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   GROQ_MODEL = "qwen/qwen3.8-27b"
   TAVILY_API_KEY = "your_tavily_api_key"
   ```

4. Save the secrets and deploy. Omit `TAVILY_API_KEY` if you do not need search.

Never put real credentials in GitHub or the source files.
