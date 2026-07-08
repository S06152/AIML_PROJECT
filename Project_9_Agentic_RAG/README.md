# 📚 Agentic RAG Knowledge Assistant

An intelligent Retrieval-Augmented Generation (RAG) system with **FastAPI backend** and **Streamlit frontend** — deployable on **Streamlit Cloud** with zero manual setup.

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                   Streamlit Cloud (Single Process)                 │
│                                                                    │
│  ┌───────────────────┐   localhost   ┌───────────────────────────┐ │
│  │ Streamlit Frontend │ ◄──────────► │  FastAPI Backend (thread) │ │
│  │                    │  POST/GET    │                           │ │
│  │ - Upload PDFs      │ ──────────► │  - PDF Ingestion          │ │
│  │ - Ask Questions    │ ──────────► │  - Embeddings + VectorDB  │ │
│  │ - View Responses   │ ◄────────── │  - Agentic RAG Pipeline   │ │
│  │ - Chat History     │             │  - Tool Selection + LLM   │ │
│  └───────────────────┘              └───────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

When `streamlit run streamlit_app.py` is executed:
1. FastAPI backend starts **automatically** in a background thread.
2. Streamlit frontend launches and communicates with the backend via `localhost`.
3. **No separate server or terminal needed.**

## 🚀 Deploy on Streamlit Cloud

### Step 1: Push Code to GitHub

```bash
git add .
git commit -m "Add Agentic RAG application"
git push origin main
```

### Step 2: Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set the **Main file path** to: `streamlit_app.py`
4. Add **Secrets** in Streamlit Cloud dashboard:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   TAVILY_API_KEY = "your-tavily-api-key"
   ```
5. Click **Deploy**

That's it! Users can now upload documents and ask questions from the deployed app.

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This single command starts both the FastAPI backend and Streamlit frontend automatically.

## 📡 API Endpoints

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /health   | Health check & indexing status        |
| POST   | /upload   | Upload and index PDF documents        |
| POST   | /query    | Submit a question for RAG processing  |

### Upload Documents

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.pdf" \
  -F "llm_model=llama-3.3-70b-versatile" \
  -F "groq_api_key=your-key"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "llm_model": "llama-3.3-70b-versatile"}'
```

## 🗂️ Project Structure

```
Project_9_Agentic_RAG/
├── app.py                      # FastAPI backend entry point
├── streamlit_app.py            # Streamlit frontend entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── src/
    ├── api/
    │   └── fastapi_app.py      # FastAPI routes & backend logic
    ├── ui/
    │   ├── streamlit_app.py    # Legacy standalone Streamlit app
    │   └── streamlit_frontend.py # New Streamlit frontend (API client)
    ├── agents/
    │   └── agents.py           # LLM Agent with tool selection
    ├── config/
    │   ├── config.ini          # Application configuration
    │   └── settings.py         # Config reader
    ├── embedding/
    │   └── embedding.py        # HuggingFace embedding manager
    ├── graph/
    │   └── workflow_graph.py   # LangGraph workflow builder
    ├── ingestion/
    │   └── pdf_loader.py       # PDF document loader
    ├── models/
    │   └── state.py            # LangGraph state model
    ├── tools/
    │   ├── retriever_tool.py   # Vector DB retriever tool
    │   ├── arxiv_tool.py       # ArXiv search tool
    │   ├── tavily_tool.py      # Tavily web search tool
    │   ├── wiki_tool.py        # Wikipedia search tool
    │   └── tool_registry.py    # Central tool registry
    ├── utils/
    │   ├── exception.py        # Custom exception handler
    │   └── logger.py           # Logging configuration
    └── vectorstore/
        └── chroma_store.py     # ChromaDB vector store
```

## 🔧 Features

- **Document Upload**: Upload PDF documents from Streamlit Cloud
- **AI-Powered Q&A**: Ask questions and get intelligent answers
- **Multi-Tool Agent**: Automatically selects the best tool (Vector DB, Wikipedia, ArXiv, Tavily)
- **Chat History**: Maintains conversation history in the session
- **Configurable LLM**: Choose model, temperature, and token limits
- **Health Monitoring**: Real-time backend connection status
