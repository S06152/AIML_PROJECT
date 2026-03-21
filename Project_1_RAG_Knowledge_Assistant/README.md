# 🤖 SmartDoc AI — RAG Knowledge Assistant

> An enterprise-grade **Retrieval-Augmented Generation (RAG)** application that enables intelligent document Q&A across multiple file formats and vector databases, powered by **LangChain**, **Groq LLMs**, and **Streamlit**.

---

## 📌 Table of Contents

- [🤖 SmartDoc AI — RAG Knowledge Assistant](#-smartdoc-ai--rag-knowledge-assistant)
  - [📌 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Architecture](#architecture)
  - [Tech Stack](#tech-stack)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Supported File Formats](#supported-file-formats)
  - [Supported Vector Databases](#supported-vector-databases)
  - [RAG Pipeline Details](#rag-pipeline-details)
    - [1. Document Ingestion](#1-document-ingestion)
    - [2. Chunking](#2-chunking)
    - [3. Embedding](#3-embedding)
    - [4. Indexing](#4-indexing)
    - [5. Retrieval](#5-retrieval)
    - [6. Generation](#6-generation)

---

## Overview

**SmartDoc AI** is a production-ready RAG Knowledge Assistant that allows users to upload documents, automatically index them into a vector database, and perform context-aware Q&A using Large Language Models. The system retrieves the most relevant document chunks and generates accurate, source-cited answers — making it ideal for enterprise knowledge management, compliance review, and internal documentation search.

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-Format Ingestion** | Supports PDF, DOCX, PPTX, CSV, TXT, and XLSX file uploads |
| **Pluggable Vector Stores** | Switch between FAISS, Chroma, Pinecone, and AstraDB via UI |
| **Configurable LLM** | Choose from multiple Groq-hosted models with adjustable temperature and token limits |
| **Auto Re-Indexing** | Detects file or vector DB changes and triggers automatic re-indexing |
| **Chat Interface** | Conversational Q&A with persistent chat history using Streamlit session state |
| **Source Citations** | Responses include document source and page references when available |
| **Factory Design Patterns** | Loader and vector store selection via factory classes for clean extensibility |
| **Centralized Configuration** | All settings managed through a single `config.ini` file |
| **Structured Logging & Exception Handling** | Custom logger and exception classes for production-level observability |

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│             (File Upload, Sidebar Config, Chat)              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Ingestion Layer                            │
│   LoaderFactory → PDF / DOCX / PPTX / CSV / TXT / XLSX       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Chunking Layer                             │
│         RecursiveCharacterTextSplitter (512 / 50)            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  Embedding Layer                             │
│           HuggingFace (all-MiniLM-L6-v2)                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               Vector Store Layer                             │
│   VectorStoreFactory → FAISS / Chroma / Pinecone / AstraDB   │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  Retrieval Layer                             │
│        Similarity Search (Top-K configurable)                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    QA Chain Layer                            │
│   Retriever → Context Formatter → Prompt → Groq LLM → Output │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM Orchestration** | LangChain, LangChain-Core |
| **LLM Provider** | Groq (ChatGroq) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector Databases** | FAISS, ChromaDB, Pinecone, AstraDB |
| **Document Parsing** | PyPDFLoader, Docx2txt, python-pptx, UnstructuredExcelLoader, CSVLoader, TextLoader |
| **Configuration** | ConfigParser (INI-based) |
| **Environment Management** | python-dotenv |
| **Language** | Python 3.12 |

---

## Project Structure

```
Project_1_RAG_Knowledge_Assistant/
├── app.py                          # Application entry point
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py            # Main Streamlit UI controller
│
├── src/
│   ├── config/
│   │   ├── config.ini              # Centralized configuration file
│   │   └── settings.py             # Config parser with typed getters
│   │
│   ├── ingestion/                  # Document loaders (Factory Pattern)
│   │   ├── loader_factory.py       # MIME-type based loader selection
│   │   ├── pdf_loader.py           # PDF ingestion (PyPDFLoader)
│   │   ├── docx_loader.py          # DOCX ingestion (Docx2txt)
│   │   ├── ppt_loader.py           # PPTX ingestion (python-pptx)
│   │   ├── ppt_loader_unstructured.py  # PPTX via Unstructured
│   │   ├── csv_loader.py           # CSV ingestion
│   │   ├── txt_loader.py           # TXT ingestion
│   │   └── xlsx_loader.py          # Excel ingestion (Unstructured)
│   │
│   ├── chunking/
│   │   └── chunk.py                # RecursiveCharacterTextSplitter
│   │
│   ├── embedding/
│   │   └── embedding.py            # HuggingFace embedding manager (singleton)
│   │
│   ├── vectorstore/                # Vector DB integrations (Factory Pattern)
│   │   ├── store_factory.py        # Vector store selection factory
│   │   ├── faiss_store.py          # FAISS (in-memory)
│   │   ├── chroma_store.py         # ChromaDB
│   │   ├── pinecone_store.py       # Pinecone (serverless cloud)
│   │   ├── astradb_store.py        # DataStax AstraDB
│   │  
│   ├── retrieval/
│   │   └── retriever.py            # Similarity-based retriever wrapper
│   │
│   ├── chain/
│   │   ├── qa_chain.py             # Full RAG chain (Retriever → LLM → Output)
│   │   └── prompt_templates.py     # System & human prompt templates
│   │
│   └── utils/
│       ├── logger.py               # Timestamped file-based logging
│       └── exception.py            # Custom exception with traceback details
│
└── data/                           # Sample test data
    ├── pdf/
    ├── word_files/
    ├── PPT/
    ├── structured_files/           # products.csv
    ├── text_files/                 # python_intro.txt, proposal.txt, etc.
```
---

## Usage

1. **Launch** the app with `streamlit run app.py`
2. **Configure** LLM model, temperature, max tokens, and vector database from the sidebar
3. **Upload** one or more documents (PDF, DOCX, PPTX, CSV, TXT, XLSX)
4. **Wait** for automatic document processing and indexing (progress shown via spinner)
5. **Ask questions** in the chat input — the system retrieves relevant chunks and generates cited answers
6. **Change** files or vector DB selection at any time — the system auto-detects changes and re-indexes

---

## Supported File Formats

| Format | Loader Class | Library |
|---|---|---|
| PDF | [`PDFLoader`](src/ingestion/pdf_loader.py) | PyPDFLoader |
| DOCX | [`DocxLoader`](src/ingestion/docx_loader.py) | Docx2txtLoader |
| PPTX | [`PPTLoader`](src/ingestion/ppt_loader.py) | python-pptx |
| CSV | [`CSVLoaderWrapper`](src/ingestion/csv_loader.py) | LangChain CSVLoader |
| TXT | [`TxtLoader`](src/ingestion/txt_loader.py) | LangChain TextLoader |
| XLSX | [`XlsxLoader`](src/ingestion/xlsx_loader.py) | UnstructuredExcelLoader |

File type detection is handled automatically by [`LoaderFactory`](src/ingestion/loader_factory.py) using MIME type mapping.

---

## Supported Vector Databases

| Vector DB | Class | Type |
|---|---|---|
| **FAISS** | [`FaissVectorStore`](src/vectorstore/faiss_store.py) | Local (in-memory) |
| **ChromaDB** | [`ChromaVectorStore`](src/vectorstore/chroma_store.py) | Local (persistent/in-memory) |
| **Pinecone** | [`PineconeVector`](src/vectorstore/pinecone_store.py) | Cloud (serverless) |
| **AstraDB** | [`AstraDBVector`](src/vectorstore/astradb_store.py) | Cloud (DataStax) |

Vector store selection is handled by [`VectorStoreFactory`](src/vectorstore/store_factory.py) using the Factory design pattern.

---

## RAG Pipeline Details

### 1. Document Ingestion
Documents are loaded via format-specific loaders, each handling temporary file creation and cleanup.

### 2. Chunking
Documents are split using [`RecursiveCharacterTextSplitter`](src/chunking/chunk.py) with configurable chunk size (default: 512) and overlap (default: 50), using hierarchical separators (`\n\n`, `\n`, `. `, ` `).

### 3. Embedding
Chunks are vectorized using HuggingFace's `all-MiniLM-L6-v2` model via the [`EmbeddingManager`](src/embedding/embedding.py) singleton.

### 4. Indexing
Vectors are stored in the selected vector database through the [`VectorStoreFactory`](src/vectorstore/store_factory.py).

### 5. Retrieval
User queries trigger a similarity search (Top-K = 5) via the [`Retriever`](src/retrieval/retriever.py) wrapper.

### 6. Generation
The [`QAChain`](src/chain/qa_chain.py) composes a LangChain LCEL pipeline:

```
User Query → Retriever → Context Formatter → Prompt Template → Groq LLM → StrOutputParser → Response
```

The prompt template ([`prompt_templates.py`](src/chain/prompt_templates.py)) instructs the LLM to answer strictly from context and cite sources.

---
