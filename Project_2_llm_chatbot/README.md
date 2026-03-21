# 🤖 Enterprise LLM Chatbot (Streamlit + LangChain + Groq)

> A lightweight, modular **Enterprise Q&A Chatbot** built with **Streamlit**, **LangChain**, and **Groq LLMs** — designed with clean layered architecture for easy extensibility and maintainability.

---

## 📌 Table of Contents

- [🤖 Enterprise LLM Chatbot (Streamlit + LangChain + Groq)](#-enterprise-llm-chatbot-streamlit--langchain--groq)
  - [📌 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Architecture](#architecture)
  - [Tech Stack](#tech-stack)
  - [Project Structure](#project-structure)

---

## Overview

**Enterprise LLM Chatbot** is a production-ready conversational Q&A application that leverages Groq-hosted Large Language Models through LangChain's orchestration framework. Users can interact with the chatbot via a clean Streamlit chat interface, configure model parameters in real-time, and receive concise, accurate responses powered by state-of-the-art LLMs.

---

## Key Features

| Feature | Description |
|---|---|
| **💬 Chat Interface** | Streamlit-based conversational UI with persistent chat history |
| **🧱 Modular Architecture** | Clean separation into UI, Service, and Core layers |
| **🧠 Multiple LLM Models** | Choose from 6 Groq-hosted models via sidebar dropdown |
| **🎛 Configurable Parameters** | Adjust temperature (0.0–1.0) and max tokens (50–2000) in real-time |
| **🔐 Secure API Key Handling** | API keys managed via Streamlit secrets (not hardcoded) |
| **🔗 LangChain Pipeline** | Prompt → LLM → Output Parser chain using LangChain Core |
| **✅ Input Validation** | User input validated before processing |

---

## Architecture

```text
┌──────────────────────────────────────────────────┐
│              Streamlit UI Layer                  │
│     (ChatbotApp, Sidebar, Chat Interface)        │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│              Service Layer                       │
│          (ChatbotService orchestration)          │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│               Core Layer                         │
│   GroqLLMClient → PromptFactory → ResponseChain  │
│         (LLM + Prompt + Chain)                   │
└──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM Orchestration** | LangChain, LangChain-Core |
| **LLM Provider** | Groq (ChatGroq) |
| **Language** | Python 3.12 |

---

## Project Structure

```
Project_2_llm_chatbot/
├── app.py                        # Application entry point
├── requirements.txt              # Python dependencies
├── .env_example                  # Environment variable template
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   └── settings.py           # Centralized application settings
│   │
│   ├── core/
│   │   ├── llm_client.py         # Groq LLM client wrapper
│   │   ├── prompt.py             # Prompt template factory
│   │   └── response_chain.py     # LangChain processing pipeline
│   │
│   ├── services/
│   │   └── chatbot_service.py    # High-level chatbot orchestration
│   │
│   ├── ui/
│   │   ├── pages.py              # Main Streamlit application class
│   │   └── sidebar.py            # Sidebar UI component
│   │
│   └── utils/
│       └── validators.py         # Input validation utilities
```



