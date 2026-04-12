# 🚗 AUTOSAR SWS Multi-Agent Software Development System

A production-ready, LangChain + LangGraph powered multi-agent AI system that reads
AUTOSAR Classic Platform SWS (Software Specification) documents and automatically
generates a complete software development lifecycle — from requirements to code review.

---

## 📋 Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Technology Stack](#technology-stack)
5. [Bug Fixes Applied](#bug-fixes-applied)
6. [Step-by-Step Explanation](#step-by-step-explanation)
7. [Installation & Setup](#installation--setup)
8. [How to Use](#how-to-use)
9. [Agent Descriptions](#agent-descriptions)
10. [RAG Pipeline Explained](#rag-pipeline-explained)
11. [Artifact Storage](#artifact-storage)
12. [Configuration Reference](#configuration-reference)
13. [OOP Design Patterns Used](#oop-design-patterns-used)
14. [Common Interview Questions](#common-interview-questions)

---

## What This System Does

This system acts as a **virtual AUTOSAR software development team**. You upload an
AUTOSAR SWS PDF (e.g., the COM module specification), describe what you need in plain
English, and 5 specialist AI agents automatically produce:

| Agent | Output Artifact | Saved To |
|-------|----------------|----------|
| Product Manager | Requirements specification (SWS-cited) | `01_requirements/` |
| Software Architect | System architecture + API design | `02_architecture/` |
| Developer | MISRA-C compliant source code | `03_source_code/` |
| QA Engineer | pytest test suite | `04_test_cases/` |
| Code Reviewer | MISRA-C review report | `05_reviews/` |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌──────────────┐           ┌───────────────────────────────┐   │
│  │  Sidebar     │           │     Main Panel                │   │
│  │  - API Key   │           │  Chat Input → User Request    │   │
│  │  - LLM Model │           │  Tabbed Results Display       │   │
│  │  - PDF Upload│           │  Download Buttons             │   │
│  └──────┬───────┘           └───────────────┬───────────────┘   │
└─────────┼─────────────────────────────────── ┼ ────────────────-┘
          │                                     │
          ▼                                     ▼
┌─────────────────────┐              ┌──────────────────────────┐
│   RAG Pipeline      │              │   LangGraph Workflow      │
│                     │              │                           │
│  PDFLoader          │   context    │  START                    │
│    ↓ text+table+img │ ─────────→  │    ↓                      │
│  ChunkingStrategy   │              │  ProductManagerAgent      │
│    ↓ 1200-char chunks│             │    ↓ product_spec         │
│  EmbeddingManager   │              │  ArchitectAgent           │
│    ↓ 384-dim vectors│              │    ↓ architecture         │
│  ChromaVectorStore  │              │  DeveloperAgent           │
│    ↓ indexed        │              │    ↓ code                 │
│  Retriever          │              │  QAAgent                  │
│    ↓ top-K chunks   │              │    ↓ tests                │
│  QAChain (LCEL)     │              │  CodeReviewAgent          │
│    ↓ grounded answer│              │    ↓ review               │
└─────────────────────┘              │  END                      │
                                     └──────────────────────────┘
                                                  │
                                                  ▼
                                     ┌──────────────────────────┐
                                     │   ArtifactSaver           │
                                     │   artifacts/<session>/    │
                                     │   01_requirements/        │
                                     │   02_architecture/        │
                                     │   03_source_code/         │
                                     │   04_test_cases/          │
                                     │   05_reviews/             │
                                     └──────────────────────────┘
```

---

## Project Structure

```
autosar_mas/
├── app.py                              # Streamlit entry point (run this)
├── requirements.txt                    # Python dependencies
├── .env.example                        # API key template
├── .streamlit/
│   └── secrets.toml                    # Groq API key (gitignored)
│
├── artifacts/                          # AUTO-CREATED per workflow run
│   └── 2026-04-12_15-30-00/
│       ├── 01_requirements/
│       │   └── COM_Requirements.md
│       ├── 02_architecture/
│       │   └── COM_Architecture.md
│       ├── 03_source_code/
│       │   └── COM_Source_Code.md
│       ├── 04_test_cases/
│       │   └── COM_Test_Cases.md
│       └── 05_reviews/
│           └── COM_Review_Report.md
│
└── src/
    ├── __init__.py
    │
    ├── agents/                         # 5 specialist AI agents
    │   ├── __init__.py
    │   ├── base_agent.py               # Abstract base: LCEL pipeline + ABC
    │   ├── product_manager.py          # Requirements generation (BUG FIXED)
    │   ├── architect.py                # Architecture design
    │   ├── developer.py                # Source code generation
    │   ├── qa.py                       # Test case generation
    │   └── code_reviewer.py            # Code review
    │
    ├── chain/                          # RAG Q&A chain
    │   ├── __init__.py
    │   ├── prompt_templates.py         # AUTOSAR RAG system prompt
    │   └── qa_chain.py                 # LCEL retrieval chain
    │
    ├── chunking/
    │   ├── __init__.py
    │   └── chunk.py                    # RecursiveCharacterTextSplitter
    │
    ├── config/
    │   ├── __init__.py
    │   ├── config.ini                  # All app configuration
    │   └── settings.py                 # Typed config reader
    │
    ├── embedding/
    │   ├── __init__.py
    │   └── embedding.py                # HuggingFace embedding singleton
    │
    ├── graph/
    │   ├── __init__.py
    │   └── workflow_graph.py           # LangGraph StateGraph orchestrator
    │
    ├── ingestion/
    │   ├── __init__.py
    │   └── pdf_loader.py               # PDF text + table + image extractor
    │
    ├── llm/
    │   ├── __init__.py
    │   └── llm_provider.py             # ChatGroq factory + singleton (BUG FIXED)
    │
    ├── models/
    │   ├── __init__.py
    │   └── state.py                    # DevTeamState TypedDict (BUG FIXED)
    │
    ├── retrieval/
    │   ├── __init__.py
    │   └── retriever.py                # VectorStore retriever config
    │
    ├── ui/
    │   ├── __init__.py
    │   └── streamlit_app.py            # Full UI controller (BUG FIXED)
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── logger.py                   # Centralized logging
    │   ├── exception.py                # CustomException with traceback
    │   └── artifact_saver.py           # NEW: organized artifact storage
    │
    └── vectorstore/
        ├── __init__.py
        └── chroma_store.py             # ChromaDB in-memory vector store
    └── main.py                         # Top-level orchestrator (BUG FIXED)
```

---

## Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.10+ | Core language |
| **Streamlit** | ≥1.35 | Web UI — file upload, chat, tabs |
| **LangChain** | ≥0.3 | LCEL pipeline, prompt templates, LCEL |
| **LangGraph** | ≥0.2 | Multi-agent DAG orchestration |
| **LangChain-Groq** | ≥0.2 | Groq LLM integration (fast inference) |
| **ChromaDB** | ≥0.5 | In-memory vector store for RAG |
| **HuggingFace** | ≥3.0 | `all-MiniLM-L6-v2` sentence embeddings |
| **PyMuPDF (fitz)** | ≥1.24 | PDF image extraction |
| **pdfplumber** | ≥0.11 | PDF table extraction |
| **PyPDF** | ≥4.0 | PDF text extraction |

---

## Bug Fixes Applied

Five critical bugs existed in the original code that would have caused runtime crashes:

### Bug 1 — Critical: Wrong dict key access (product_manager.py)

```python
# ❌ ORIGINAL — KeyError crash: using a runtime string as a dict key
product_spec = self.run(state[combined_input])

# ✅ FIXED — pass the string as input to the LLM
product_spec = self.run(combined_input)
```

### Bug 2 — LLM init crash (llm_provider.py)

```python
# ❌ ORIGINAL — 'token' is not a valid ChatGroq parameter
self._llm_instance = ChatGroq(
    model=self._model_name,
    token=self._max_tokens,   # ← Wrong parameter name
)

# ✅ FIXED
self._llm_instance = ChatGroq(
    model=self._model_name,
    max_tokens=self._max_tokens,  # ← Correct
)
```

### Bug 3 — Config key mismatch (llm_provider.py)

```python
# ❌ ORIGINAL — Key doesn't match what the UI dict stores
self._model_name = user_controls_input["LLM_Model_Name"]   # KeyError

# ✅ FIXED — Matches the key set in streamlit_app.py
self._model_name = user_controls_input.get("LLM_MODEL", "")
```

### Bug 4 — Undefined attribute (main.py)

```python
# ❌ ORIGINAL — self._user_controls is never defined → AttributeError
workflow = DevTeamWorkflow(self._user_controls)

# ✅ FIXED — correct attribute name from __init__
workflow = DevTeamWorkflow(self._user_input)
```

### Bug 5 — Wrong logging import pattern (all agents)

```python
# ❌ ORIGINAL — 'logging' was imported but logger.py doesn't export it
from src.utils.logger import logging  # ImportError / wrong object

# ✅ FIXED — module exports a proper logger instance
from src.utils.logger import logger
logger.info("Message")
```

---

## Step-by-Step Explanation

### For Beginners: What Happens When You Click Run

**Step 1: You upload an AUTOSAR SWS PDF**

The PDF contains the official AUTOSAR specification (e.g., the COM module).
It has text, tables (requirement IDs), and diagrams.

**Step 2: RAG Ingestion (indexing the document)**

```
PDF
 │
 ├─ PyPDFLoader    → extracts text per page → List[Document]
 ├─ pdfplumber     → extracts tables        → List[Document]
 └─ PyMuPDF        → extracts images        → List[Document]
                            │
                            ▼
               RecursiveCharacterTextSplitter
               (splits into 1200-char chunks with 200-char overlap)
                            │
                            ▼
               HuggingFace all-MiniLM-L6-v2
               (converts each chunk into a 384-number vector)
                            │
                            ▼
               ChromaDB (stores all vectors in memory)
```

Now ChromaDB holds a searchable index of your AUTOSAR specification.

**Step 3: You type a request like "Implement COM module signal transmission"**

```
Your request → also converted to a 384-number vector
             → ChromaDB finds the 5 most similar chunks
             → Those 5 chunks = "AUTOSAR context"
```

**Step 4: The 5-Agent LangGraph Pipeline runs**

Each agent is a Python class with a `run()` method that calls the Groq LLM:

```
ProductManagerAgent
  Input:  user_request + autosar_context
  Output: Formal requirements document with SWS clause references

ArchitectAgent
  Input:  requirements document
  Output: System architecture with AUTOSAR layer diagram and API signatures

DeveloperAgent
  Input:  architecture document
  Output: MISRA-C compliant C header (.h) and source (.c) files

QAAgent
  Input:  source code
  Output: pytest test suite with unit, integration, and compliance tests

CodeReviewAgent
  Input:  source code
  Output: MISRA-C:2012 review with severity-tagged findings table
```

**Step 5: Artifacts are saved**

```
artifacts/
└── 2026-04-12_15-30-00/
    ├── 01_requirements/COM_Requirements.md
    ├── 02_architecture/COM_Architecture.md
    ├── 03_source_code/COM_Source_Code.md
    ├── 04_test_cases/COM_Test_Cases.md
    └── 05_reviews/COM_Review_Report.md
```

Each file has a metadata header and appears as a download button in the sidebar.

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- A free Groq API key: https://console.groq.com

### 1. Clone / Extract the project

```bash
cd autosar_mas
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First-time install downloads the HuggingFace embedding model (~80MB). Subsequent runs use the local cache.

### 4. Configure your API key

Edit `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

### 5. Run the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How to Use

### Step 1: Upload AUTOSAR SWS PDF (Sidebar)

Click **"📂 Upload AUTOSAR SWS PDF"** and select any AUTOSAR Classic Platform SWS document.

> **Example PDFs**: AUTOSAR_SWS_COM.pdf, AUTOSAR_SWS_NvM.pdf, AUTOSAR_SWS_EcuM.pdf
> Available from: https://www.autosar.org/standards/classic-platform

### Step 2: Configure the LLM (Sidebar)

- **LLM Model**: `llama-3.3-70b-versatile` (recommended — best quality)
- **Temperature**: `0.2` (low = more deterministic, precise)
- **Max Tokens**: `2048` (increase for more detailed output)

### Step 3: Type Your Request

In the chat input at the bottom:

```
Implement the COM module signal transmission API including Com_SendSignal
and Com_ReceiveSignal with full DET error handling
```

### Step 4: Wait for the Pipeline (~1–3 minutes)

Watch the progress bar as each agent completes.

### Step 5: Review Results & Download

- Switch between tabs: Requirements | Architecture | Source Code | Test Cases | Code Review
- Download artifacts using the ⬇ buttons in the sidebar

---

## Agent Descriptions

### BaseAgent (Abstract Base Class)

All agents inherit from `BaseAgent` which provides the shared LCEL pipeline:

```python
class BaseAgent(ABC):
    def run(self, input_text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            ("human", "{input_text}"),
        ])
        chain = prompt | self._llm | StrOutputParser()
        return chain.invoke({"input_text": input_text})

    @abstractmethod
    def execute(self, state: DevTeamState) -> dict:
        ...
```

### ProductManagerAgent

- **Role**: Senior AUTOSAR Product Manager
- **Input**: user_request + autosar_context (RAG-retrieved SWS chunks)
- **Output**: Requirements spec with numbered SWS clause references
- **Key fix**: Combined both inputs into a single `self.run()` call

### ArchitectAgent

- **Role**: Senior AUTOSAR Software Architect (15 years BSW experience)
- **Input**: product_spec
- **Output**: Architecture document with ASCII layer diagrams and API signatures
- **Special**: Generates AUTOSAR layer diagrams (App/RTE/BSW/MCAL)

### DeveloperAgent

- **Role**: Senior AUTOSAR C Developer
- **Input**: architecture
- **Output**: Complete .h header + .c source in MISRA-C:2012 style
- **Standards**: Uses uint8/uint16/uint32/Std_ReturnType, Doxygen comments, DET checks

### QAAgent

- **Role**: Senior AUTOSAR QA Engineer (ISO 26262 ASIL-B/D)
- **Input**: code
- **Output**: pytest test suite with unit + integration + compliance tests
- **Coverage**: Happy path, error path, boundary conditions, DET behavior

### CodeReviewAgent

- **Role**: Principal AUTOSAR Code Reviewer
- **Input**: code
- **Output**: Structured review with MISRA-C table, severity ratings, corrected snippets
- **Sections**: AUTOSAR compliance, MISRA-C, code quality, security, performance

---

## RAG Pipeline Explained

RAG = Retrieval-Augmented Generation. It grounds the LLM's output in your actual
AUTOSAR SWS document rather than relying on general training knowledge.

```
                  ┌─────────────────────────────────┐
                  │       INDEXING (once per upload) │
                  │                                  │
  PDF Upload ─→  PDFLoader ─→ Chunks ─→ Vectors ─→ ChromaDB
                  └─────────────────────────────────┘

                  ┌─────────────────────────────────┐
                  │       RETRIEVAL (per request)    │
                  │                                  │
  User Request ─→ Embed ─→ Similarity Search ─→ Top-5 Chunks
                  └──────────────────┬──────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────┐
                  │  AUGMENTED GENERATION            │
                  │                                  │
                  │  System Prompt + Context + Query │
                  │        │                         │
                  │        ▼                         │
                  │     ChatGroq LLM                 │
                  │        │                         │
                  │        ▼                         │
                  │   Grounded Answer                │
                  └─────────────────────────────────┘
```

### Why chunk_size=1200?

AUTOSAR SWS requirements are structured as:
```
[SWS_Com_00001] The COM module shall initialize all internal variables...
[SWS_Com_00002] The function Com_Init shall be called once during...
```

At 1200 chars, each chunk typically contains 2-4 complete requirement clauses,
preserving enough context for the embedding model to understand the semantic unit.
The 200-char overlap prevents losing a clause that spans a chunk boundary.

---

## Artifact Storage

Every workflow run creates a timestamped session folder:

```
artifacts/
└── YYYY-MM-DD_HH-MM-SS/
    ├── 01_requirements/
    │   └── <MODULE>_Requirements.md
    ├── 02_architecture/
    │   └── <MODULE>_Architecture.md
    ├── 03_source_code/
    │   └── <MODULE>_Source_Code.md
    ├── 04_test_cases/
    │   └── <MODULE>_Test_Cases.md
    └── 05_reviews/
        └── <MODULE>_Review_Report.md
```

Each file has a metadata header:
```markdown
# AUTOSAR MAS — Requirements
**Module**: COM
**Generated**: 2026-04-12 15:30:00
**Session**: 2026-04-12_15-30-00

---
[agent output below]
```

---

## Configuration Reference

`src/config/config.ini`:

```ini
[DEFAULT]
PAGE_TITLE = AUTOSAR SWS Multi-Agent Software Development System

# Available Groq models
GROQ_MODEL_OPTIONS = llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it

# Temperature: min, default, max
TEMPERATURE = 0.0, 0.2, 1.0

# Max tokens: min, default, max
TOKEN = 512, 2048, 8192

# Chunk size (chars) — 1200 captures ~2-4 AUTOSAR SWS clauses
CHUNK_SIZE = 1200

# Overlap (chars) — prevents context loss at boundaries
CHUNK_OVERLAP = 200

# HuggingFace embedding model
EMBEDDING_MODEL = all-MiniLM-L6-v2

# Top-K retrieved chunks per query
TOP_K = 5
```

---

## OOP Design Patterns Used

### 1. Template Method Pattern (BaseAgent)
`BaseAgent` defines the `run()` pipeline skeleton. Subclasses override only
`execute()` and the system prompt — the LCEL chain is shared.

### 2. Singleton Pattern (LLMProvider, EmbeddingManager, Retriever, QAChain)
All expensive resources (LLM, embedding model, retriever) are created once
and cached as private attributes. Subsequent calls return the cached instance.

### 3. Factory Pattern (LLMProvider)
`LLMProvider.get_llm()` encapsulates the `ChatGroq` construction logic and
validation. Callers never instantiate `ChatGroq` directly.

### 4. Strategy Pattern (ChunkingStrategy)
The chunking algorithm is encapsulated behind a strategy class, making it
easy to swap `RecursiveCharacterTextSplitter` for a different splitter.

### 5. Facade Pattern (DevTeamWorkflow)
`DevTeamWorkflow.execute()` provides a single clean interface that hides the
complexity of LangGraph graph construction, node registration, and compilation.

---

## Common Interview Questions

**Q: Why LangGraph instead of a simple sequential for-loop?**
A: LangGraph's `StateGraph` gives us: (1) automatic partial state merging after
each node, (2) easy conditional branching (e.g., re-run developer on bad review),
(3) built-in visualization, (4) future-proof for parallel node execution.

**Q: Why ChromaDB in-memory instead of persistent storage?**
A: AUTOSAR SWS PDFs are large and session-specific. In-memory indexing is fast,
requires no disk management, and each PDF upload creates a fresh, clean index.
For production, switching to persistent Chroma with a collection_name is trivial.

**Q: How does the RAG grounding work end-to-end?**
A: PDF → chunks → embeddings → ChromaDB. On each request, the query is embedded,
top-K similar chunks are retrieved, injected into the ProductManagerAgent prompt
as "AUTOSAR SWS Context", and all downstream agents inherit that grounding.

**Q: How would you add a 6th agent (e.g., DocumentationAgent)?**
A: 1) Create `src/agents/documentation.py` inheriting `BaseAgent`. 2) Add it to
`DevTeamWorkflow.__init__`. 3) Add `graph.add_node()` and `graph.add_edge()` calls.
4) Add `"documentation"` key to `DevTeamState`. That's it.

**Q: What's the OOP relationship between agents?**
A: `BaseAgent` (ABC) ← `ProductManagerAgent`, `ArchitectAgent`, `DeveloperAgent`,
`QAAgent`, `CodeReviewAgent`. Each specializes only the system prompt and which
state keys they read/write. The LCEL pipeline is inherited from `BaseAgent`.
