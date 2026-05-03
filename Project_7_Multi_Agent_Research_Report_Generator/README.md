# Multi-Agent Research & Report Generator

A production-grade LangGraph pipeline where four specialised AI agents
collaborate to research any topic and produce a reviewed, cited report.

```
User Topic
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Orchestrator  — builds research plan + queries     │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│  Search Agent  — runs web queries via Tavily        │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│  Extraction Agent  — deduplicates & structures data │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│  Writer Agent  — drafts structured markdown report  │◄─────┐
└──────────────────────────┬──────────────────────────┘      │ revise
                           │                                 │
┌──────────────────────────▼──────────────────────────┐      │
│  Reviewer Agent  — audits quality & factual accuracy│──────┘
└──────────────────────────┬──────────────────────────┘
                           │ approved
                           ▼
                     Final Report
```

---

## Project Structure

```
multi-agent-research-report-generator/
│
├── app.py                         # Entry point (Streamlit launcher)
├── requirements.txt               # All dependencies
├── README.md                      # Project documentation
│
├── src/                           # Main source code
│   │
│   ├── main.py                    # Orchestrator class (app controller)
│   │
│   ├── config/                    # Configuration layer
│   │   └── settings.py            # App configs (models, tokens, etc.)
│   │
│   ├── utils/                     # Utility modules
│   │   ├── logger.py              # Logging configuration
│   │   └── exception.py           # Custom exception handling
│   │
│   ├── models/                    # Data models / state definitions
│   │   └── state.py               # ResearchState, TypedDict schemas
│   │
│   ├── llm/                       # LLM provider layer
│   │   └── llm_provider.py        # ChatGroq wrapper (singleton)
│   │
│   ├── tools/                     # External tool integrations
│   │   └── search_tool.py         # Tavily search wrapper
│   │
│   ├── agents/                    # Multi-agent system
│   │   ├── base_agent.py          # Base agent (LLM + prompt pipeline)
│   │   ├── orchestrator_agent.py  # Planner agent
│   │   ├── search_agent.py        # Search agent
│   │   ├── extraction_agent.py    # Data extraction agent
│   │   ├── writer_agent.py        # Report writer agent
│   │   └── reviewer_agent.py      # Review & feedback agent
│   │
│   ├── graph/                     # LangGraph workflow
│   │   └── workflow_graph.py      # GraphBuilder + routing logic
│   │
│   ├── report/                    # Output generation
│   │   └── pdf_generator.py       # PDF creation utility
│   │
│   └── ui/                        # Streamlit UI layer
│       └── streamlit_app.py       # UI rendering & controls







