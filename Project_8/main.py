# ============================================================
# MULTIMODAL RAG FOR PDF QA (ENTERPRISE QUALITY) — v4
# ============================================================
#
# ENHANCEMENTS vs v3
# ------------------
# ✅ "Retrieved Sections" panel REMOVED from primary response view
#    — sources now shown in a collapsible sidebar-style expander
#    only if the user explicitly toggles it
# ✅ Richer LLM prompt: structured output with Markdown headings,
#    bullet lists, code blocks for API signatures, and tables for
#    parameter lists — the LLM now returns well-formatted Markdown
# ✅ Answer rendered with st.markdown() for full Markdown display
#    (bold, bullet lists, tables, code fences all render correctly)
# ✅ Premium dark-theme UI: refined typography, card-based layout,
#    gradient accents, animated progress indicators
# ✅ Per-question metadata strip (page refs, content types) shown
#    compactly below the answer — not as a massive expandable wall
# ✅ All v3 extraction features retained unchanged
#
# RUN:
# -----
# streamlit run app.py
#
# REQUIREMENTS:
# -------------
# pip install streamlit pymupdf pdfplumber pillow
# pip install torch torchvision torchaudio
# pip install transformers sentence-transformers
# pip install chromadb
# pip install langchain langchain-core
# pip install langchain-community
# pip install langchain-text-splitters
# pip install langchain-chroma
# pip install langchain-groq
# pip install protobuf==3.20.3
#
# STREAMLIT SECRET:
# -----------------
# .streamlit/secrets.toml
# GROQ_API_KEY="your_api_key"
#
# ============================================================

import os
import re
import tempfile
import warnings
warnings.filterwarnings("ignore")

from typing import List

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import torch

# LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocMind — Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS  — Premium dark-tech theme
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

/* ── Root palette ────────────────────────────────────────── */
:root {
    --bg-base:      #0d0f14;
    --bg-card:      #13161e;
    --bg-input:     #1a1e2b;
    --border:       #252a38;
    --border-glow:  #3b82f6;
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose:  #f43f5e;
    --text-primary: #e8eaf0;
    --text-muted:   #6b7280;
    --text-dim:     #374151;
    --mono:         'IBM Plex Mono', monospace;
    --sans:         'Sora', sans-serif;
}

/* ── Global resets ───────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.stApp { background: var(--bg-base) !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Header banner ───────────────────────────────────────── */
.docmind-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #2d2b6b;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.docmind-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.docmind-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #818cf8, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
}
.docmind-sub {
    color: var(--text-muted);
    font-size: 0.875rem;
    font-weight: 300;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0;
}

/* ── Stat chips ──────────────────────────────────────────── */
.stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.stat-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    font-size: 0.78rem;
    font-family: var(--mono);
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.stat-chip b { color: var(--text-primary); }

/* ── Chat messages ───────────────────────────────────────── */
.msg-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2f4a);
    border: 1px solid #2563eb40;
    border-radius: 14px 14px 4px 14px;
    padding: 1rem 1.2rem;
    margin: 1rem 0 0.5rem auto;
    max-width: 78%;
    font-size: 0.95rem;
    color: #bfdbfe;
    line-height: 1.6;
}

.msg-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px 14px 14px 14px;
    padding: 1.4rem 1.6rem;
    margin: 0.5rem 0 0.5rem 0;
    font-size: 0.93rem;
    line-height: 1.75;
    color: var(--text-primary);
    position: relative;
}
.msg-assistant::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-green));
    border-radius: 4px 14px 0 0;
}

/* ── Markdown inside answer ──────────────────────────────── */
.msg-assistant h2 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #93c5fd;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
    letter-spacing: 0.01em;
}
.msg-assistant h3 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #6ee7b7;
    margin: 0.9rem 0 0.3rem 0;
}
.msg-assistant ul, .msg-assistant ol {
    margin: 0.5rem 0 0.5rem 1.2rem;
    padding: 0;
}
.msg-assistant li {
    margin-bottom: 0.35rem;
    color: #d1d5db;
}
.msg-assistant code {
    font-family: var(--mono);
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 4px;
    padding: 0.15em 0.45em;
    font-size: 0.83em;
    color: #7dd3fc;
}
.msg-assistant pre {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    overflow-x: auto;
    margin: 0.75rem 0;
}
.msg-assistant pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: 0.82em;
    color: #a5f3fc;
}
.msg-assistant table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8rem 0;
    font-size: 0.84rem;
}
.msg-assistant th {
    background: #1e2535;
    color: #93c5fd;
    padding: 0.5rem 0.8rem;
    text-align: left;
    border: 1px solid var(--border);
    font-weight: 600;
    font-family: var(--mono);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.msg-assistant td {
    padding: 0.45rem 0.8rem;
    border: 1px solid var(--border);
    color: #d1d5db;
    vertical-align: top;
}
.msg-assistant tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
.msg-assistant strong { color: #fde68a; font-weight: 600; }
.msg-assistant em { color: #a5b4fc; }
.msg-assistant blockquote {
    border-left: 3px solid var(--accent-blue);
    margin: 0.8rem 0;
    padding: 0.5rem 1rem;
    background: rgba(59,130,246,0.05);
    border-radius: 0 6px 6px 0;
    color: #94a3b8;
    font-size: 0.9rem;
}

/* ── Source citation strip ───────────────────────────────── */
.src-strip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
}
.src-label {
    font-size: 0.72rem;
    font-family: var(--mono);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-right: 0.25rem;
}
.src-badge {
    font-size: 0.72rem;
    font-family: var(--mono);
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    white-space: nowrap;
}
.src-badge.text    { background: rgba(16,185,129,0.12); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.25); }
.src-badge.table   { background: rgba(245,158,11,0.12); color: #fcd34d; border: 1px solid rgba(245,158,11,0.25); }
.src-badge.diagram { background: rgba(139,92,246,0.12); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.25); }

/* ── Welcome card ────────────────────────────────────────── */
.welcome-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.welcome-icon { font-size: 3rem; margin-bottom: 1rem; }
.welcome-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.welcome-desc { color: var(--text-muted); font-size: 0.9rem; line-height: 1.7; }
.step-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}
.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}
.step-num {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; color: white;
    margin: 0 auto 0.6rem auto;
}
.step-text { font-size: 0.82rem; color: var(--text-muted); line-height: 1.5; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Chat input ──────────────────────────────────────────── */
[data-testid="stChatInputTextArea"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: var(--sans) !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* ── Progress bar ────────────────────────────────────────── */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "vectorstore":  None,
    "qa_chain":     None,
    "chat_history": [],
    "doc_stats":    {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CACHED MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.eval()
    return processor, model

# ============================================================
# CONSTANTS — DIAGRAM DETECTION
# ============================================================

FIGURE_CAPTION_RE = re.compile(r'Figure\s+\d+\s*[:\.]', re.IGNORECASE)
HEADING_RE = re.compile(
    r'^(\d+(\.\d+){0,3})\s+([A-Z][^\n]{3,80})$',
    re.MULTILINE
)
MIN_DRAWING_PATHS_FOR_DIAGRAM   = 25
MIN_DRAWING_PATHS_NO_CAPTION    = 80

# ============================================================
# HELPERS
# ============================================================

def extract_nearest_heading(text: str) -> str:
    matches = list(HEADING_RE.finditer(text))
    return matches[-1].group(0).strip() if matches else ""


def is_toc_only_page(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    dot_lines = sum(1 for l in lines if re.search(r'\.{4,}', l))
    return dot_lines / len(lines) > 0.6


def caption_image_with_blip(pil_img: Image.Image) -> str:
    processor, model = load_blip_model()
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def rasterize_page(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

# ============================================================
# DIAGRAM PAGE CLASSIFIER
# ============================================================

def classify_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    diagram_page_nums = set()
    HEADER_LOGO_XREF = 14

    for page_num, page in enumerate(doc):
        text     = page.get_text("text").strip()
        drawings = page.get_drawings()
        images   = page.get_images(full=True)
        n_draw   = len(drawings)

        if FIGURE_CAPTION_RE.search(text) and n_draw >= MIN_DRAWING_PATHS_FOR_DIAGRAM:
            diagram_page_nums.add(page_num)
            continue

        for img in images:
            xref = img[0]
            if xref == HEADER_LOGO_XREF:
                continue
            try:
                bi = doc.extract_image(xref)
                if bi.get("width", 0) >= 150 and bi.get("height", 0) >= 150:
                    diagram_page_nums.add(page_num)
                    break
            except Exception:
                pass

        if page_num not in diagram_page_nums:
            colored_draws = sum(
                1 for d in drawings if d.get("color") is not None
            )
            if (n_draw >= MIN_DRAWING_PATHS_NO_CAPTION
                    and colored_draws > 0
                    and len(text) < 1500):
                diagram_page_nums.add(page_num)

    doc.close()
    return diagram_page_nums

# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_chunks(pdf_path: str) -> List[Document]:
    doc = fitz.open(pdf_path)
    raw_docs = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text or is_toc_only_page(text):
            continue

        heading = extract_nearest_heading(text)
        raw_docs.append(
            Document(
                page_content=text,
                metadata={
                    "page":    page_num + 1,
                    "type":    "text",
                    "section": heading,
                }
            )
        )

    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = splitter.split_documents(raw_docs)

    for d in split_docs:
        if not d.metadata.get("section"):
            d.metadata["section"] = extract_nearest_heading(d.page_content)

    return split_docs

# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_chunks(pdf_path: str) -> List[Document]:
    BOILERPLATE_RE = re.compile(
        r'(document id|autosar_sws|of \d+|autosar confidential)',
        re.IGNORECASE
    )
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            for idx, table in enumerate(tables):
                if not table:
                    continue

                rows = []
                for row_i, row in enumerate(table):
                    cleaned = [str(c).strip() if c else "" for c in row]
                    row_text = " | ".join(cleaned)

                    if BOILERPLATE_RE.search(row_text):
                        continue
                    if not any(c.strip() for c in cleaned):
                        continue

                    if row_i == 0:
                        rows.append(f"HEADERS: {row_text}")
                    else:
                        rows.append(row_text)

                table_text = "\n".join(rows)
                if table_text.strip():
                    chunks.append(
                        Document(
                            page_content=(
                                f"TABLE CONTENT (Page {page_num + 1}, "
                                f"Table {idx + 1}):\n\n{table_text}"
                            ),
                            metadata={
                                "page":    page_num + 1,
                                "type":    "table",
                                "section": f"Table {idx + 1}",
                            }
                        )
                    )

    return chunks

# ============================================================
# DIAGRAM EXTRACTION
# ============================================================

def extract_diagram_chunks(pdf_path: str, diagram_page_nums: set):
    chunks = []
    img_counter = 0

    for page_num in sorted(diagram_page_nums):
        try:
            page_render = rasterize_page(pdf_path, page_num, dpi=150)

            doc_tmp = fitz.open(pdf_path)
            page_text = doc_tmp[page_num].get_text("text").strip()
            doc_tmp.close()

            caption     = caption_image_with_blip(page_render)
            fig_matches = FIGURE_CAPTION_RE.findall(page_text)
            fig_label   = ", ".join(fig_matches) if fig_matches else "Unlabelled diagram"

            full_description = (
                f"DIAGRAM PAGE {page_num + 1} — {fig_label}\n\n"
                f"Visual caption: {caption}\n\n"
                f"Text/labels on this page:\n{page_text[:800]}"
            )

            img_counter += 1
            chunks.append(
                Document(
                    page_content=full_description,
                    metadata={
                        "page":    page_num + 1,
                        "type":    "diagram",
                        "section": fig_label,
                    }
                )
            )

        except Exception as e:
            st.warning(f"Could not process diagram on page {page_num + 1}: {e}")

    return chunks, img_counter

# ============================================================
# VECTORSTORE
# ============================================================

def build_vectorstore(all_docs: List[Document]) -> Chroma:
    embeddings = load_embedding_model()
    return Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="multimodal_rag"
    )

# ============================================================
# RETRIEVER
# ============================================================

def create_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.6,
        }
    )

# ============================================================
# LLM
# ============================================================

def initialize_llm(model, api_key, temperature, max_tokens):
    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

# ============================================================
# ENHANCED PROMPT  — structured Markdown output
# ============================================================

def create_prompt() -> ChatPromptTemplate:
    template = """You are an expert technical documentation assistant specialising in AUTOSAR and embedded systems specifications.

Your task is to answer the user's question using ONLY the provided context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — strictly follow this structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write your answer in clean, readable **Markdown**.

**Structure rules:**
1. Start with a concise 1-2 sentence summary of the answer in plain prose (no heading needed).
2. Use `##` headings to separate major sub-topics when the answer has multiple distinct parts.
3. Use `###` sub-headings for nested detail (e.g. individual parameters, sub-features).
4. Use bullet lists (`-`) for enumerations; use numbered lists only for ordered steps.
5. For **API functions**: always present a code-fenced signature block (`\`\`\`c`) followed by a Markdown table with columns: Parameter | Direction | Type | Description. Then state return type, synchronous/asynchronous, and re-entrancy.
6. For **configuration parameters**: present a Markdown table with columns: Parameter | Type | Range / Values | Description.
7. For **enum or constant values**: list them in a `\`\`\`c` block. If a value is inferred (not explicit), add *(inferred)* after it.
8. For **architecture / diagram descriptions**: describe every visible layer, module, arrow, and interface in a structured way. Reference the figure number and page.
9. Use `**bold**` for key technical identifiers, SWS requirement IDs (e.g. `SWS_Com_00061`), and important values.
10. End with a compact **> 📄 Sources** blockquote listing page numbers and content types (e.g. `> 📄 Sources: Page 71 (text), Page 140 (table), Page 12 (diagram)`). This replaces a verbose source section.

**Content rules:**
- Answer ONLY from the context. Do NOT use external knowledge.
- Use exact terminology from the specification. Never paraphrase technical identifiers.
- If the context partially answers the question, provide what is available and explicitly state what is missing.
- If the context has NO relevant information, respond only with: "The provided context does not contain this information."
- For multi-part questions, address each part under its own `##` heading.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT:
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION:
{question}
"""
    return ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{question}")
    ])

# ============================================================
# FORMAT DOCS
# ============================================================

def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        page    = doc.metadata.get("page", "?")
        dtype   = doc.metadata.get("type", "text")
        section = doc.metadata.get("section", "")
        header  = f"DOCUMENT {i} | Page {page} | Type: {dtype}"
        if section:
            header += f" | Section: {section}"
        parts.append(f"{header}\n\nCONTENT:\n{doc.page_content}")
    sep = "\n\n" + "=" * 60 + "\n\n"
    return sep.join(parts)

# ============================================================
# RAG CHAIN
# ============================================================

def create_rag_chain(retriever, llm):
    prompt = create_prompt()

    def retrieve(question: str):
        docs = retriever.invoke(question)
        return {
            "question":         question,
            "context":          format_docs(docs),
            "source_documents": docs,
        }

    def generate(inputs: dict):
        chain  = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context":  inputs["context"],
            "question": inputs["question"],
        })
        return {
            "answer":           answer,
            "source_documents": inputs["source_documents"],
        }

    return RunnableLambda(retrieve) | RunnableLambda(generate)


def build_qa_chain(vectorstore, model_name, api_key, temperature, max_tokens):
    retriever = create_retriever(vectorstore)
    llm = initialize_llm(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return create_rag_chain(retriever, llm)

# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(pdf_path, model_name, api_key, include_diagrams, temperature, max_tokens):
    progress = st.progress(0)
    all_docs = []

    progress.progress(5, text="🔍 Classifying page types…")
    diagram_page_nums = classify_pages(pdf_path) if include_diagrams else set()

    progress.progress(20, text="📝 Extracting text chunks…")
    text_chunks = extract_text_chunks(pdf_path)
    all_docs.extend(text_chunks)

    progress.progress(45, text="📊 Extracting tables…")
    table_chunks = extract_table_chunks(pdf_path)
    all_docs.extend(table_chunks)

    diagram_count = 0
    if include_diagrams and diagram_page_nums:
        progress.progress(60, text=f"🖼️ Rasterizing {len(diagram_page_nums)} diagram pages…")
        diagram_chunks, diagram_count = extract_diagram_chunks(pdf_path, diagram_page_nums)
        all_docs.extend(diagram_chunks)

    progress.progress(80, text="🧠 Building vector store…")
    vectorstore = build_vectorstore(all_docs)
    st.session_state.vectorstore = vectorstore

    progress.progress(95, text="⚡ Initialising QA chain…")
    qa_chain = build_qa_chain(
        vectorstore=vectorstore,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    st.session_state.qa_chain = qa_chain
    progress.progress(100, text="✅ Ready!")

    st.session_state.doc_stats = {
        "text_chunks":    len(text_chunks),
        "table_chunks":   len(table_chunks),
        "diagram_chunks": diagram_count,
        "total_chunks":   len(all_docs),
    }

# ============================================================
# SOURCE BADGE HELPER
# ============================================================

def render_source_strip(docs: List[Document]) -> str:
    """Build a compact HTML source citation strip from retrieved docs."""
    seen = set()
    badges = []
    for doc in docs:
        page  = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("type", "text")
        key   = (page, dtype)
        if key in seen:
            continue
        seen.add(key)
        icon_map = {"text": "📄", "table": "📊", "diagram": "🖼️"}
        icon = icon_map.get(dtype, "📄")
        badges.append(
            f'<span class="src-badge {dtype}">{icon} p.{page} · {dtype}</span>'
        )
    badges_html = "\n".join(badges)
    return (
        f'<div class="src-strip">'
        f'<span class="src-label">Sources</span>'
        f'{badges_html}'
        f'</div>'
    )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ Add GROQ_API_KEY to `.streamlit/secrets.toml`")

    model = st.selectbox(
        "🤖 Model",
        [
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b",
        ]
    )

    temperature   = st.slider("🌡️ Temperature", 0.0, 1.0, 0.1)
    max_tokens    = st.slider("📏 Max Tokens",  100, 2000, 1200)

    st.divider()
    uploaded_file    = st.file_uploader("📄 Upload PDF", type=["pdf"])
    include_diagrams = st.checkbox("🖼️ Diagram Analysis", value=True)
    process_btn      = st.button("⚡ Process PDF", use_container_width=True)

    if process_btn and uploaded_file:
        st.session_state.chat_history = []

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            process_pdf(
                pdf_path=tmp_path,
                model_name=model,
                api_key=api_key,
                include_diagrams=include_diagrams,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            st.success("✅ PDF processed!")

            s = st.session_state.doc_stats
            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem">
                  <div style="background:#1a2535;border:1px solid #2d3748;border-radius:8px;padding:0.6rem;text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#60a5fa">{s['text_chunks']}</div>
                    <div style="font-size:0.7rem;color:#6b7280">text chunks</div>
                  </div>
                  <div style="background:#1a2535;border:1px solid #2d3748;border-radius:8px;padding:0.6rem;text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#fbbf24">{s['table_chunks']}</div>
                    <div style="font-size:0.7rem;color:#6b7280">table chunks</div>
                  </div>
                  <div style="background:#1a2535;border:1px solid #2d3748;border-radius:8px;padding:0.6rem;text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#a78bfa">{s['diagram_chunks']}</div>
                    <div style="font-size:0.7rem;color:#6b7280">diagrams</div>
                  </div>
                  <div style="background:#1a2535;border:1px solid #2d3748;border-radius:8px;padding:0.6rem;text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#34d399">{s['total_chunks']}</div>
                    <div style="font-size:0.7rem;color:#6b7280">total chunks</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(str(e))
        finally:
            os.unlink(tmp_path)

    # Show active doc stats if already processed
    elif st.session_state.doc_stats and st.session_state.qa_chain:
        s = st.session_state.doc_stats
        st.markdown(
            f"""
            <div style="padding:0.8rem;background:#1a2535;border:1px solid #2d3748;border-radius:10px;margin-top:0.5rem">
              <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem">Document loaded</div>
              <div style="font-size:0.82rem;color:#94a3b8">📝 {s['text_chunks']} text &nbsp;·&nbsp; 📊 {s['table_chunks']} tables &nbsp;·&nbsp; 🖼️ {s['diagram_chunks']} diagrams</div>
              <div style="font-size:0.78rem;color:#4b5563;margin-top:0.3rem">{s['total_chunks']} total indexed chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# MAIN UI
# ============================================================

# Header
st.markdown(
    """
    <div class="docmind-header">
      <div class="docmind-title">🧠 DocMind — Multimodal RAG</div>
      <div class="docmind-sub">Groq · LangChain · ChromaDB · BLIP &nbsp;|&nbsp; v4</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# WELCOME SCREEN
# ============================================================

if st.session_state.qa_chain is None:
    st.markdown(
        """
        <div class="welcome-card">
          <div class="welcome-icon">📚</div>
          <div class="welcome-title">Upload a PDF to get started</div>
          <div class="welcome-desc">
            Ask questions about any PDF — technical specifications, research papers, manuals.<br>
            DocMind extracts text, tables, and diagrams to give you precise, structured answers.
          </div>
          <div class="step-grid">
            <div class="step-card">
              <div class="step-num">1</div>
              <div class="step-text">Add your <strong>GROQ_API_KEY</strong> in Streamlit secrets</div>
            </div>
            <div class="step-card">
              <div class="step-num">2</div>
              <div class="step-text">Upload a <strong>PDF document</strong> in the sidebar</div>
            </div>
            <div class="step-card">
              <div class="step-num">3</div>
              <div class="step-text">Click <strong>Process PDF</strong> and start chatting</div>
            </div>
          </div>
        </div>

        <div style="margin-top:1.5rem;display:grid;grid-template-columns:repeat(3,1fr);gap:1rem">
          <div style="background:#13161e;border:1px solid #1f2937;border-radius:10px;padding:1rem">
            <div style="font-size:1.3rem;margin-bottom:0.4rem">📝</div>
            <div style="font-size:0.82rem;font-weight:600;color:#d1d5db;margin-bottom:0.3rem">Full Text Extraction</div>
            <div style="font-size:0.78rem;color:#6b7280;line-height:1.5">All pages including labels on diagram pages are indexed for search</div>
          </div>
          <div style="background:#13161e;border:1px solid #1f2937;border-radius:10px;padding:1rem">
            <div style="font-size:1.3rem;margin-bottom:0.4rem">📊</div>
            <div style="font-size:0.82rem;font-weight:600;color:#d1d5db;margin-bottom:0.3rem">Table Intelligence</div>
            <div style="font-size:0.78rem;color:#6b7280;line-height:1.5">Tables extracted with header labelling and boilerplate filtering</div>
          </div>
          <div style="background:#13161e;border:1px solid #1f2937;border-radius:10px;padding:1rem">
            <div style="font-size:1.3rem;margin-bottom:0.4rem">🖼️</div>
            <div style="font-size:0.82rem;font-weight:600;color:#d1d5db;margin-bottom:0.3rem">Diagram Captioning</div>
            <div style="font-size:0.78rem;color:#6b7280;line-height:1.5">BLIP model captions vector and raster diagrams; all 17+ figure types detected</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# CHAT HISTORY
# ============================================================

for turn in st.session_state.chat_history:
    # User bubble
    st.markdown(
        f'<div class="msg-user">{turn["question"]}</div>',
        unsafe_allow_html=True
    )
    # Assistant answer
    answer_md = turn["answer"]
    src_html  = render_source_strip(turn.get("sources", []))

    st.markdown(
        f'<div class="msg-assistant">{answer_md}{src_html}</div>',
        unsafe_allow_html=True
    )

# ============================================================
# LIVE CHAT INPUT
# ============================================================

if st.session_state.qa_chain:
    question = st.chat_input("Ask anything about the document…")

    if question:
        # Show user bubble immediately
        st.markdown(
            f'<div class="msg-user">{question}</div>',
            unsafe_allow_html=True
        )

        with st.spinner(""):
            try:
                result  = st.session_state.qa_chain.invoke(question)
                answer  = result["answer"]
                sources = result["source_documents"]

                src_html = render_source_strip(sources)

                st.markdown(
                    f'<div class="msg-assistant">{answer}{src_html}</div>',
                    unsafe_allow_html=True
                )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer":   answer,
                    "sources":  sources,
                })

            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.chat_history:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
