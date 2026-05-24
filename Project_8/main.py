# ============================================================
# MULTIMODAL RAG FOR AUTOSAR PDF QA — v5 (Dynamic, Clean UI)
# ============================================================
#
# RUN:  streamlit run app.py
#
# REQUIREMENTS:
#   pip install streamlit pymupdf pdfplumber pillow
#   pip install torch torchvision torchaudio
#   pip install transformers sentence-transformers
#   pip install chromadb
#   pip install langchain langchain-core langchain-community
#   pip install langchain-text-splitters langchain-chroma
#   pip install langchain-groq protobuf==3.20.3
#
# STREAMLIT SECRET:
#   .streamlit/secrets.toml  →  GROQ_API_KEY = "gsk_..."
# ============================================================

import os
import re
import tempfile
import warnings
warnings.filterwarnings("ignore")

from typing import List, Set, Tuple

import streamlit as st
import fitz
import pdfplumber
from PIL import Image
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AUTOSAR RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================

_defaults = {
    "vectorstore":  None,
    "qa_chain":     None,
    "chat_history": [],
    "doc_stats":    {},
    "pdf_name":     "",
    "model_name":   "llama-3.3-70b-versatile",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CACHED MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.eval()
    return processor, model

# ============================================================
# CONSTANTS
# ============================================================

TOC_LINE_RE      = re.compile(r'\.{5,}')
FIGURE_RE        = re.compile(r'Figure\s+\d+', re.I)
TABLE_CAPTION_RE = re.compile(r'Table\s+\d+', re.I)
HEADING_RE       = re.compile(
    r'^(\d+(?:\.\d+){0,3})\s{1,5}([A-Z][^\n]{3,80})$', re.MULTILINE
)
BOILERPLATE_WORDS = re.compile(
    r'(AUTOSAR\s+Release|AUTOSAR\s+Confidential|Document\s+ID'
    r'|www\.autosar\.org|of\s+\d+\s+pages?)', re.I
)

# ============================================================
# PDF OUTLINE EXTRACTOR
# ============================================================

def get_pdf_outline(pdf_path: str) -> dict:
    headings = {}
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)
    if toc:
        for _level, title, page1 in toc:
            pg = page1 - 1
            if pg not in headings:
                headings[pg] = title.strip()
    else:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            m = HEADING_RE.search(text)
            if m:
                headings[i] = m.group(0).strip()
    doc.close()
    return headings

# ============================================================
# DYNAMIC PAGE CLASSIFIER
# ============================================================

def classify_pages_dynamic(pdf_path: str) -> Tuple[Set[int], Set[int]]:
    doc = fitz.open(pdf_path)
    diagram_pages: Set[int] = set()
    skip_pages: Set[int] = {0}  # cover page

    for page_num, page in enumerate(doc):
        text  = page.get_text("text").strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        if not lines:
            skip_pages.add(page_num)
            continue

        dot_ratio = sum(1 for l in lines if TOC_LINE_RE.search(l)) / len(lines)
        if dot_ratio > 0.5:
            skip_pages.add(page_num)
            continue

        # Detect embedded images
        for img in page.get_images(full=True):
            try:
                bi = doc.extract_image(img[0])
                if bi.get("width", 0) >= 80 and bi.get("height", 0) >= 80:
                    diagram_pages.add(page_num)
                    break
            except Exception:
                pass

        # Detect vector drawings
        if page_num not in diagram_pages:
            drawings = page.get_drawings()
            colored  = [d for d in drawings if d.get("color") or d.get("fill")]
            if len(colored) >= 20 and len(text) < 2000:
                diagram_pages.add(page_num)
            elif len(drawings) >= 60 and len(text) < 3000:
                diagram_pages.add(page_num)

    doc.close()
    return diagram_pages, skip_pages

# ============================================================
# HELPERS
# ============================================================

def rasterize_page(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def caption_with_blip(pil_img: Image.Image) -> str:
    processor, model = load_blip()
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120)
    return processor.decode(out[0], skip_special_tokens=True).strip()

def nearest_section(page_num: int, outline: dict) -> str:
    for pg in sorted(outline.keys(), reverse=True):
        if pg <= page_num:
            return outline[pg]
    return ""

# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_chunks(pdf_path: str, skip_pages: Set[int], outline: dict) -> List[Document]:
    doc      = fitz.open(pdf_path)
    raw_docs = []

    for page_num, page in enumerate(doc):
        if page_num in skip_pages:
            continue
        text = page.get_text("text").strip()
        if not text or len(text) < 40:
            continue
        section = outline.get(page_num) or nearest_section(page_num, outline)
        raw_docs.append(Document(
            page_content=text,
            metadata={"page": page_num + 1, "type": "text", "section": section},
        ))
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    for c in chunks:
        if not c.metadata.get("section"):
            m = HEADING_RE.search(c.page_content)
            if m:
                c.metadata["section"] = m.group(0).strip()
    return chunks

# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_chunks(pdf_path: str, skip_pages: Set[int], outline: dict) -> List[Document]:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            if page_num in skip_pages:
                continue
            tables = page.extract_tables()
            if not tables:
                continue
            section = outline.get(page_num) or nearest_section(page_num, outline)
            for t_idx, table in enumerate(tables):
                if not table:
                    continue
                rows = []
                for r_i, row in enumerate(table):
                    cells    = [str(c).strip() if c else "" for c in row]
                    row_text = " | ".join(cells)
                    if BOILERPLATE_WORDS.search(row_text):
                        continue
                    if not any(c.strip() for c in cells):
                        continue
                    rows.append(("HEADERS: " if r_i == 0 else "") + row_text)
                if rows:
                    chunks.append(Document(
                        page_content=(
                            f"TABLE (Page {page_num+1}, Table {t_idx+1})"
                            + (f" — {section}" if section else "")
                            + ":\n\n" + "\n".join(rows)
                        ),
                        metadata={"page": page_num+1, "type": "table",
                                  "section": section or f"Table {t_idx+1}"},
                    ))
    return chunks

# ============================================================
# DIAGRAM EXTRACTION
# ============================================================

def extract_diagram_chunks(
    pdf_path: str, diagram_pages: Set[int],
    skip_pages: Set[int], outline: dict
) -> Tuple[List[Document], int]:
    chunks, counter = [], 0
    for page_num in sorted(diagram_pages - skip_pages):
        try:
            img       = rasterize_page(pdf_path, page_num)
            doc_tmp   = fitz.open(pdf_path)
            page_text = doc_tmp[page_num].get_text("text").strip()
            doc_tmp.close()

            section    = outline.get(page_num) or nearest_section(page_num, outline)
            caption    = caption_with_blip(img)
            fig_labels = FIGURE_RE.findall(page_text) + TABLE_CAPTION_RE.findall(page_text)
            label_str  = ", ".join(fig_labels) if fig_labels else "Unlabelled diagram"

            chunks.append(Document(
                page_content=(
                    f"DIAGRAM — Page {page_num+1}"
                    + (f" | {section}" if section else "")
                    + f"\nLabel(s): {label_str}\n"
                    f"Visual description: {caption}\n\n"
                    f"Text/labels on page:\n{page_text[:1000]}"
                ),
                metadata={"page": page_num+1, "type": "diagram",
                          "section": section or label_str},
            ))
            counter += 1
        except Exception as e:
            st.warning(f"Page {page_num+1} diagram skipped: {e}")
    return chunks, counter

# ============================================================
# VECTOR STORE & RETRIEVER
# ============================================================

def build_vectorstore(docs: List[Document]) -> Chroma:
    return Chroma.from_documents(
        documents=docs,
        embedding=load_embeddings(),
        collection_name="autosar_rag_v5",
    )

def create_retriever(vs: Chroma):
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.55},
    )

# ============================================================
# LLM & PROMPT
# ============================================================

def init_llm(model: str, api_key: str, temperature: float, max_tokens: int):
    return ChatGroq(model=model, api_key=api_key,
                    temperature=temperature, max_tokens=max_tokens)

SYSTEM_PROMPT = """You are an expert AUTOSAR technical documentation assistant.
Answer questions using ONLY the supplied context.

RESPONSE FORMAT:
1. Start with a 1–3 sentence summary.
2. Use ## for major sections, ### for sub-sections.
3. API functions: code-fence the signature, then a parameter table.
4. Config parameters: table with Parameter | Multiplicity | Type | Range | Description.
5. Enumerations: code-fence block; mark inferred values with *(inferred)*.
6. Architecture/Diagrams: describe every layer, module, arrow, and interface.
7. Bold all SWS requirement IDs (e.g. `SWS_Com_00061`).
8. End every answer with:
   > 📄 Sources: Page X (text), Page Y (table), Page Z (diagram)

RULES:
- Use ONLY the context below. No external knowledge.
- Preserve exact AUTOSAR identifiers and acronyms.
- If context partially covers the question, state what is missing.
- If no relevant info exists, respond only with:
  "The provided context does not contain information about this topic."

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}
"""

def create_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        pg  = d.metadata.get("page", "?")
        dt  = d.metadata.get("type", "text")
        sec = d.metadata.get("section", "")
        hdr = f"[DOC {i}] Page {pg} | {dt.upper()}" + (f" | {sec}" if sec else "")
        parts.append(f"{hdr}\n\n{d.page_content}")
    return ("\n\n" + "─" * 60 + "\n\n").join(parts)

# ============================================================
# RAG CHAIN
# ============================================================

def create_rag_chain(retriever, llm):
    prompt = create_prompt()

    def retrieve(inp: dict) -> dict:
        docs     = retriever.invoke(inp["question"])
        history  = st.session_state.chat_history[-3:]
        hist_str = "\n".join(
            f"User: {t['question']}\nAssistant: {t['answer'][:300]}…"
            for t in history
        ) or "None"
        return {
            "question":         inp["question"],
            "context":          format_docs(docs),
            "history":          hist_str,
            "source_documents": docs,
        }

    def generate(inp: dict) -> dict:
        chain  = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context":  inp["context"],
            "history":  inp["history"],
            "question": inp["question"],
        })
        return {"answer": answer, "source_documents": inp["source_documents"]}

    return RunnableLambda(retrieve) | RunnableLambda(generate)

def build_qa_chain(vs, model_name, api_key, temperature, max_tokens):
    return create_rag_chain(
        create_retriever(vs),
        init_llm(model_name, api_key, temperature, max_tokens),
    )

# ============================================================
# FULL PROCESSING PIPELINE
# ============================================================

def process_pdf(pdf_path, model_name, api_key, include_diagrams,
                temperature, max_tokens, filename=""):
    progress = st.progress(0)
    all_docs: List[Document] = []

    progress.progress(5,  text="Reading PDF structure…")
    outline = get_pdf_outline(pdf_path)

    progress.progress(12, text="Classifying pages…")
    diagram_pages, skip_pages = classify_pages_dynamic(pdf_path)

    progress.progress(22, text="Extracting text chunks…")
    text_chunks = extract_text_chunks(pdf_path, skip_pages, outline)
    all_docs.extend(text_chunks)

    progress.progress(45, text="Extracting tables…")
    table_chunks = extract_table_chunks(pdf_path, skip_pages, outline)
    all_docs.extend(table_chunks)

    diagram_count = 0
    if include_diagrams and diagram_pages:
        n = len(diagram_pages - skip_pages)
        progress.progress(58, text=f"Captioning {n} diagram page(s)…")
        diag_chunks, diagram_count = extract_diagram_chunks(
            pdf_path, diagram_pages, skip_pages, outline
        )
        all_docs.extend(diag_chunks)

    progress.progress(80, text="Building vector store…")
    vs = build_vectorstore(all_docs)
    st.session_state.vectorstore = vs

    progress.progress(93, text="Initialising QA chain…")
    st.session_state.qa_chain   = build_qa_chain(vs, model_name, api_key, temperature, max_tokens)
    st.session_state.model_name = model_name
    st.session_state.pdf_name   = filename

    progress.progress(100, text="Ready!")
    st.session_state.doc_stats = {
        "text_chunks":    len(text_chunks),
        "table_chunks":   len(table_chunks),
        "diagram_chunks": diagram_count,
        "total_chunks":   len(all_docs),
        "outline_pages":  len(outline),
        "diagram_pages":  len(diagram_pages),
        "skip_pages":     len(skip_pages),
    }

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    model = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
         "gemma2-9b-it", "mixtral-8x7b-32768"],
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.05, 0.05)
    max_tokens  = st.slider("Max Tokens",  200, 4096, 1800, 100)

    # Hot-swap model without re-processing
    if (st.session_state.qa_chain is not None
            and model != st.session_state.model_name):
        if st.button("Apply Model Change", use_container_width=True):
            vs = st.session_state.vectorstore
            st.session_state.qa_chain   = build_qa_chain(vs, model, api_key, temperature, max_tokens)
            st.session_state.model_name = model
            st.success(f"Switched to {model}")

    st.divider()

    uploaded_file    = st.file_uploader("Upload AUTOSAR PDF", type=["pdf"])
    include_diagrams = st.checkbox("Enable Diagram Captioning (slower)", value=True)

    if st.button("Process PDF", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a PDF first.")
        elif not api_key:
            st.error("API key is required.")
        else:
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
                    filename=uploaded_file.name,
                )
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.doc_stats and st.session_state.qa_chain:
        s = st.session_state.doc_stats
        st.divider()
        st.caption(f"📂 {st.session_state.pdf_name or 'Document loaded'}")
        col1, col2 = st.columns(2)
        col1.metric("Text chunks",  s["text_chunks"])
        col2.metric("Tables",       s["table_chunks"])
        col1.metric("Diagrams",     s["diagram_chunks"])
        col2.metric("Total indexed", s["total_chunks"])
        st.caption(
            f"📑 {s['outline_pages']} sections · "
            f"🖼️ {s['diagram_pages']} diagram pages · "
            f"⏭️ {s['skip_pages']} skipped"
        )

# ============================================================
# MAIN AREA
# ============================================================

st.title("🧠 AUTOSAR Multimodal RAG")
st.caption("Groq · LangChain · ChromaDB · BLIP — works with any AUTOSAR PDF")
st.divider()

# ── Welcome screen ──
if st.session_state.qa_chain is None:
    st.info("👈 Upload an AUTOSAR PDF in the sidebar and click **Process PDF** to begin.")
    st.markdown("""
**What this app does:**
- 📝 Extracts and chunks all text, preserving section headings
- 📊 Extracts all tables with header labelling
- 🖼️ Detects and captions diagram pages using the BLIP vision model
- 🔍 Indexes everything in a vector store with MMR retrieval
- 🧠 Answers your questions with a structured, cited Markdown response
- 🔄 Supports model hot-swap and 3-turn conversation memory
    """)

# ── Chat history ──
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        sources = turn.get("sources", [])
        if sources:
            seen, parts = set(), []
            for d in sources:
                key = (d.metadata.get("page","?"), d.metadata.get("type","text"))
                if key not in seen:
                    seen.add(key)
                    parts.append(f"p.{key[0]} ({key[1]})")
            st.caption("Sources: " + " · ".join(parts))
        with st.expander("View retrieved chunks", expanded=False):
            for i, d in enumerate(sources, 1):
                pg  = d.metadata.get("page", "?")
                dt  = d.metadata.get("type", "?")
                sec = d.metadata.get("section", "")
                st.markdown(f"**[{i}] Page {pg} · {dt.upper()}**" + (f" · *{sec}*" if sec else ""))
                st.code(d.page_content[:600], language="")

# ── Live chat input ──
if st.session_state.qa_chain:
    question = st.chat_input("Ask anything about the AUTOSAR document…")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer…"):
                try:
                    result  = st.session_state.qa_chain.invoke({"question": question})
                    answer  = result["answer"]
                    sources = result["source_documents"]

                    st.markdown(answer)

                    if sources:
                        seen, parts = set(), []
                        for d in sources:
                            key = (d.metadata.get("page","?"), d.metadata.get("type","text"))
                            if key not in seen:
                                seen.add(key)
                                parts.append(f"p.{key[0]} ({key[1]})")
                        st.caption("Sources: " + " · ".join(parts))

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer":   answer,
                        "sources":  sources,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Clear chat ──
if st.session_state.chat_history:
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
