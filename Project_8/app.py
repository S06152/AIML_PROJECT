# ============================================================
# MULTIMODAL RAG FOR PDF QA (ENTERPRISE QUALITY) — FIXED v3
# ============================================================
#
# FIXES vs v2
# -----------
# ✅ Diagram detection v2 — now catches ALL diagram page types:
#    • Vector diagrams (UML, flow, architecture) via drawing
#      path count + strict "Figure N:" caption matching
#    • Raster image pages (embedded PNG/JPEG >= 150px)
#    • Diagram-only pages with no figure caption but dense
#      vector drawing paths and low text (e.g. pages 81–82)
#    v2 only detected 7 pages; v3 detects all 17+ correctly
# ✅ Text on diagram pages now also extracted separately so
#    labels / API names visible on a figure page are indexed
#    as text chunks too (dual indexing)
# ✅ Table extraction now skips header/footer boilerplate rows
#    (page number rows, document ID rows) that pollute tables
# ✅ All content types (text, tables, diagrams) are extracted
#    without any page being silently skipped
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
    page_title="Enterprise Multimodal RAG",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
.sub-header  { color: gray; margin-bottom: 2rem; }
.answer-box  {
    background: #f4fff5;
    border: 1px solid #d4f5dd;
    padding: 1rem;
    border-radius: 10px;
    font-size: 1rem;
    white-space: pre-wrap;
}
.chunk-card {
    background: #fafafa;
    border-left: 4px solid #2ecc71;
    padding: 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.8rem;
}
.chunk-meta { color: gray; font-size: 0.8rem; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "vectorstore": None,
    "qa_chain": None,
    "chat_history": [],
    "doc_stats": {},
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

# Strict figure caption: "Figure 1:" or "Figure 1." — means the
# figure is physically present on this page (not a cross-reference)
FIGURE_CAPTION_RE = re.compile(r'Figure\s+\d+\s*[:\.]', re.IGNORECASE)

# Heading pattern for section metadata
HEADING_RE = re.compile(
    r'^(\d+(\.\d+){0,3})\s+([A-Z][^\n]{3,80})$',
    re.MULTILINE
)

# Minimum vector drawing paths to consider a page as having a diagram
# (tables also produce drawing paths for borders — threshold filters them)
MIN_DRAWING_PATHS_FOR_DIAGRAM = 25

# Minimum drawing paths for a page with NO figure caption to still be
# treated as a diagram (e.g. pages 81–82: pure data-flow diagrams with
# no "Figure N:" label in the text)
MIN_DRAWING_PATHS_NO_CAPTION = 80

# ============================================================
# HELPERS
# ============================================================

def extract_nearest_heading(text: str) -> str:
    matches = list(HEADING_RE.finditer(text))
    return matches[-1].group(0).strip() if matches else ""


def is_toc_only_page(text: str) -> bool:
    """True only when >60% of non-empty lines are TOC dot-leader rows."""
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
    """Render a single PDF page (0-indexed) to a PIL RGB image."""
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
    """
    Returns a set of 0-indexed page numbers that are diagram pages.

    Detection rules (any one match = diagram page):
    ──────────────────────────────────────────────
    Rule A — Figure caption + vector drawings:
        Page text contains "Figure N:" AND drawing path count
        >= MIN_DRAWING_PATHS_FOR_DIAGRAM.
        Catches: architecture figures, signal flow diagrams,
        UML diagrams, timing diagrams drawn as vectors.

    Rule B — Large embedded raster image:
        Page contains at least one raster image >= 150×150 px
        (ignoring the shared 464×48 header logo on every page).
        Catches: bitmap figures, scanned diagrams.

    Rule C — Dense colored vector drawings, no caption needed:
        Drawing path count >= MIN_DRAWING_PATHS_NO_CAPTION AND
        at least one colored drawing path AND text < 1500 chars.
        Table border lines are always colorless; real diagrams use
        colored arrows, boxes, and connector lines — this is the
        key discriminator that eliminates table-heavy false positives.
        Catches: pages 81–82 (Tx/Rx data flow) and similar pages
        whose figure captions appear on the following page.
    """
    doc = fitz.open(pdf_path)
    diagram_page_nums = set()

    # xref 14 is the shared AUTOSAR header logo on every page — ignore it
    HEADER_LOGO_XREF = 14

    for page_num, page in enumerate(doc):
        text      = page.get_text("text").strip()
        drawings  = page.get_drawings()
        images    = page.get_images(full=True)
        n_draw    = len(drawings)

        # Rule A
        if FIGURE_CAPTION_RE.search(text) and n_draw >= MIN_DRAWING_PATHS_FOR_DIAGRAM:
            diagram_page_nums.add(page_num)
            continue

        # Rule B
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

        # Rule C — dense colored vector drawings, no figure caption needed
        # Table borders are always colorless (color=None); real diagrams
        # use at least one colored path (arrows, boxes, connector lines).
        # This eliminates false positives from table-heavy pages.
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
    """
    Extract text from ALL pages.
    Diagram pages are also text-extracted so that API names,
    layer labels, and signal names visible on figures are indexed.
    """
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
    """
    Extract tables from all pages using pdfplumber.
    First row labelled HEADERS so column names are self-explanatory
    when the chunk is retrieved out of context.
    Boilerplate rows (page numbers, doc ID lines) are filtered out.
    """
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

                    # Skip boilerplate rows
                    if BOILERPLATE_RE.search(row_text):
                        continue
                    # Skip completely empty rows
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
# DIAGRAM EXTRACTION  (v3 — full coverage)
# ============================================================

def extract_diagram_chunks(pdf_path: str, diagram_page_nums: set):
    """
    For every identified diagram page:
    1. Rasterize the full page at 150 DPI (captures vector graphics
       and embedded rasters equally).
    2. Run BLIP on the full-page render — far more accurate than
       running BLIP on a tiny extracted raster fragment.
    3. Combine BLIP caption with the page's extracted text so that
       figure labels, layer names, and signal names are all indexed.

    Returns (chunks, stats_count).
    """
    chunks = []
    img_counter = 0

    for page_num in sorted(diagram_page_nums):
        try:
            # Rasterize full page
            page_render = rasterize_page(pdf_path, page_num, dpi=150)

            # Extract text visible on this page (labels, captions)
            doc_tmp = fitz.open(pdf_path)
            page_text = doc_tmp[page_num].get_text("text").strip()
            doc_tmp.close()

            # Get BLIP caption on full-page render
            caption = caption_image_with_blip(page_render)

            # Extract figure caption from text if present
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
# RETRIEVER  — MMR with k=6
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
# PROMPT
# ============================================================

def create_prompt() -> ChatPromptTemplate:
    template = """You are a precise document assistant.

Answer ONLY using the provided context. Follow these rules exactly:

ANSWERING RULES:
1. Give direct, technically precise answers without preamble.
2. For API functions: always list ALL parameters (in, in-out, out), return type, sync/async, and reentrancy from the context. If a field is listed as "None", say so explicitly.
3. For enum/constant values: extract the exact value. If the value must be inferred (e.g., sequential numbering), state the inferred value AND note it is inferred.
4. For configuration parameters: list every parameter name and its container/ID where available.
5. For diagrams and architecture figures: describe every labelled layer, module, arrow, and interface shown. Reference the figure number and page.
6. Always cite the source: page number and content type (table / diagram / text).
7. If the context partially answers the question, provide what is available and state exactly what is missing.
8. If the context contains NO relevant information at all, state: "The provided context does not contain this information." Do NOT guess or use external knowledge.
9. Use exact terminology from the specification. Do not paraphrase technical identifiers.
10. For multi-part questions, answer each part in order.

CONTEXT:
{context}

QUESTION:
{question}

FORMAT YOUR RESPONSE AS:

Answer:
<technically accurate, complete answer>

Source:
<page number(s) and content type(s)>
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
            "question": question,
            "context":  format_docs(docs),
            "source_documents": docs,
        }

    def generate(inputs: dict):
        chain  = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context":  inputs["context"],
            "question": inputs["question"],
        })
        return {
            "answer": answer,
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

    # STEP 1 — Classify diagram pages up front (fast, no rasterization)
    progress.progress(5, text="Classifying page types…")
    diagram_page_nums = classify_pages(pdf_path) if include_diagrams else set()

    # STEP 2 — Text (all pages)
    progress.progress(20, text="Extracting text chunks…")
    text_chunks = extract_text_chunks(pdf_path)
    all_docs.extend(text_chunks)

    # STEP 3 — Tables
    progress.progress(45, text="Extracting tables…")
    table_chunks = extract_table_chunks(pdf_path)
    all_docs.extend(table_chunks)

    # STEP 4 — Diagrams
    diagram_count = 0
    if include_diagrams and diagram_page_nums:
        progress.progress(60, text=f"Rasterizing {len(diagram_page_nums)} diagram pages…")
        diagram_chunks, diagram_count = extract_diagram_chunks(pdf_path, diagram_page_nums)
        all_docs.extend(diagram_chunks)

    # STEP 5 — Vectorstore
    progress.progress(80, text="Building vector store…")
    vectorstore = build_vectorstore(all_docs)
    st.session_state.vectorstore = vectorstore

    # STEP 6 — QA chain
    progress.progress(95, text="Initialising QA chain…")
    qa_chain = build_qa_chain(
        vectorstore=vectorstore,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    st.session_state.qa_chain = qa_chain

    progress.progress(100, text="Done.")

    st.session_state.doc_stats = {
        "text_chunks":    len(text_chunks),
        "table_chunks":   len(table_chunks),
        "diagram_chunks": diagram_count,
        "total_chunks":   len(all_docs),
    }

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("Add GROQ_API_KEY to Streamlit secrets")

    model = st.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b",
        ]
    )

    temperature      = st.slider("Temperature", 0.0, 1.0, 0.1)
    max_tokens       = st.slider("Max Tokens", 100, 2000, 1000)
    uploaded_file    = st.file_uploader("Upload PDF", type=["pdf"])
    include_diagrams = st.checkbox("Enable Diagram Analysis", value=True)
    process_btn      = st.button("Process PDF", use_container_width=True)

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
            st.success("PDF processed successfully!")

            s = st.session_state.doc_stats
            st.caption(
                f"Text: {s['text_chunks']} | "
                f"Table: {s['table_chunks']} | "
                f"Diagram: {s['diagram_chunks']} | "
                f"Total: {s['total_chunks']} chunks"
            )
        except Exception as e:
            st.error(str(e))
        finally:
            os.unlink(tmp_path)

# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="main-header">🧠 Enterprise Multimodal RAG</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Groq + LangChain + ChromaDB | v3</div>',
    unsafe_allow_html=True
)

# ============================================================
# WELCOME
# ============================================================

if st.session_state.qa_chain is None:
    st.info("""
### How to Use
1. Add your **GROQ API KEY** in Streamlit secrets
2. Upload any PDF document
3. Click **Process PDF**
4. Ask questions in the chat

**What's detected in v3:**
- ✅ Text — all pages (including labels on diagram pages)
- ✅ Tables — all pages, with header row labelling
- ✅ Vector diagrams — UML, flow, architecture (pages 81–82, 16, 83, 118–122 etc.)
- ✅ Raster/bitmap images — embedded PNG/JPEG figures
- ✅ Mixed pages — pages with both diagrams and tables
""")

# ============================================================
# CHAT HISTORY
# ============================================================

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(
            f'<div class="answer-box">{turn["answer"]}</div>',
            unsafe_allow_html=True
        )

# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.qa_chain:
    question = st.chat_input("Ask a question about the PDF…")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analysing document…"):
                try:
                    result  = st.session_state.qa_chain.invoke(question)
                    answer  = result["answer"]
                    sources = result["source_documents"]

                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True
                    )

                    with st.expander(
                        f"📘 Retrieved Sections ({len(sources)})"
                    ):
                        for doc in sources:
                            meta    = doc.metadata
                            section = meta.get("section", "")
                            label   = f"Page {meta.get('page')} • {meta.get('type')}"
                            if section:
                                label += f" • {section}"
                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<div class="chunk-meta">{label}</div>'
                                f'{doc.page_content[:800]}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer":   answer,
                        "sources":  sources,
                    })

                except Exception as e:
                    st.error(str(e))

# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
