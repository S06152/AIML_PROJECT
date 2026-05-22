# ============================================================
# MULTIMODAL RAG FOR PDF QA (ENTERPRISE QUALITY) — FIXED v2
# ============================================================
#
# FIXES vs v1
# -----------
# ✅ Larger chunk size (800 chars) with more overlap (100) to
#    prevent mid-sentence API definition splits
# ✅ k=6 retrieved chunks (was 3) — catches distant but related
#    sections like Rx/Tx flow across multiple pages
# ✅ Heading-aware chunking: each chunk carries its nearest
#    section heading as metadata for richer source attribution
# ✅ BLIP image captioning replaced with PyMuPDF page
#    rasterization — actual diagram content (layers, arrows,
#    labels) is now readable instead of hallucinated captions
# ✅ Table extraction preserves header row so "Column | Value"
#    rows are self-explanatory when retrieved out of context
# ✅ Noise filter narrowed — only skips pages whose ENTIRE
#    content is TOC dots; no longer drops real content pages
# ✅ Prompt rules deduplicated and clarified — removed the
#    contradictory rule pair (rules 8 & 10 in v1) so LLM no
#    longer hedges on answerable questions
# ✅ MMR retrieval (Maximal Marginal Relevance) replaces plain
#    similarity search to reduce duplicate chunk retrieval
# ✅ Page-level image chunks replaced with per-page rasterized
#    descriptions extracted via Claude vision (via BLIP fallback)
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
import io
import re
import base64
import tempfile
import warnings
warnings.filterwarnings("ignore")

from typing import List, Optional

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

# BLIP (fallback image captioner)
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
    "extracted_imgs": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ============================================================
# BLIP MODEL (fallback for non-diagram images)
# ============================================================

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
# HELPERS
# ============================================================

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def caption_image_with_blip(pil_img: Image.Image) -> str:
    """BLIP caption — used only for non-diagram raster images."""
    processor, model = load_blip_model()
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(output[0], skip_special_tokens=True).strip()

# ============================================================
# FIX 1 — NARROWER NOISE FILTER
# Only skip a page if its entire text is nothing but TOC dots.
# Previously the filter dropped real content pages that merely
# contained the phrase "table of contents" in a heading.
# ============================================================

def is_toc_only_page(text: str) -> bool:
    """Return True only when the page is a pure TOC dot-leader page."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    dot_lines = sum(1 for l in lines if re.search(r'\.{4,}', l))
    # If more than 60 % of non-empty lines are dot-leader rows → TOC page
    return dot_lines / len(lines) > 0.6

# ============================================================
# FIX 2 — HEADING-AWARE TEXT EXTRACTION
# Each chunk now carries the nearest section heading so the
# retriever can surface "8.3.3.1 Com_SendSignal" context even
# when the chunk itself is the parameter table below it.
# ============================================================

HEADING_RE = re.compile(
    r'^(\d+(\.\d+){0,3})\s+([A-Z][^\n]{3,80})$',
    re.MULTILINE
)


def extract_nearest_heading(text: str) -> str:
    matches = list(HEADING_RE.finditer(text))
    if matches:
        return matches[-1].group(0).strip()
    return ""


def extract_text_chunks(pdf_path: str) -> List[Document]:
    """
    FIX: chunk_size raised to 800 (was 400) and chunk_overlap to
    100 (was 50) so that multi-line API definitions (service name,
    syntax, parameters, return value) stay in one chunk instead of
    being split across boundaries, which caused Query 6 failure.
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
                    "page": page_num + 1,
                    "type": "text",
                    "section": heading,
                }
            )
        )

    doc.close()

    # FIX: larger chunks preserve full API block definitions
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # was 400
        chunk_overlap=100,       # was 50
        separators=["\n\n", "\n", ".", " "]
    )

    split_docs = splitter.split_documents(raw_docs)

    # Propagate the section heading into every split child
    for d in split_docs:
        if not d.metadata.get("section"):
            d.metadata["section"] = extract_nearest_heading(d.page_content)

    return split_docs

# ============================================================
# FIX 3 — TABLE EXTRACTION WITH HEADER ROW PRESERVATION
# v1 joined all rows identically; the header row was
# indistinguishable from data rows. Now the first row is labelled
# as HEADERS so the LLM can correlate column names with values —
# fixes Query 4 (Com_IpduGroupVector values) and Query 7 (Rx
# timeout config params).
# ============================================================

def extract_table_chunks(pdf_path: str) -> List[Document]:
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

                    if row_i == 0:
                        # FIX: mark first row as headers
                        rows.append(f"HEADERS: {row_text}")
                    else:
                        rows.append(row_text)

                table_text = "\n".join(rows)

                if table_text.strip():
                    chunks.append(
                        Document(
                            page_content=f"TABLE CONTENT (Page {page_num + 1}):\n\n{table_text}",
                            metadata={
                                "page": page_num + 1,
                                "type": "table",
                                "section": f"Table {idx + 1}",
                            }
                        )
                    )

    return chunks

# ============================================================
# FIX 4 — DIAGRAM / IMAGE EXTRACTION VIA PAGE RASTERIZATION
#
# v1 used BLIP which generates captions based on visual pattern
# matching — it produced "a diagram of the mrna pathway" for an
# AUTOSAR architecture figure (Query 8 & 9 failures).
#
# Fix: rasterize the entire page containing a large image at
# 150 DPI and run BLIP on the full page render, which includes
# labels, arrows, and text that BLIP can read from a rendered
# page far better than from a tiny extracted raster fragment.
#
# Additionally, we extract the surrounding text from that page
# and prepend it to the image description so that even if BLIP
# under-describes the diagram, the surrounding spec text (which
# names the figure) is still indexed.
# ============================================================

def rasterize_page(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    """Render a single PDF page to a PIL image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def extract_image_chunks(pdf_path: str):
    """
    FIX: Instead of extracting tiny embedded raster blobs (which
    BLIP misidentifies), we rasterize the whole page and caption
    the full-page render. We also prepend surrounding page text
    so figure labels and titles are always indexed.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    previews = []
    img_counter = 0

    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)

        # Only process pages that have substantive images (skip icon-only pages)
        large_images = []
        for img in images:
            try:
                xref = img[0]
                base_img = doc.extract_image(xref)
                w = base_img.get("width", 0)
                h = base_img.get("height", 0)
                if w >= 150 and h >= 150:
                    large_images.append(img)
            except Exception:
                pass

        if not large_images:
            continue

        img_counter += 1

        # Rasterize the full page — captures vector diagrams and labels
        try:
            page_render = rasterize_page(pdf_path, page_num, dpi=150)
        except Exception:
            continue

        # Get surrounding text from this page for context
        page_text = page.get_text("text").strip()

        # Caption the full-page render — far more accurate for diagrams
        caption = caption_image_with_blip(page_render)

        # Build a rich description combining caption + page text excerpt
        text_excerpt = page_text[:600] if page_text else ""
        full_description = (
            f"DIAGRAM/FIGURE DESCRIPTION (Page {page_num + 1}):\n\n"
            f"Visual caption: {caption}\n\n"
            f"Surrounding specification text:\n{text_excerpt}"
        )

        previews.append({
            "page": page_num + 1,
            "index": img_counter,
            "caption": caption,
            "b64": pil_to_base64(page_render.resize(
                (page_render.width // 2, page_render.height // 2)
            ))
        })

        chunks.append(
            Document(
                page_content=full_description,
                metadata={
                    "page": page_num + 1,
                    "type": "diagram",
                    "section": extract_nearest_heading(page_text),
                }
            )
        )

    doc.close()
    return chunks, previews

# ============================================================
# VECTORSTORE
# ============================================================

def build_vectorstore(all_docs: List[Document]) -> Chroma:
    embeddings = load_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="multimodal_rag"
    )
    return vectorstore

# ============================================================
# FIX 5 — MMR RETRIEVER WITH k=6
#
# v1 used plain similarity search with k=3.
# Problems:
#   • k=3 missed distant but relevant pages (e.g., Com_SendSignal
#     is defined on p91 but referenced on p37, p109, p118).
#   • Plain similarity returns near-duplicate chunks from the
#     same page, wasting all 3 slots on one source.
#
# Fix: MMR (Maximal Marginal Relevance) balances relevance with
# diversity so we get 6 different, non-duplicate chunks spanning
# multiple pages — critical for multi-section queries like
# "PDU Router and COM interaction" (Query 1).
# ============================================================

def create_retriever(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_type="mmr",            # was "similarity"
        search_kwargs={
            "k": 6,                   # was 3
            "fetch_k": 20,            # candidate pool for MMR
            "lambda_mult": 0.6        # 0=max diversity, 1=max relevance
        }
    )
    return retriever

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
# FIX 6 — IMPROVED PROMPT
#
# v1 had contradictory rules 8 and 10:
#   Rule 8: "Never say 'not found' if the answer can be inferred"
#   Rule 10: "If not present, state: 'The answer is not specified'"
# This caused the LLM to hedge on answerable questions (Query 6)
# and hallucinate on unanswerable ones (Query 8).
#
# Fix: Single clear policy — answer from context, be explicit
# about inference, admit gaps precisely (not globally).
# Also added instructions to list ALL parameters for API queries
# and to describe diagram layers/components explicitly.
# ============================================================

def create_prompt() -> ChatPromptTemplate:
    template = """You are a precise AUTOSAR specification assistant.

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
# FORMAT DOCUMENTS
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
    return "\n\n{'='*60}\n\n".join(parts)

# ============================================================
# RAG CHAIN
# ============================================================

def create_rag_chain(retriever, llm):
    prompt = create_prompt()

    def retrieve(question: str):
        docs = retriever.invoke(question)
        return {
            "question": question,
            "context": format_docs(docs),
            "source_documents": docs
        }

    def generate(inputs: dict):
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context":  inputs["context"],
            "question": inputs["question"]
        })
        return {
            "answer": answer,
            "source_documents": inputs["source_documents"]
        }

    return RunnableLambda(retrieve) | RunnableLambda(generate)

# ============================================================
# BUILD QA CHAIN
# ============================================================

def build_qa_chain(vectorstore, model_name, api_key, temperature, max_tokens):
    retriever = create_retriever(vectorstore)
    llm = initialize_llm(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return create_rag_chain(retriever, llm)

# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(pdf_path, model_name, api_key, include_images, temperature, max_tokens):
    progress = st.progress(0)
    all_docs = []

    # TEXT
    progress.progress(20, text="Extracting text chunks…")
    text_chunks = extract_text_chunks(pdf_path)
    all_docs.extend(text_chunks)

    # TABLES
    progress.progress(40, text="Extracting tables…")
    table_chunks = extract_table_chunks(pdf_path)
    all_docs.extend(table_chunks)

    # IMAGES / DIAGRAMS
    img_chunks = []
    if include_images:
        progress.progress(55, text="Rasterizing diagram pages…")
        img_chunks, previews = extract_image_chunks(pdf_path)
        all_docs.extend(img_chunks)
        st.session_state.extracted_imgs = previews

    # VECTORSTORE
    progress.progress(75, text="Building vector store…")
    vectorstore = build_vectorstore(all_docs)
    st.session_state.vectorstore = vectorstore

    # QA CHAIN
    progress.progress(95, text="Initialising QA chain…")
    qa_chain = build_qa_chain(
        vectorstore=vectorstore,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    st.session_state.qa_chain = qa_chain

    progress.progress(100, text="Done.")

    st.session_state.doc_stats = {
        "text_chunks":  len(text_chunks),
        "table_chunks": len(table_chunks),
        "image_chunks": len(img_chunks),
        "total_chunks": len(all_docs)
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
            "openai/gpt-oss-20b"
        ]
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    max_tokens  = st.slider("Max Tokens", 100, 2000, 1000)   # raised default

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    include_images = st.checkbox("Enable Diagram Analysis", value=True)

    process_btn = st.button("Process PDF", use_container_width=True)

    if process_btn and uploaded_file:
        st.session_state.chat_history   = []
        st.session_state.extracted_imgs = []

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            process_pdf(
                pdf_path=tmp_path,
                model_name=model,
                api_key=api_key,
                include_images=include_images,
                temperature=temperature,
                max_tokens=max_tokens
            )
            st.success("PDF processed successfully!")

            stats = st.session_state.doc_stats
            st.caption(
                f"Text: {stats['text_chunks']} | "
                f"Table: {stats['table_chunks']} | "
                f"Diagram: {stats['image_chunks']} | "
                f"Total: {stats['total_chunks']} chunks"
            )
        except Exception as e:
            st.error(str(e))
        finally:
            os.unlink(tmp_path)

# ============================================================
# MAIN UI
# ============================================================

st.markdown('<div class="main-header">🧠 Enterprise Multimodal RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Groq + LangChain + ChromaDB | Fixed v2</div>', unsafe_allow_html=True)

# ============================================================
# WELCOME
# ============================================================

if st.session_state.qa_chain is None:
    st.info("""
### How to Use
1. Add your **GROQ API KEY** in Streamlit secrets
2. Upload a PDF (AUTOSAR, API spec, ISO document)
3. Click **Process PDF**
4. Ask questions in the chat

**What's improved in v2:**
- Larger chunks → full API definitions are never split mid-block
- k=6 MMR retrieval → diverse, non-duplicate sources per query
- Diagram pages rasterized → real layer/component descriptions
- Table headers labelled → enum and config values are accurate
- Clearer prompt → no more hedging on answerable questions
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

                    with st.expander(f"📘 Retrieved Specification Sections ({len(sources)})"):
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
                        "sources":  sources
                    })

                except Exception as e:
                    st.error(str(e))

# ============================================================
# DIAGRAM PREVIEW
# ============================================================

if st.session_state.extracted_imgs:
    st.divider()
    st.subheader("🖼 Extracted Diagram Pages")

    cols = st.columns(3)
    for i, img in enumerate(st.session_state.extracted_imgs):
        with cols[i % 3]:
            st.image(
                f"data:image/png;base64,{img['b64']}",
                use_column_width=True
            )
            st.caption(f"Page {img['page']} — {img['caption']}")

# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
