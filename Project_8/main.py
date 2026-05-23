import os
import re
import uuid
import tempfile
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import fitz
import pdfplumber
import camelot
import numpy as np
import pandas as pd

from PIL import Image
from paddleocr import PaddleOCR

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_groq import ChatGroq

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise MultiModal RAG",
    layout="wide"
)

# ============================================================
# OCR
# ============================================================

@st.cache_resource
def load_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang='en'
    )

ocr_engine = load_ocr()

# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

# ============================================================
# RERANKER
# ============================================================

@st.cache_resource
def load_reranker():
    return CrossEncoder(
        'cross-encoder/ms-marco-MiniLM-L-6-v2'
    )

# ============================================================
# LLM
# ============================================================

@st.cache_resource
def load_llm(model_name):

    api_key = st.secrets["GROQ_API_KEY"]

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=2000
    )

# ============================================================
# HELPERS
# ============================================================

HEADING_RE = re.compile(
    r'^(\\d+(\\.\\d+){0,4})\\s+(.+)$',
    re.MULTILINE
)

def extract_heading(text):
    matches = list(HEADING_RE.finditer(text))
    if matches:
        return matches[-1].group(0)
    return ""

# ============================================================
# PAGE CLASSIFIER
# ============================================================

def is_diagram_page(page):

    drawings = len(page.get_drawings())
    images = len(page.get_images(full=True))
    text = page.get_text("text")

    score = 0

    if drawings > 15:
        score += 2

    if images > 0:
        score += 3

    if len(text) < 500:
        score += 1

    if re.search(
        r'figure|diagram|architecture|overview|flow',
        text,
        re.I
    ):
        score += 2

    return score >= 3

# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_ocr_text(image_path):

    result = ocr_engine.ocr(image_path)

    lines = []

    for block in result:
        for line in block:
            lines.append(line[1][0])

    return "\n".join(lines)

# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_chunks(pdf_path):

    doc = fitz.open(pdf_path)

    raw_docs = []

    for page_num, page in enumerate(doc):

        text = page.get_text("text")

        if not text.strip():
            continue

        heading = extract_heading(text)

        raw_docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "type": "text",
                    "section": heading,
                    "chunk_id": str(uuid.uuid4())
                }
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    split_docs = splitter.split_documents(raw_docs)

    doc.close()

    return split_docs

# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_tables(pdf_path):

    docs = []

    try:

        tables = camelot.read_pdf(
            pdf_path,
            pages='all',
            flavor='stream'
        )

        for idx, table in enumerate(tables):

            df = table.df

            table_text = df.to_string(index=False)

            docs.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "type": "table",
                        "table_id": idx
                    }
                )
            )

    except:
        pass

    return docs

# ============================================================
# DIAGRAM EXTRACTION
# ============================================================

def extract_diagram_chunks(pdf_path):

    os.makedirs("temp_images", exist_ok=True)

    doc = fitz.open(pdf_path)

    docs = []

    for page_num, page in enumerate(doc):

        if not is_diagram_page(page):
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        image_path = f"temp_images/page_{page_num+1}.png"

        pix.save(image_path)

        ocr_text = extract_ocr_text(image_path)

        content = f"""
DIAGRAM PAGE: {page_num+1}

OCR TEXT:
{ocr_text}
"""

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "page": page_num + 1,
                    "type": "diagram"
                }
            )
        )

    doc.close()

    return docs

# ============================================================
# VECTOR STORE
# ============================================================

def build_vectorstore(documents):

    embeddings = load_embeddings()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="autosar_rag"
    )

    return vectordb

# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(self, vectordb, docs):

        self.vectordb = vectordb
        self.docs = docs

        texts = [d.page_content for d in docs]

        tokenized = [t.split() for t in texts]

        self.bm25 = BM25Okapi(tokenized)

        self.reranker = load_reranker()

    def retrieve(self, query, k=6):

        dense_docs = self.vectordb.similarity_search(
            query,
            k=15
        )

        bm25_scores = self.bm25.get_scores(query.split())

        bm25_top = np.argsort(bm25_scores)[::-1][:15]

        bm25_docs = [self.docs[i] for i in bm25_top]

        merged = dense_docs + bm25_docs

        unique_docs = []
        seen = set()

        for d in merged:

            key = d.page_content[:100]

            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        pairs = [[query, d.page_content] for d in unique_docs]

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(unique_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [x[0] for x in ranked[:k]]

# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are an AUTOSAR technical expert.

Answer ONLY using the provided context.

RULES:
1. Never hallucinate.
2. Preserve exact API names.
3. Preserve parameter names.
4. Include page numbers.
5. Explain tables accurately.
6. Explain diagrams using OCR text.
7. If answer missing say:
   'The provided context does not contain this information.'

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(
    PROMPT_TEMPLATE
)

# ============================================================
# FORMAT DOCS
# ============================================================

def format_docs(docs):

    parts = []

    for d in docs:

        meta = d.metadata

        txt = f"""
PAGE: {meta.get('page')}

TYPE: {meta.get('type')}

SECTION: {meta.get('section', '')}

CONTENT:
{d.page_content}
"""

        parts.append(txt)

    return "\n\n".join(parts)

# ============================================================
# RAG CHAIN
# ============================================================

def build_rag_chain(retriever, llm):

    def retrieve(question):

        docs = retriever.retrieve(question)

        return {
            "question": question,
            "context": format_docs(docs),
            "docs": docs
        }

    def generate(inputs):

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke({
            "context": inputs["context"],
            "question": inputs["question"]
        })

        return {
            "answer": answer,
            "docs": inputs["docs"]
        }

    return RunnableLambda(retrieve) | RunnableLambda(generate)

# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(pdf_path, model_name):

    text_docs = extract_text_chunks(pdf_path)

    table_docs = extract_tables(pdf_path)

    diagram_docs = extract_diagram_chunks(pdf_path)

    all_docs = (
        text_docs
        + table_docs
        + diagram_docs
    )

    vectordb = build_vectorstore(all_docs)

    retriever = HybridRetriever(
        vectordb,
        all_docs
    )

    llm = load_llm(model_name)

    qa_chain = build_rag_chain(
        retriever,
        llm
    )

    return qa_chain, len(all_docs)

# ============================================================
# SESSION STATE
# ============================================================

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Configuration")

    model_name = st.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b"
        ]
    )

    uploaded = st.file_uploader(
        "Upload AUTOSAR PDF",
        type=["pdf"]
    )

    process_btn = st.button(
        "Process PDF",
        use_container_width=True
    )

# ============================================================
# MAIN UI
# ============================================================

st.title("Enterprise Generic Multi-Modal RAG")

st.markdown("""
Supports:
- AUTOSAR Classic
- AUTOSAR Adaptive
- UML diagrams
- Tables
- OCR
- Scanned PDFs
- Architecture documents
""")

# ============================================================
# PROCESS PDF
# ============================================================

if process_btn and uploaded:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded.read())

        tmp_path = tmp.name

    with st.spinner("Processing PDF..."):

        qa_chain, total_chunks = process_pdf(
            tmp_path,
            model_name
        )

        st.session_state.qa_chain = qa_chain

    st.success(
        f"PDF processed successfully. "
        f"Total chunks: {total_chunks}"
    )

# ============================================================
# CHAT
# ============================================================

if st.session_state.qa_chain:

    question = st.chat_input(
        "Ask question about PDF"
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                result = st.session_state.qa_chain.invoke(
                    question
                )

                st.write(result["answer"])

                with st.expander("Retrieved Chunks"):

                    for d in result["docs"]:

                        st.write(d.metadata)

                        st.write(
                            d.page_content[:1000]
                        )