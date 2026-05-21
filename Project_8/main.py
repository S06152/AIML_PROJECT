# ============================================================
#  MULTIMODAL RAG PIPELINE — STREAMLIT APP
#  Text + Tables + Images from PDFs
#
#  STACK:
#  - LangChain
#  - ChromaDB
#  - Groq LLM
#  - HuggingFace Embeddings
#  - BLIP Image Captioning
#
#  RUN:
#     streamlit run app.py
#
#  REQUIREMENTS:
#     pip install streamlit pymupdf pdfplumber pillow torch
#     pip install transformers chromadb sentence-transformers
#     pip install langchain langchain-core langchain-community
#     pip install langchain-text-splitters langchain-chroma
#     pip install langchain-groq
#
#  STREAMLIT SECRETS:
#     .streamlit/secrets.toml
#
#     GROQ_API_KEY="your_api_key"
# ============================================================

import os
import io
import base64
import tempfile
import warnings
warnings.filterwarnings("ignore")

from typing import List

import streamlit as st
import fitz
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
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# BLIP
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.sub-header {
    color: gray;
    margin-bottom: 2rem;
}

.answer-box {
    background: #f3fdf5;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #d4f5dd;
}

.chunk-card {
    background: #fafafa;
    padding: 0.8rem;
    border-radius: 8px;
    border-left: 4px solid #2ecc71;
    margin-bottom: 0.7rem;
}

.chunk-meta {
    color: gray;
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
}
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
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# ============================================================
# BLIP MODEL
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
# IMAGE HELPERS
# ============================================================

def pil_to_base64(img: Image.Image):

    buf = io.BytesIO()

    img.save(buf, format="PNG")

    return base64.b64encode(
        buf.getvalue()
    ).decode()

# ============================================================
# IMAGE CAPTIONING
# ============================================================

def caption_image_with_blip(pil_img: Image.Image):

    processor, model = load_blip_model()

    inputs = processor(
        images=pil_img,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=80
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption.strip()

# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_chunks(pdf_path: str):

    doc = fitz.open(pdf_path)

    raw_docs = []

    for page_num, page in enumerate(doc):

        text = page.get_text("text").strip()

        if text:

            raw_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "type": "text"
                    }
                )
            )

    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(raw_docs)

# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_chunks(pdf_path: str):

    chunks = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages):

            tables = page.extract_tables()

            for idx, table in enumerate(tables):

                if not table:
                    continue

                rows = []

                for row in table:

                    cleaned = [
                        str(c).strip() if c else ""
                        for c in row
                    ]

                    rows.append(
                        " | ".join(cleaned)
                    )

                table_text = "\n".join(rows)

                chunks.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "page": page_num + 1,
                            "type": "table"
                        }
                    )
                )

    return chunks

# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_image_chunks(pdf_path: str):

    doc = fitz.open(pdf_path)

    chunks = []

    previews = []

    img_counter = 0

    for page_num, page in enumerate(doc):

        images = page.get_images(full=True)

        for img in images:

            try:

                xref = img[0]

                base_img = doc.extract_image(xref)

                img_bytes = base_img["image"]

                pil_img = Image.open(
                    io.BytesIO(img_bytes)
                ).convert("RGB")

                # Skip tiny icons
                if pil_img.width < 80 or pil_img.height < 80:
                    continue

                img_counter += 1

                caption = caption_image_with_blip(pil_img)

                previews.append({
                    "page": page_num + 1,
                    "index": img_counter,
                    "caption": caption,
                    "b64": pil_to_base64(pil_img)
                })

                chunks.append(
                    Document(
                        page_content=f"""
                        IMAGE DESCRIPTION:
                        {caption}
                        """,
                        metadata={
                            "page": page_num + 1,
                            "type": "image"
                        }
                    )
                )

            except Exception as e:
                st.warning(f"Image skipped: {e}")

    doc.close()

    return chunks, previews

# ============================================================
# VECTORSTORE
# ============================================================

def build_vectorstore(all_docs):

    embeddings = load_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="multimodal_rag"
    )

    return vectorstore

# ============================================================
# RETRIEVER
# ============================================================

def create_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    return retriever

# ============================================================
# LLM
# ============================================================

def initialize_llm(
    model,
    api_key,
    temperature,
    max_tokens
):

    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return llm

# ============================================================
# PROMPT
# ============================================================

def create_prompt():

    template = """
You are a helpful assistant answering questions
from a PDF document.

The document may contain:
- Text
- Tables
- Image descriptions

Use ONLY the context below.

If answer not found, say:
"I don't have enough information in this document."

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Mention page numbers
- Mention whether answer came from text/table/image
- Be concise
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{question}")
        ]
    )

    return prompt

# ============================================================
# FORMAT DOCS
# ============================================================

def format_docs(docs: List[Document]):

    formatted = []

    for doc in docs:

        page = doc.metadata.get("page", "?")

        dtype = doc.metadata.get("type", "text")

        formatted.append(
            f"[Page {page} | Type: {dtype}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)

# ============================================================
# RAG CHAIN
# ============================================================

def create_rag_chain(retriever, llm):

    prompt = create_prompt()

    def retrieve(question):

        docs = retriever.invoke(question)

        return {
            "question": question,
            "context": format_docs(docs),
            "source_documents": docs
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
            "source_documents": inputs["source_documents"]
        }

    rag_chain = (
        RunnableLambda(retrieve)
        | RunnableLambda(generate)
    )

    return rag_chain

# ============================================================
# BUILD QA CHAIN
# ============================================================

def build_qa_chain(
    vectorstore,
    model_name,
    api_key,
    temperature,
    max_tokens
):

    retriever = create_retriever(vectorstore)

    llm = initialize_llm(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    qa_chain = create_rag_chain(
        retriever,
        llm
    )

    return qa_chain

# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path,
    model_name,
    api_key,
    include_images,
    temperature,
    max_tokens
):

    progress = st.progress(0)

    all_docs = []

    # TEXT
    progress.progress(20)

    text_chunks = extract_text_chunks(pdf_path)

    all_docs.extend(text_chunks)

    # TABLES
    progress.progress(40)

    table_chunks = extract_table_chunks(pdf_path)

    all_docs.extend(table_chunks)

    # IMAGES
    img_chunks = []

    if include_images:

        progress.progress(60)

        img_chunks, previews = extract_image_chunks(
            pdf_path
        )

        all_docs.extend(img_chunks)

        st.session_state.extracted_imgs = previews

    # VECTORSTORE
    progress.progress(80)

    vectorstore = build_vectorstore(all_docs)

    st.session_state.vectorstore = vectorstore

    # QA CHAIN
    progress.progress(100)

    qa_chain = build_qa_chain(
        vectorstore=vectorstore,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    st.session_state.qa_chain = qa_chain

    st.session_state.doc_stats = {
        "text_chunks": len(text_chunks),
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
        st.error("Add GROQ_API_KEY in .streamlit/secrets.toml")

    model = st.selectbox(
        "Select Groq Model",
        [
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b"
        ]
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.2
    )

    max_tokens = st.slider(
        "Max Tokens",
        100,
        2000,
        800
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    include_images = st.checkbox(
        "Enable Image Analysis",
        value=True
    )

    process_btn = st.button(
        "Process PDF",
        use_container_width=True
    )

    if process_btn and uploaded_file:

        st.session_state.chat_history = []
        st.session_state.extracted_imgs = []

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

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

        except Exception as e:

            st.error(str(e))

        finally:

            os.unlink(tmp_path)

    # Stats
    if st.session_state.doc_stats:

        st.divider()

        stats = st.session_state.doc_stats

        st.write(stats)

# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="main-header">🧠 Multimodal RAG Pipeline</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'Groq + LangChain + BLIP + ChromaDB'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# WELCOME
# ============================================================

if st.session_state.qa_chain is None:

    st.info("""
    ### How to Use

    1. Add GROQ API KEY to Streamlit secrets
    2. Upload PDF
    3. Click Process PDF
    4. Ask questions
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

    question = st.chat_input(
        "Ask question about PDF..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    result = st.session_state.qa_chain.invoke(
                        question
                    )

                    answer = result["answer"]

                    sources = result["source_documents"]

                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True
                    )

                    with st.expander(
                        f"Sources ({len(sources)})"
                    ):

                        for doc in sources:

                            meta = doc.metadata

                            st.markdown(
                                f"""
                                <div class="chunk-card">
                                    <div class="chunk-meta">
                                        Page {meta.get("page")}
                                        | {meta.get("type")}
                                    </div>

                                    {doc.page_content[:500]}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(str(e))

# ============================================================
# IMAGE PREVIEWS
# ============================================================

if st.session_state.extracted_imgs:

    st.divider()

    st.subheader("Extracted Images")

    for img in st.session_state.extracted_imgs:

        st.image(
            f"data:image/png;base64,{img['b64']}",
            width=300
        )

        st.caption(
            f"Page {img['page']} | {img['caption']}"
        )

# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.chat_history:

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()