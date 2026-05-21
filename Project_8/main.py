# ============================================================
# MULTIMODAL RAG FOR PDF QA (ENTERPRISE QUALITY)
# ============================================================
#
# FEATURES
# --------
# ✅ Better Retrieval Quality
# ✅ Better Prompt Engineering
# ✅ Cleaner Answers
# ✅ Technical Specification Optimized
# ✅ AUTOSAR / API / ISO Document Friendly
# ✅ Text + Tables + Images
# ✅ Groq LLM
# ✅ ChromaDB
# ✅ BLIP Image Captioning
# ✅ HuggingFace Embeddings
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
#
# GROQ_API_KEY="your_api_key"
#
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
from langchain_community.vectorstores import Chroma
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
    page_title="Enterprise Multimodal RAG",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.sub-header {
    color: gray;
    margin-bottom: 2rem;
}

.answer-box {
    background: #f4fff5;
    border: 1px solid #d4f5dd;
    padding: 1rem;
    border-radius: 10px;
    font-size: 1rem;
}

.chunk-card {
    background: #fafafa;
    border-left: 4px solid #2ecc71;
    padding: 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.8rem;
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
# HELPERS
# ============================================================

def pil_to_base64(img):

    buf = io.BytesIO()

    img.save(buf, format="PNG")

    return base64.b64encode(
        buf.getvalue()
    ).decode()

# ============================================================
# REMOVE TOC / NOISE PAGES
# ============================================================

def is_noise_text(text):

    text = text.lower()

    patterns = [
        "table of contents",
        "contents",
        "........",
        ".........",
        ".........."
    ]

    return any(p in text for p in patterns)

# ============================================================
# IMAGE CAPTIONING
# ============================================================

def caption_image_with_blip(pil_img):

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

def extract_text_chunks(pdf_path):

    doc = fitz.open(pdf_path)

    raw_docs = []

    for page_num, page in enumerate(doc):

        text = page.get_text("text").strip()

        if text and not is_noise_text(text):

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
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    return splitter.split_documents(raw_docs)

# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_chunks(pdf_path):

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

                if table_text.strip():

                    chunks.append(
                        Document(
                            page_content=f"""
TABLE CONTENT:

{table_text}
""",
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

def extract_image_chunks(pdf_path):

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

            except Exception:
                pass

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
        search_type="similarity",
        search_kwargs={"k": 3}
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
You are an expert AUTOSAR specification assistant.

Answer ONLY using the provided context.

RULES:
1. Give direct, technically precise answers.
2. When asked for values (e.g., enums, constants), extract the exact value from tables or text. If not explicitly stated but can be inferred (e.g., next value after 0 is 1), state the value and mention it is inferred from context.
3. For API parameters, always look for and output the exact C-style function signature if present in the context. List all parameters with their names and types if available. If not all are listed, state what is available and mention if any are inferred.
4. For table-based questions, use the table headers and values as shown in the document.
5. For diagrams or architecture, describe the structure and reference the diagram or page.
6. For image-based questions, use the extracted image captions and descriptions.
7. Always mention the source: page number and whether it is from a table, diagram, image, or text.
8. If the answer comes from a table, diagram, image, or is inferred, explicitly say so.
9. Use exact wording from the specification where possible.
10. Never say "not found" if the answer can be inferred or partially answered from the context.
11. Keep answers concise, professional, and technically accurate.

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE FORMAT:

Answer:
<direct, technically accurate answer>

Source:
<Page number and type (e.g., table, diagram, image, text, or inferred)>
"""
# ============================================================
# FUNCTION SIGNATURE AND TABLE HEADER EXTRACTION
# ============================================================

import re

def extract_function_signatures(text):
    """
    Extract C-style function signatures from a block of text.
    Returns a list of signatures found.
    """
    # Only process if input is a string
    if not isinstance(text, str):
        return []
    try:
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_\s\*]+\([a-zA-Z0-9_,\s\*\[\]\.]*(void)?\);"
        matches = re.findall(pattern, text, re.MULTILINE)
        return matches if matches else []
    except Exception:
        return []

def extract_table_headers(table_text):
    """
    Extract table headers from a table chunk (first row).
    Returns the header row if present.
    """
    lines = table_text.strip().splitlines()
    if lines:
        header = lines[0]
        # Heuristic: header row usually has column names separated by |
        if '|' in header:
            return header
    return None

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{question}")
        ]
    )

    return prompt

# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs: List[Document]):

    formatted = []
    all_signatures = []
    all_table_headers = []

    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("type", "text")
        content = doc.page_content
        # Fully robust extraction of function signatures
        signatures = []
        if isinstance(content, str):
            try:
                signatures = extract_function_signatures(content)
            except Exception:
                signatures = []
        if signatures:
            all_signatures.extend([f"Page {page}: {sig}" for sig in signatures])
        # Extract table headers if this is a table chunk
        if dtype == "table":
            header = extract_table_headers(content)
            if header:
                all_table_headers.append(f"Page {page}: {header}")
        formatted.append(
            f"""
DOCUMENT {i}

Page: {page}
Type: {dtype}

CONTENT:
{content}
"""
        )

    # Prepend all found signatures and table headers to the context
    context_blocks = []
    if all_signatures:
        context_blocks.append("FUNCTION SIGNATURES FOUND IN DOCUMENT:\n" + "\n".join(all_signatures))
    if all_table_headers:
        context_blocks.append("TABLE HEADERS FOUND IN DOCUMENT:\n" + "\n".join(all_table_headers))
    context_blocks.append("\n\n".join(formatted))
    return "\n\n".join(context_blocks)

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

    # Error handling: If no content was extracted, show error and abort
    if not all_docs:
        st.session_state.qa_chain = None
        st.session_state.vectorstore = None
        st.error("No extractable content (text, tables, or images) found in the uploaded PDF. Please check the document and try again.")
        return

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
        st.error("Add GROQ_API_KEY to Streamlit secrets")

    model = st.selectbox(
        "Groq Model",
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
        0.1
    )

    max_tokens = st.slider(
        "Max Tokens",
        100,
        2000,
        700
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

# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="main-header">🧠 Enterprise Multimodal RAG</div>',
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

1. Add GROQ API KEY
2. Upload PDF
3. Process Document
4. Ask Questions

Optimized for:
- AUTOSAR Docs
- Technical Specifications
- API Manuals
- ISO Documents
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

if st.session_state.qa_chain is not None:
    question = st.chat_input(
        "Ask question about the PDF..."
    )
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
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
                        f"📘 Relevant Specification Sections ({len(sources)})"
                    ):
                        for doc in sources:
                            meta = doc.metadata
                            st.markdown(
                                f"""
<div class="chunk-card">

<div class="chunk-meta">
Page {meta.get("page")} • {meta.get("type")}
</div>

{doc.page_content[:700]}

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
else:
    st.warning("No QA chain available. Please upload and process a valid PDF document with extractable content (text, tables, or images) before asking questions.")

# ============================================================
# IMAGE PREVIEW
# ============================================================

if st.session_state.extracted_imgs:

    st.divider()

    st.subheader("🖼 Extracted Images")

    for img in st.session_state.extracted_imgs:

        st.image(
            f"data:image/png;base64,{img['b64']}",
            width=300
        )

        st.caption(
            f"Page {img['page']} • {img['caption']}"
        )

# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.chat_history:

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()