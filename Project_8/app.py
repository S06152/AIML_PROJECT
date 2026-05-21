# ============================================================
#  Multimodal RAG Pipeline — Streamlit App (100% Open Source)
#  Handles : Text + Images + Tables from PDF files
#  Stack   : LangChain · ChromaDB · Ollama · HuggingFace BLIP
#  Cost    : FREE — no paid API key needed
# ============================================================
#
#  PRE-REQUISITES
#  ──────────────
#  1. Install Ollama (local LLM runner):
#       https://ollama.com/download
#     Then pull a model (choose ONE based on your RAM):
#       ollama pull llama3       # best quality  (~5 GB RAM)
#       ollama pull mistral      # good quality  (~4 GB RAM)
#       ollama pull phi3         # lightest       (~2 GB RAM)
#     Start the server:
#       ollama serve
#
#  2. Install Python packages:
#       pip install -r requirements_opensource.txt
#
#  3. Run the app:
#       streamlit run multimodal_rag_opensource.py
# ============================================================

import os
import io
import base64
import tempfile
import warnings
warnings.filterwarnings("ignore")
from typing import List

import streamlit as st
import fitz                   # PyMuPDF  — text + image extraction
import pdfplumber             # table extraction
from PIL import Image
import torch

# LangChain core
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate

# LangChain open-source integrations
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# HuggingFace BLIP — free image captioning (runs locally)
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.base import BaseLanguageModel


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG — Open Source",
    page_icon="🦙",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .chunk-card {
        background: #f8f9fa;
        border-left: 4px solid #2ecc71;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: #333;
    }
    .chunk-meta {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 4px;
    }
    .answer-box {
        background: #eafaf1;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        font-size: 1rem;
        color: #1a1a2e;
        border: 1px solid #a9dfbf;
    }
    .stat-box {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .stat-number { font-size: 1.5rem; font-weight: 700; color: #27ae60; }
    .stat-label  { font-size: 0.75rem; color: #777; }
    .model-pill {
        display: inline-block;
        background: #eafaf1;
        border: 1px solid #a9dfbf;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #1e8449;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "vectorstore":    None,
    "qa_chain":       None,
    "chat_history":   [],
    "doc_stats":      {},
    "extracted_imgs": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════
#  CACHED MODEL LOADERS
#  @st.cache_resource → loaded once, reused
# ═══════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~80 MB)…")
def load_embedding_model():
    """
    HuggingFace sentence-transformers — fully local, no API key.
    'all-MiniLM-L6-v2'  → small & fast  (~80 MB)
    'BAAI/bge-base-en-v1.5' → better quality (~430 MB)
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading BLIP image captioning model (~900 MB on first run)…")
def load_blip_model():
    """
    Salesforce BLIP — free image-to-text model.
    Downloads once, then cached by HuggingFace locally.
    """
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32,
    )
    model.eval()
    return processor, model


# ═══════════════════════════════════════════════
#  HELPER UTILITIES
# ═══════════════════════════════════════════════

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def caption_image_with_blip(pil_img: Image.Image) -> str:
    """
    Run BLIP on a PIL image and return a text caption.
    BLIP reads the pixels and writes a natural language description,
    e.g. 'a bar chart showing quarterly revenue growth'.
    That caption is what gets embedded and stored in ChromaDB.
    """
    processor, model = load_blip_model()
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            early_stopping=True,
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


# ═══════════════════════════════════════════════
#  EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════

def extract_text_chunks(pdf_path: str) -> list[Document]:
    """
    Step 1 — Text extraction.
    PyMuPDF reads every page and pulls the raw text.
    LangChain RecursiveCharacterTextSplitter breaks it into
    overlapping chunks so no context is lost at boundaries.
    """
    doc = fitz.open(pdf_path)
    raw_docs = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            raw_docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page":   page_num + 1,
                    "type":   "text",
                }
            ))
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(raw_docs)


def extract_table_chunks(pdf_path: str) -> list[Document]:
    """
    Step 2 — Table extraction.
    pdfplumber detects structured tables and returns them as
    lists of rows. We convert each table to a readable
    pipe-separated text block so it can be embedded as text.
    """
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table:
                    continue
                rows = []
                for row in table:
                    cleaned = [str(c).strip() if c else "" for c in row]
                    rows.append(" | ".join(cleaned))
                table_text = "\n".join(rows).strip()
                if table_text:
                    chunks.append(Document(
                        page_content=(
                            f"[TABLE on page {page_num + 1}, "
                            f"table {t_idx + 1}]\n{table_text}"
                        ),
                        metadata={
                            "source":  pdf_path,
                            "page":    page_num + 1,
                            "type":    "table",
                            "t_index": t_idx + 1,
                        }
                    ))
    return chunks


def extract_image_chunks(pdf_path: str) -> tuple[list[Document], list[dict]]:
    """
    Step 3 — Image extraction + BLIP captioning.

    For every image in the PDF:
      a) Extract raw bytes with PyMuPDF → convert to PIL Image
      b) Skip tiny images (decorations / icons < 80px)
      c) Run BLIP → get a natural language caption
      d) Store caption as a Document (this is what gets embedded)
      e) Keep a base64 copy for the sidebar preview

    Why BLIP and not CLIP?
      CLIP outputs a similarity score (number), not text.
      ChromaDB stores text. BLIP gives us embeddable text.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    previews = []
    img_counter = 0

    progress_placeholder = st.empty()

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_img  = doc.extract_image(xref)
                img_bytes = base_img["image"]
                pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Skip tiny decorative images
                if pil_img.width < 80 or pil_img.height < 80:
                    continue

                img_counter += 1
                progress_placeholder.info(
                    f"Captioning image {img_counter} on page {page_num + 1}…"
                )

                # Generate caption with BLIP
                caption = caption_image_with_blip(pil_img)

                # Save preview for sidebar
                previews.append({
                    "page":    page_num + 1,
                    "index":   img_counter,
                    "b64":     pil_to_base64(pil_img),
                    "caption": caption,
                })

                # Store caption as searchable document
                chunks.append(Document(
                    page_content=(
                        f"[IMAGE on page {page_num + 1}, "
                        f"image {img_counter}]\n"
                        f"Description: {caption}"
                    ),
                    metadata={
                        "source":  pdf_path,
                        "page":    page_num + 1,
                        "type":    "image",
                        "img_idx": img_counter,
                    }
                ))

            except Exception as e:
                st.warning(f"Skipped an image on page {page_num + 1}: {e}")

    doc.close()
    progress_placeholder.empty()
    return chunks, previews


# ═══════════════════════════════════════════════
#  VECTORSTORE + QA CHAIN BUILDERS
# ═══════════════════════════════════════════════

def build_vectorstore(all_docs: list[Document]) -> Chroma:
    """
    Embed all chunks with HuggingFace sentence-transformers
    and store them in an in-memory ChromaDB collection.
    Every chunk (text, table, image caption) becomes a vector.
    """
    embeddings = load_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="multimodal_rag_opensource",
    )
    return vectorstore

def create_retriever(vectorstore: Chroma, search_type: str = "mmr", k: int = 5) -> VectorStoreRetriever:

    """
    Create a retriever from a FAISS vector store.
    
    Parameters:
    -----------
    vectorstore : FAISS
        FAISS vector store to create retriever from
    search_type : str, optional
        Type of search to perform (default is "similarity")
        Options: "similarity", "mmr", "similarity_score_threshold"
        k : int, optional
        Number of documents to retrieve (default is 5)
    
    Returns:
    --------
    VectorStoreRetriever
        Retriever object for querying the vector store
    """
    
    retriever = vectorstore.as_retriever(search_type = search_type, search_kwargs = {"k" : k})
    
    return retriever

def initialize_llm(model: str, api_key: str, temperature: float = 0.2, max_tokens: int = 800) -> ChatGroq:

    # Get API key from parameter or environment
    if not api_key:
        raise ValueError("GROQ API Key is missing")
                
    llm = ChatGroq(model = model, api_key = api_key, temperature = temperature, max_tokens = max_tokens)
    
    return llm

def create_rag_prompt(user_prompt: str) -> ChatPromptTemplate:
    """
    Create a RAG prompt template.
    
    Parameters:
    -----------
    user_prompt : str
        Custom system prompt
    
    Returns:
    --------
    ChatPromptTemplate
        Prompt template for the RAG chain
    """

    system_prompt = """You are a helpful assistant that answers questions
about a PDF document. The document may contain text paragraphs, tables,
and images (described as captions).

Use ONLY the context provided below to answer the question.
If the context does not contain enough information, say:
"I don't have enough information in this document to answer that."

Context:
{context}

Question: {question}

Instructions:
- Answer clearly and concisely.
- If the answer involves table data, present it in a readable format.
- If the answer is based on an image description, mention that it comes from an image.
- Always mention the page number(s) where the information was found.

Answer:"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}")
        ]
    )

    return prompt

def format_docs(docs: List[Document]) -> str:

    """
    Format a list of documents into a single string.
    
    Parameters:
    -----------
    docs : List[Document]
        List of Document objects to format
    
    Returns:
    --------
    str
        Concatenated string of all document contents separated by double newlines
    """
    
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------------------------
# 9. RAG CHAIN
# -------------------------------------------------------------------

def create_rag_chain(retriever: VectorStoreRetriever, prompt: ChatPromptTemplate, llm: BaseLanguageModel):
    """
    Create a complete RAG (Retrieval-Augmented Generation) chain.
    
    Parameters:
    -----------
    retriever : VectorStoreRetriever
        Retriever for fetching relevant documents
    prompt : ChatPromptTemplate
        Prompt template for the LLM
    llm : BaseLanguageModel
        Language model for generating responses
    
    Returns:
    --------
    Runnable
        Complete RAG chain ready for invocation
    """
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ═══════════════════════════════════════════════
#  MAIN PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════

def process_pdf(pdf_path: str, ollama_model: str, include_images: bool):
    """
    Full pipeline:
    Extract → Chunk → Embed → Store → Build QA chain
    Shows a progress bar in the Streamlit UI.
    """
    progress = st.progress(0, text="Starting…")
    all_docs  = []

    # ── 1. Text ───────────────────────────────
    progress.progress(10, text="Extracting text…")
    text_chunks = extract_text_chunks(pdf_path)
    all_docs.extend(text_chunks)
    st.session_state.doc_stats["text_chunks"] = len(text_chunks)

    # ── 2. Tables ─────────────────────────────
    progress.progress(30, text="Extracting tables…")
    table_chunks = extract_table_chunks(pdf_path)
    all_docs.extend(table_chunks)
    st.session_state.doc_stats["table_chunks"] = len(table_chunks)

    # ── 3. Images (optional) ──────────────────
    img_chunks = []
    if include_images:
        progress.progress(50, text="Captioning images with BLIP…")
        img_chunks, previews = extract_image_chunks(pdf_path)
        all_docs.extend(img_chunks)
        st.session_state.extracted_imgs = previews
    st.session_state.doc_stats["image_chunks"] = len(img_chunks)
    st.session_state.doc_stats["total_chunks"] = len(all_docs)

    # ── 4. Embed + store ──────────────────────
    progress.progress(75, text="Embedding chunks with sentence-transformers…")
    vectorstore = build_vectorstore(all_docs)
    st.session_state.vectorstore = vectorstore

    # ── 5. QA chain ───────────────────────────
    progress.progress(90, text=f"Building QA chain with {ollama_model}…")
    st.session_state.qa_chain = build_qa_chain(vectorstore, ollama_model)

    progress.progress(100, text="Done!")
    progress.empty()


# ═══════════════════════════════════════════════
#  UI — SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.header("Configuration")

    # Model info pills
    st.markdown(
        '<span class="model-pill">Embeddings: all-MiniLM-L6-v2</span>'
        '<span class="model-pill">Images: BLIP</span>'
        '<span class="model-pill">DB: ChromaDB</span>',
        unsafe_allow_html=True,
    )
    st.caption("All models run locally — no API key needed.")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    api_key = st.secrets.get("GROQ_API_KEY")
    model = st.sidebar.selectbox("🧠 Select LLM Model:", ["openai/gpt-oss-safeguard-20b", "qwen/qwen3-32b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"])
    temperature = st.sidebar.slider("🔥 Temperature:", min_value = 0.0, max_value = 1.0, value = 0.2)
    max_tokens = st.sidebar.slider("📏 Max Tokens:", min_value = 50, max_value = 2000, value = 800)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a PDF containing text, images, and/or tables.",
    )

    include_images = st.checkbox(
        "Include image analysis (BLIP)",
        value=True,
        help=(
            "Uses BLIP to describe images so they can be searched. "
            "First run downloads ~900 MB. Slower on CPU."
        ),
    )

    process_btn = st.button(
        "Process PDF",
        type="primary",
        disabled=not uploaded_file,
        use_container_width=True,
    )

    if process_btn and uploaded_file:
        # Reset state for new upload
        st.session_state.chat_history   = []
        st.session_state.extracted_imgs = []
        st.session_state.doc_stats      = {}
        st.session_state.vectorstore    = None
        st.session_state.qa_chain       = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            process_pdf(tmp_path, ollama_model, include_images)
            st.success("PDF indexed successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info(
                "If Ollama is not running, start it with:\n"
                "  ollama serve\n\n"
                "Then pull your model:\n"
                f"  ollama pull {ollama_model}"
            )
        finally:
            os.unlink(tmp_path)

    # ── Document stats ───────────────────────
    if st.session_state.doc_stats:
        st.divider()
        st.subheader("Document stats")
        s = st.session_state.doc_stats
        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{s.get("text_chunks", 0)}</div>'
            f'<div class="stat-label">Text chunks</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{s.get("table_chunks", 0)}</div>'
            f'<div class="stat-label">Table chunks</div></div>',
            unsafe_allow_html=True,
        )
        c3, c4 = st.columns(2)
        c3.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{s.get("image_chunks", 0)}</div>'
            f'<div class="stat-label">Image chunks</div></div>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{s.get("total_chunks", 0)}</div>'
            f'<div class="stat-label">Total chunks</div></div>',
            unsafe_allow_html=True,
        )

    # ── Extracted image previews ─────────────
    if st.session_state.extracted_imgs:
        st.divider()
        st.subheader(f"Extracted images ({len(st.session_state.extracted_imgs)})")
        for img_data in st.session_state.extracted_imgs:
            st.caption(
                f"Page {img_data['page']} — Image {img_data['index']}"
            )
            st.image(
                f"data:image/png;base64,{img_data['b64']}",
                use_column_width=True,
            )
            st.caption(f"BLIP caption: {img_data['caption']}")


# ═══════════════════════════════════════════════
#  UI — MAIN CHAT AREA
# ═══════════════════════════════════════════════
st.markdown(
    '<p class="main-header">🦙 Multimodal RAG — Open Source PDF Q&A</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">'
    'Powered by Ollama · HuggingFace BLIP · sentence-transformers · ChromaDB'
    '</p>',
    unsafe_allow_html=True,
)

# ── Welcome state ────────────────────────────
if st.session_state.qa_chain is None:
    st.info(
        "**How to get started:**\n\n"
        "1. Make sure Ollama is running: `ollama serve`\n"
        "2. Pull a model: `ollama pull llama3`\n"
        "3. Upload a PDF in the sidebar\n"
        "4. Click **Process PDF**\n"
        "5. Ask any question below!\n\n"
        "**All processing runs locally on your machine. No API key needed.**"
    )

# ── Render chat history ──────────────────────
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(
            f'<div class="answer-box">{turn["answer"]}</div>',
            unsafe_allow_html=True,
        )
        if turn.get("sources"):
            with st.expander(f"View sources ({len(turn['sources'])} chunks)"):
                for doc in turn["sources"]:
                    meta = doc.metadata
                    label = {"text": "Text", "table": "Table", "image": "Image"}.get(
                        meta.get("type", ""), "Chunk"
                    )
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div class="chunk-meta">{label} · Page {meta.get("page","?")}</div>'
                        f'{doc.page_content[:400]}'
                        f'{"…" if len(doc.page_content) > 400 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── Chat input ───────────────────────────────
if st.session_state.qa_chain:
    question = st.chat_input("Ask a question about your PDF…")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking… (Ollama is running locally)"):
                try:
                    result  = st.session_state.qa_chain.invoke({"query": question})
                    answer  = result["result"]
                    sources = result.get("source_documents", [])

                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True,
                    )

                    if sources:
                        with st.expander(f"View sources ({len(sources)} chunks)"):
                            for doc in sources:
                                meta  = doc.metadata
                                label = {
                                    "text":  "Text",
                                    "table": "Table",
                                    "image": "Image",
                                }.get(meta.get("type", ""), "Chunk")
                                st.markdown(
                                    f'<div class="chunk-card">'
                                    f'<div class="chunk-meta">'
                                    f'{label} · Page {meta.get("page","?")}'
                                    f'</div>'
                                    f'{doc.page_content[:400]}'
                                    f'{"…" if len(doc.page_content) > 400 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer":   answer,
                        "sources":  sources,
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info(
                        "Make sure Ollama is still running: `ollama serve`\n"
                        f"And the model is pulled: `ollama pull {ollama_model}`"
                    )
else:
    st.chat_input("Upload and process a PDF first…", disabled=True)

# ── Clear history ────────────────────────────
if st.session_state.chat_history:
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()
