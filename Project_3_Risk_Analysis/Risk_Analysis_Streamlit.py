import os
import tempfile
from typing import List
import streamlit as st

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.base import BaseLanguageModel
from langchain_groq import ChatGroq

# -------------------------------------------------------------------
# 1. LOAD EXCEL AS UNSTRUCTURED DOCUMENTS
# -------------------------------------------------------------------

def load_excel_documents(uploaded_file) -> List[Document]:
    """
    Load documents from an Excel file using UnstructuredExcelLoader.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile or file-like object
                    Excel file to be loaded
    Returns:
    --------
    List[Document]
        List of LangChain Document objects
    """

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".xlsx") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        # Load the Excel file using the temporary file path
        loader = UnstructuredExcelLoader(temp_file_path)
        documents = loader.load()
    finally:
        # Ensure the temporary file is deleted after processing
        os.remove(temp_file_path)

    return documents

# -------------------------------------------------------------------
# 2. SPLIT DOCUMENTS
# -------------------------------------------------------------------

def split_documents_into_chunks(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:

    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Parameters:
    -----------
    documents : List[Document]
        List of LangChain Document objects to be split
    chunk_size : int, optional
        Maximum size of each chunk in characters (default is 400)
    chunk_overlap : int, optional
        Number of characters to overlap between chunks (default is 100)
    
    Returns:
    --------
    List[Document]
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# -------------------------------------------------------------------
# 3. EMBEDDINGS
# -------------------------------------------------------------------

def create_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:

    """
    Create a HuggingFace embeddings model.
    
    Parameters:
    -----------
    model_name : str, optional
        Name of the HuggingFace model to use for embeddings 
        (default is "all-MiniLM-L6-v2")
    
    Returns:
    --------
    HuggingFaceEmbeddings
        Initialized HuggingFace embeddings model
    """
    embeddings = HuggingFaceEmbeddings(model_name = model_name)
    
    return embeddings

# -------------------------------------------------------------------
# 4. VECTOR STORE
# -------------------------------------------------------------------

def create_faiss_vectorstore(documents: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:

    """
    Create a FAISS vector database from documents and embeddings.
    
    Parameters:
    -----------
    documents : List[Document]
        List of Document objects to be stored in the vector database
    embeddings : HuggingFaceEmbeddings
        Embeddings model to use for vectorizing the documents
    
    Returns:
    --------
    FAISS
        FAISS vector store containing the embedded documents
    """
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# -------------------------------------------------------------------
# 5. RETRIEVER
# -------------------------------------------------------------------

def create_retriever(vectorstore: FAISS, search_type: str = "mmr", k: int = 5) -> VectorStoreRetriever:

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

# -------------------------------------------------------------------
# 6. LLM INITIALIZATION
# -------------------------------------------------------------------

def initialize_llm(model: str, api_key: str, temperature: float = 0.2, max_tokens: int = 800) -> ChatGroq:

    """
    Initialize a ChatGroq LLM instance.
    
    Parameters:
    -----------
    model : str
        Name of the model to use 
    api_key : str
        Groq API key. 
    temperature : float
        Sampling temperature (default is 0.2)
    max_tokens : int
        Maximum tokens in response (default is 800)
    
    Returns:
    --------
    ChatGroq
        Initialized ChatGroq LLM instance
    """
    # Get API key from parameter or environment
    if not api_key:
        raise ValueError("GROQ API Key is missing")
                
    llm = ChatGroq(model = model, api_key = api_key, temperature = temperature, max_tokens = max_tokens)
    
    return llm

# -------------------------------------------------------------------
# 7. RAG PROMPT
# -------------------------------------------------------------------

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

    system_prompt = """
    User prompt:
    "{{user_prompt}}"

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    return prompt

# -------------------------------------------------------------------
# 8. DOCUMENT FORMATTER
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# 10. Streamlit UI
# -------------------------------------------------------------------
def streamlit_app():

    st.set_page_config(page_title = "Project Risk Intelligence", page_icon = "🔍", layout = "wide")
    st.title("🔍 Project Risk Document Intelligence Assistant")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    api_key = st.secrets.get("GROQ_API_KEY")
    model = st.sidebar.selectbox("🧠 Select LLM Model:", ["openai/gpt-oss-safeguard-20b", "qwen/qwen3-32b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"])
    temperature = st.sidebar.slider("🔥 Temperature:", min_value = 0.0, max_value = 1.0, value = 0.2)
    max_tokens = st.sidebar.slider("📏 Max Tokens:", min_value = 50, max_value = 2000, value = 800)
    user_prompt = st.sidebar.text_area("📝 System Prompt", placeholder = "Enter the Prompt based on your project or requirement")
    
    # File uploader for document upload in the sidebar
    uploaded_file = st.sidebar.file_uploader("📂 Upload your document", type = ["xlsx"], help = "Supported file types: .xlsx")
    
    # ✅ Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display Chat History
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("Enter your query:")
    
    # Validation checks
    if not api_key or not uploaded_file or not user_prompt:
        st.info("⬅️ Please complete configuration in sidebar.")
        st.stop()

    # Validate file type
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension != "xlsx":
        st.warning("Only .xlsx files are currently supported.")
        st.stop()

    # Processing the Document once
    if "vectorstore" not in st.session_state:
        with st.sidebar.spinner("Processing document...."):
            # Load and split documents
            documents = load_excel_documents(uploaded_file)
            chunks = split_documents_into_chunks(documents)

            # Vector store
            embeddings = create_embeddings()
            vectorstore = create_faiss_vectorstore(chunks, embeddings)
            st.session_state.vectorstore = vectorstore
        st.success("✅ Document processed successfully!")
    
    # Create Retriever
    vectorstore = st.session_state.vectorstore
    retriever = create_retriever(vectorstore)

    # Initialize GROQ LLM Model
    llm = initialize_llm(model = model, api_key = api_key, temperature = temperature, max_tokens = max_tokens)

    # RAG
    prompt = create_rag_prompt(user_prompt)
    rag_chain = create_rag_chain(retriever, prompt, llm)

    # Handle Query
    if user_query:
        # Store user message
        st.session_state["messages"].append({"role": "user","content": user_query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        result = None  

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 AI Assistance is Analyzing..."):
                try:
                    result = rag_chain.invoke(user_query)
                    st.write(result)

                    # Show retrieved docs
                    with st.expander("🔍 View Retrieved Context"):
                        docs = retriever.invoke(user_query)
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)

                except Exception as e:
                    result = f"❌ Error processing query: {str(e)}"
                    st.error(result)

        # ✅ Store assistant response safely
        st.session_state["messages"].append({"role": "assistant","content": str(result)})

# -------------------------------------------------------------------
# 12. MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    streamlit_app()