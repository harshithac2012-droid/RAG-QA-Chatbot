import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile
import hashlib
import shutil

# --- API Key Configuration ---
# BEST PRACTICE: Use st.secrets or environment variables for API Key
# The Streamlit environment automatically detects the GOOGLE_API_KEY environment variable.
# If running locally, make sure GOOGLE_API_KEY is set in your terminal or a .env file.
# You can also use st.secrets.
# For simplicity in this direct code, we'll configure based on environment or user input
# but it's crucial to handle the key securely.

# Check for API Key in environment or Streamlit secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Fallback for Streamlit Cloud deployment or if key isn't in environment
if not GOOGLE_API_KEY and "gemini_api_key" in st.secrets:
    GOOGLE_API_KEY = st.secrets["gemini_api_key"]

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        st.session_state['api_ready'] = True
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        st.session_state['api_ready'] = False
else:
    st.session_state['api_ready'] = False
    st.warning("Please set the GOOGLE_API_KEY environment variable or use st.secrets.")


# --- LLM and Embedding Functions ---

@st.cache_resource
def get_llm():
    """Initializes and caches the Language Model."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

@st.cache_resource
def get_embeddings():
    """Initializes and caches the Embeddings Model."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Core RAG Functions ---

def load_pdf(file_path):
    """Loads text from a PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_text(data):
    """Splits documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(data)

def get_vector_store(chunks, file_hash):
    """Creates a Chroma vector store for the document chunks, using a temp directory."""
    temp_dir = os.path.join(tempfile.gettempdir(), f"pdfqa_{file_hash}")
    
    # Simple check to see if the store might exist (though st.session_state is better)
    # For robust temp dir handling, we ensure it's clean if we are reprocessing.
    if os.path.exists(temp_dir):
        # We delete the previous version to ensure fresh data if the file hash is the same
        # but a previous session was interrupted.
        shutil.rmtree(temp_dir, ignore_errors=True) 

    vector_store = Chroma.from_documents(chunks, get_embeddings(), persist_directory=temp_dir)
    return vector_store

@st.cache_resource(hash_funcs={Chroma: lambda _: None})
def process_pdf(uploaded_file, file_hash):
    """Handles the full PDF processing pipeline using st.cache_resource."""
    
    # 1. Write the uploaded file to a temporary disk location
    temp_filepath = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # 2. Load, Split, and Embed
        docs = load_pdf(temp_filepath)
        if not docs:
            st.error("Empty PDF - no text found.")
            return None
            
        chunks = split_text(docs)
        
        # 3. Create Vector Store
        vector_store = get_vector_store(chunks, file_hash)
        
        # Clean up the temporary file (not the vector store directory)
        os.remove(temp_filepath)
        
        return vector_store, len(chunks)
        
    except Exception as e:
        # Clean up the temporary file on error
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        st.error(f"❌ Processing Error: {str(e)}")
        return None

# --- Main Streamlit Application ---

st.set_page_config(page_title="PDF Q&A - Gemini Powered", layout="wide")
st.title("📄 PDF Q&A - Gemini Powered")
st.markdown("Upload a PDF and ask questions. All processing happens locally (or within the Streamlit environment).")

# --- State Management ---
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'file_hash' not in st.session_state:
    st.session_state['file_hash'] = None

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload PDF")
    
    uploaded_file = st.file_uploader(
        "📁 Choose a PDF file", 
        type="pdf", 
        accept_multiple_files=False, 
        key="pdf_uploader"
    )
    
    # Simple Reset button for fresh start
    if st.button("Clear Cache/Reset"):
        st.session_state['vector_store'] = None
        st.session_state['file_hash'] = None
        st.cache_resource.clear()
        st.rerun()

# --- Main Content ---

if st.session_state.get('api_ready'):
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.read()).hexdigest()[:8]
        uploaded_file.seek(0) # Reset file pointer after hashing
        
        # Check if file is new or needs reprocessing
        if file_hash != st.session_state['file_hash'] or st.session_state['vector_store'] is None:
            with st.spinner(f"Processing PDF: {uploaded_file.name}..."):
                result = process_pdf(uploaded_file, file_hash)
                
                if result:
                    st.session_state['vector_store'], num_chunks = result
                    st.session_state['file_hash'] = file_hash
                    st.success(f"✅ PDF Processed! ({num_chunks} chunks indexed)")
                else:
                    st.session_state['vector_store'] = None
                    st.session_state['file_hash'] = None
                    st.error("Could not process PDF. See error details above.")
        else:
            st.info(f"Using cached vectors for **{uploaded_file.name}**.")

        # --- Q&A Section ---
        if st.session_state['vector_store']:
            st.header("2. Ask a Question")
            question = st.text_area(
                "❓ Enter your question here:",
                placeholder="What is the main topic or what are the key findings?",
                key="question_input"
            )

            if st.button("🔍 Get Answer", key="submit_button"):
                if question.strip():
                    with st.spinner("Thinking..."):
                        try:
                            vector_store = st.session_state['vector_store']
                            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                            
                            prompt = PromptTemplate(
                                input_variables=["context", "question"],
                                template="Answer from context only. If unknown, say 'Not in document.'\nContext: {context}\nQuestion: {question}\nAnswer:"
                            )
                            
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=get_llm(), 
                                chain_type="stuff", 
                                retriever=retriever,
                                return_source_documents=True, 
                                chain_type_kwargs={"prompt": prompt}
                            )
                            
                            result = qa_chain.invoke({"query": question})
                            
                            answer = result["result"]
                            sources = [f"Page {doc.metadata.get('page', 'Unknown') + 1}" for doc in result["source_documents"]]
                            
                            st.subheader("💬 Answer")
                            st.markdown(answer)
                            st.info(f"**Sources:** {', '.join(sorted(list(set(sources))))}")
                            
                        except Exception as e:
                            st.error(f"❌ Question Error: {str(e)}")
                else:
                    st.warning("Please enter a question.")
            
    else:
        st.info("Please upload a PDF file in the sidebar to begin.")

        
