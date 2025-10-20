from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def get_llm():
    """Initialize and return the Watsonx LLM instance."""
    model_id = 'ibm/granite-3-8b-instruct'
    parameters = {
        'temperature': 0.5,
        'max_new_tokens': 256
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters
    )
    return watsonx_llm

def document_loader(file):
    """Load a PDF file and return the document content."""
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(data):
    """Split documents into chunks and return the chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def watsonx_embedding():
    """Initialize and return the Watsonx embeddings model."""
    embed_params = {
        'truncate_input_tokens': 512,
        'return_options': {'input_text': True}
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id='ibm/slate-125m-english-rtrvr',
        url='https://us-south.ml.cloud.ibm.com',
        project_id='skills-network',
        params=embed_params
    )
    return watsonx_embedding

def vector_database(chunks):
    """Create a Chroma vector store from document chunks."""
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    return vectordb

def retriever(file):
    """Create a retriever from a PDF file."""
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

def retriever_qa(file, query):
    """Run a question-answering chain on a PDF file and query."""
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True
    )
    response = qa.invoke({"query": query})
    return response['result']

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question-Answering Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port=7861, share=True)
