from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import streamlit as st
from io import StringIO
import tempfile
import os
import asyncio


def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found in secrets or environment variables.")
            st.stop()
        return api_key

google_api_key = get_api_key()


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vector_store = InMemoryVectorStore(embeddings)

st.title("Semantic Search Engine")
# st.write("LangChain and Google Generative AI")

async def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


def process_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        pdf = asyncio.run(load_pdf(tmp_file_path))
        return pdf
    finally:
        os.unlink(tmp_file_path)

uploaded_file = st.file_uploader(
    "Upload a PDF file", 
    type="pdf",
)

if uploaded_file is not None:
    docs = process_uploaded_pdf(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200, add_start_index=True )
    all_splits = text_splitter.split_documents(docs)

    ids = vector_store.add_documents(documents=all_splits)
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)


prompt = st.chat_input("Search document")
if prompt:
    st.info(f"User: {prompt}")
    results = vector_store.similarity_search_with_score(prompt)
    # Question: Does the Transformer allow parallelization?
    doc, score = results[0]
    st.success(f"Result: {doc.page_content}")
    st.warning(f"Score: {score:.2f}")
