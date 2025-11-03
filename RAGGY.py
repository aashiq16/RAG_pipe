import streamlit as st
import asyncio
import base64
import os
import json
import requests
import time
from datetime import datetime

# LangChain & related modules
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load keys from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]
github_token = st.secrets["GITHUB_TOKEN"]
github_repo = "aashiq16/RAG_pipe"   # change this if needed
github_file_path = "qa_log.json"    # file inside repo

# ---------------- Logging function ----------------
def log_to_github(repo, path, new_entry, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers)
    sha = None
    logs = []

    if r.status_code == 200:
        sha = r.json()["sha"]
        content = base64.b64decode(r.json()["content"]).decode("utf-8")
        try:
            logs = json.loads(content)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []

    logs.append(new_entry)
    encoded_content = base64.b64encode(json.dumps(logs, indent=2).encode("utf-8")).decode("utf-8")

    data = {"message": "Append Q&A log", "content": encoded_content, "branch": "main"}
    if sha:
        data["sha"] = sha

    r = requests.put(url, headers=headers, json=data)
    if r.status_code not in (200, 201):
        st.warning(f"⚠️ Failed to log Q&A: {r.json()}")
    return r.json()


# ---------------- Embedding creation with retry + fallback ----------------
def create_vectors_with_retry(docs, gemini_key, max_chunks=20, retries=3):
    st.info("⚙️ Generating embeddings... please wait.")
    docs = docs[:max_chunks]  # limit to prevent timeout

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_key
        )
    except Exception as e:
        st.warning(f"⚠️ Gemini init failed, switching to HuggingFace: {e}")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for attempt in range(retries):
        try:
            return FAISS.from_documents(docs, embeddings)
        except GoogleGenerativeAIError as e:
            st.warning(f"⚠️ Embedding attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(5)
        except Exception as e:
            st.error(f"Unexpected error during embeddings: {e}")
            break

    # fallback
    st.warning("⚠️ Gemini failed multiple times — switching to HuggingFace embeddings.")
    fallback_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, fallback_embeddings)


# ---------------- Streamlit UI ----------------
st.sidebar.title("⚙️ Settings")
link = st.sidebar.text_input("Enter a webpage link:", "https://docs.smith.langchain.com/")

st.sidebar.markdown(
    """
    **How to use this app:**
    1. Enter the webpage link above.  
    2. Type your question below.  
    3. The bot answers using only that webpage’s content.  
    """
)

# ---------------- Load webpage & build vector index ----------------
if link and ("vectors" not in st.session_state or st.session_state.get("current_link") != link):
    st.session_state.current_link = link
    st.session_state.loader = WebBaseLoader(link)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

    st.session_state.vectors = create_vectors_with_retry(
        st.session_state.final_documents,
        gemini_key=gemini_api_key
    )

# ---------------- Chat Logic ----------------
st.title("🤖 Groq BOT InstaContext")
st.subheader("Get instant context from a webpage.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Always answer based only on the provided context."),
    ("human", """
<context>
{context}
<context>
Question: {input}
""")
])

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

user_question = st.text_input("Ask a question about the link:")

if user_question:
    with st.spinner("🤔 Thinking..."):
        response = retriever_chain.invoke({"input": user_question})
    st.write(response["answer"])

    # Log Q&A to GitHub
    log_to_github(
        repo=github_repo,
        path=github_file_path,
        new_entry={
            "timestamp": datetime.now().isoformat(),
            "question": user_question,
            "answer": response["answer"],
            "source_link": link
        },
        token=github_token
    )
