import streamlit as st
import asyncio
import os
import json
import requests
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

groq_api_key = st.secrets["GROQ_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]
github_token = st.secrets["GITHUB_TOKEN"]  # 🔑 Add this to your Streamlit secrets
github_repo = "aashiq16/RAG_pipe"   # change this
github_file_path = "qa_log.json"          # file inside repo

# ---------------- Logging function ----------------
def log_to_github(question, answer):
    url = f"https://api.github.com/repos/{github_repo}/contents/{github_file_path}"

    headers = {"Authorization": f"token {github_token}"}

    # Get current file content
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        sha = data["sha"]
        try:
            content = json.loads(requests.get(data["download_url"]).text)
        except Exception:
            content = []
    else:
        sha = None
        content = []

    # Append new log
    content.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer
    })

    # Encode content
    new_content = json.dumps(content, indent=2).encode("utf-8")

    # Commit back to GitHub
    res = requests.put(
        url,
        headers=headers,
        json={
            "message": f"Log Q&A {datetime.now().isoformat()}",
            "content": new_content.decode("utf-8").encode("ascii", "ignore").decode(),
            "sha": sha,
        },
    )

    if res.status_code not in [200, 201]:
        st.error(f"⚠️ Failed to log Q&A: {res.json()}")

# ---------------- Streamlit UI ----------------
st.sidebar.title("Settings")
link = st.sidebar.text_input("Enter a webpage link:", "https://docs.smith.langchain.com/")

st.sidebar.markdown(
    """
    **How to use this app:**
    1. Enter the webpage link above.  
    2. Type your question in the text box on the main page.  
    3. The bot will answer based only on the content of that webpage.  
    4. Expand *Document Similarity Search* to see related text snippets.  
    """
)

# Only load vectors when link changes
if link and ("vector" not in st.session_state or st.session_state.get("current_link") != link):
    st.session_state.current_link = link
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )
    st.session_state.loader = WebBaseLoader(link)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Groq BOT InstaContext")
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
    response = retriever_chain.invoke({"input": user_question})
    st.write(response["answer"])

    # Log Q&A to GitHub
    log_to_github(user_question, response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
