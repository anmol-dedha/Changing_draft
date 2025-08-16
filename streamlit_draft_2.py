import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils.openrouter_api import stream_chat  # ✅ your existing streaming function
import os

# -------------------------
# Setup
# -------------------------
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Farmer AI Assistant", page_icon="🌾")

st.title("🌾 Farmer AI Assistant")
st.write("Ask me anything about government schemes, crops, or agriculture. The answers will come from both AI + your PDFs!")

# User query
user_query = st.text_input("💬 Enter your question:")

if user_query:
    # 1. Retrieve context from PDFs
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    st.subheader("🔎 Retrieved Context")
    with st.expander("Show context"):
        st.write(context)

    # 2. Ask LLM with RAG
    prompt = f"""
    You are an agriculture expert assistant.
    Use the following context to answer the user query.
    If the context is not relevant, say you don't know.

    Context:
    {context}

    Question: {user_query}
    """

    st.subheader("🤖 Assistant Response")
    response = stream_chat(prompt)  # your OpenRouter streaming
    st.write(response)
