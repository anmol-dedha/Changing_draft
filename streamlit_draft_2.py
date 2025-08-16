import os
import json
import requests
import streamlit as st

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------
# Config / Constants
# -----------------------
st.set_page_config(page_title="AnnaData - किसानों की सेवा में", page_icon="🌱")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Load OPENROUTER key from Streamlit secrets (set in Streamlit Cloud or .streamlit/secrets.toml)
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-r1-0528:free"  # keep your model here

# -----------------------
# Utilities: vectorstore build & retrieval
# -----------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore_from_file(file_path):
    """Load file, split into chunks, embed, and return FAISS vectorstore."""
    # choose loader
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_path.lower().endswith(".csv"):
        loader = CSVLoader(file_path)
        docs = loader.load()
    else:
        raise ValueError("Unsupported file type. Upload PDF or CSV.")

    # splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # embeddings (sentence-transformers local model)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # build FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def retrieve_context(vectorstore, query, top_k=3):
    """Return concatenated top_k chunks as context string."""
    if vectorstore is None:
        return ""
    docs = vectorstore.similarity_search(query, k=top_k)
    if not docs:
        return ""
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    return context

# -----------------------
# Session state init
# -----------------------
if "messages" not in st.session_state:
    # seed with a friendly system message (Optional)
    st.session_state.messages = [
        {"role": "system", "content": "You are AnnaData — a helpful assistant for farmers. Answer in simple Hindi when possible."}
    ]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# -----------------------
# Sidebar - settings & uploader
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.number_input("RAG: top K chunks", min_value=1, max_value=10, value=3, step=1)
    st.markdown("---")

    st.markdown("### 📂 Knowledge Base (PDF / CSV)")
    uploaded_file = st.file_uploader("Upload a PDF or CSV", type=["pdf", "csv"])
    if uploaded_file is not None:
        # persist file to data/
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved to `{save_path}`")

        # rebuild vectorstore and store in session
        with st.spinner("Indexing file and building vectorstore (may take a few seconds)..."):
            try:
                vs = build_vectorstore_from_file(save_path)
                st.session_state.vectorstore = vs
                if uploaded_file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(uploaded_file.name)
                st.success("Knowledge base updated and indexed ✅")
            except Exception as e:
                st.error(f"Failed to index file: {e}")

    st.markdown("#### Saved files")
    if st.session_state.uploaded_files:
        for fn in st.session_state.uploaded_files:
            st.write(f"- {fn}")
    else:
        st.write("_No files uploaded yet_")

    st.markdown("---")
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = [st.session_state.messages[0]] if st.session_state.messages else []
        st.experimental_rerun()

    st.caption("Files persist in `data/` directory on the server. Re-indexing happens at upload.")

# -----------------------
# Main UI
# -----------------------
st.title("🧑‍🌾 AnnaData - किसानों की सेवा में")

# Show chat history
chat_placeholder = st.container()
with chat_placeholder:
    for msg in st.session_state.messages:
        # st.chat_message will style messages if available
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

# Chat input area
prompt = st.chat_input("अपना सवाल लिखें...")

if prompt:
    # append immediate user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG retrieval
    vs = st.session_state.vectorstore
    rag_context = retrieve_context(vs, prompt, top_k=top_k) if vs is not None else ""
    if rag_context:
        # short preview to user (optional)
        with st.expander("📚 Retrieved context (from knowledge base) — click to expand"):
            st.write(rag_context[:1000] + ("..." if len(rag_context) > 1000 else ""))

    # Build messages with context injected as a system message
    if rag_context:
        rag_message = {"role": "system", "content": f"Use the following knowledge from the uploaded documents to answer the user's question. If it's not relevant, do not hallucinate.\n\n{rag_context}"}
        messages_with_context = [m for m in st.session_state.messages]  # copy existing convo
        # Prepend the RAG system message so the model sees the context first
        messages_with_context.insert(0, rag_message)
    else:
        messages_with_context = [m for m in st.session_state.messages]

    # Stream response from OpenRouter (same streaming pattern as your working code)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        if not OPENROUTER_API_KEY:
            placeholder.markdown("⚠️ OPENROUTER_API_KEY not set in Streamlit secrets. Please add it to use the chat.")
        else:
            try:
                with requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": messages_with_context,
                        "stream": True,
                    },
                    stream=True,
                    timeout=120,
                ) as r:
                    # iterate streaming lines
                    for line in r.iter_lines():
                        if line and line.startswith(b"data: "):
                            data_str = line[len(b"data: "):].decode("utf-8")
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                delta = data_json["choices"][0]["delta"].get("content", "")
                                full_response += delta
                                placeholder.markdown(full_response)
                            except Exception as e:
                                # small parse errors shouldn't break stream
                                placeholder.markdown(full_response + f"\n\n⚠️ parse error: {e}")
            except Exception as e:
                placeholder.markdown(f"⚠️ Error calling OpenRouter: {e}")

        # Save assistant response into conversation
        st.session_state.messages.append({"role": "assistant", "content": full_response})
