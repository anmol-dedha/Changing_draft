# app.py
import os
import json
import requests
import streamlit as st

# import RAG helper you created
import rag

# ✅ Load API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

MODEL = "deepseek/deepseek-r1-0528:free"

st.set_page_config(page_title="AnnaData Draft 1", page_icon="🌱")
st.title("🧑‍🌾 AnnaData - किसानों की सेवा में")

# ----------------------
# file & persistence dirs
# ----------------------
DATA_DIR = "data"
VSTORE_DIR = "vectorstore"   # local persist folder for FAISS
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VSTORE_DIR, exist_ok=True)

# ----------------------
# load or build vectorstore at startup (if exists)
# ----------------------
if "vectorstore" not in st.session_state:
    # try to load persisted vectorstore (if previously saved)
    vs = rag.load_vectorstore(VSTORE_DIR)
    if vs is None:
        # if no persisted vectorstore, try to build from files in data/ (if any)
        vs = rag.build_vectorstore_from_dir(DATA_DIR)
        if vs is not None:
            # persist it so next start is faster
            rag.save_vectorstore(vs, VSTORE_DIR)
    st.session_state.vectorstore = vs

# ----------------------
# session state for chat
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    # discover files already in data/
    st.session_state.uploaded_files = [
        f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".csv"))
    ]

# ----------------------
# Sidebar: uploader + controls
# ----------------------
with st.sidebar:
    st.header("⚙️ Knowledge base & settings")

    top_k = st.number_input("RAG: top K chunks", min_value=1, max_value=10, value=3)

    st.markdown("### 📂 Upload PDF / CSV (saved to server)")
    uploaded_file = st.file_uploader("Upload a PDF or CSV", type=["pdf", "csv"])
    if uploaded_file is not None:
        # save to data/ folder
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved to `{save_path}`")

        # reindex all files or just the new one (here: rebuild from data/)
        with st.spinner("Indexing uploaded files..."):
            try:
                vs = rag.build_vectorstore_from_dir(DATA_DIR)
                if vs is not None:
                    rag.save_vectorstore(vs, VSTORE_DIR)
                    st.session_state.vectorstore = vs
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                    st.success("Knowledge base indexed and saved ✅")
                else:
                    st.info("No indexable files found after upload.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.markdown("#### Saved files")
    if st.session_state.uploaded_files:
        for fn in st.session_state.uploaded_files:
            st.write(f"- {fn}")
    else:
        st.write("_No files in data/_")

    st.markdown("---")
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.experimental_rerun()

# ----------------------
# Show existing messages
# ----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------
# Chat input handling
# ----------------------
if prompt := st.chat_input("अपना सवाल लिखें..."):
    # append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use RAG to get context (if vectorstore exists)
    vs = st.session_state.vectorstore
    rag_context = rag.retrieve_context(vs, prompt, top_k=top_k) if vs is not None else ""
    if rag_context:
        with st.expander("📚 Retrieved context (from knowledge base) — click to expand"):
            # show a short preview
            st.write(rag_context[:1500] + ("..." if len(rag_context) > 1500 else ""))

    # Build messages with RAG context as a system message
    messages_with_context = st.session_state.messages.copy()
    if rag_context:
        rag_msg = {
            "role": "system",
            "content": (
                "Use the following context from uploaded documents to answer the user's question. "
                "If the context is not relevant, do not hallucinate. Context:\n\n" + rag_context
            ),
        }
        # put the rag system message at the front so the model sees it early
        messages_with_context.insert(0, rag_msg)

    # Stream response from OpenRouter (same pattern you used)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        if not OPENROUTER_API_KEY:
            placeholder.markdown("⚠️ OPENROUTER_API_KEY not set in Streamlit secrets.")
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
                                # show partial response + parse note
                                placeholder.markdown(full_response + f"\n\n⚠️ parse error: {e}")
            except Exception as e:
                placeholder.markdown(f"⚠️ Error calling OpenRouter: {e}")

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
