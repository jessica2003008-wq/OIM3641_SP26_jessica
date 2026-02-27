import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# --- APP SETUP ---
st.set_page_config(page_title="Babson Handbook RAG System", layout="centered")

# Load .env reliably (from this file's folder)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DATA_DIR = Path("data")
API_ENV = "GEMINI_API_KEY"

Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def require(condition: bool, msg: str) -> None:
    """Small guard: show helpful message and stop."""
    if not condition:
        st.error(msg)
        st.stop()


def validate_config() -> None:
    """Config validation (API key + data folder + files)."""
    require(os.getenv(API_ENV, "").strip(), f"❌ Missing API key: set `{API_ENV}` (in .env or environment).")
    require(DATA_DIR.exists() and DATA_DIR.is_dir(), f"❌ Data directory not found: `{DATA_DIR}`")
    files = [p for p in DATA_DIR.iterdir() if p.is_file()]
    require(len(files) > 0, f"❌ No files found in `{DATA_DIR}`. Add at least one document (e.g., PDF).")


@st.cache_resource(show_spinner=True)
def get_query_engine():
    """Cached RAG engine with fast-fail handling."""
    try:
        documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
        require(len(documents) > 0, f"❌ No readable documents loaded from `{DATA_DIR}`.")
        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine()
    except Exception as e:
        # fast fail with helpful message
        raise RuntimeError(f"Failed to initialize RAG engine: {type(e).__name__}: {e}") from e


# --- STREAMLIT UI ---
st.title("Bare Bones RAG Chatbot")

# 1) Validate before doing anything heavy
validate_config()

# 2) Build engine (cached) with fast-fail
try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(str(e))
    st.stop()

prompt = st.chat_input("Ask me a question...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                response = query_engine.query(prompt)
                bot_response = response.response if hasattr(response, "response") else str(response)
            except Exception as e:
                bot_response = f"❌ Query failed: {type(e).__name__}: {e}"
        st.markdown(bot_response)

    # ✅ fix: store actual bot_response, not the string "bot_response"
    st.session_state.messages.append({"role": "assistant", "content": bot_response})