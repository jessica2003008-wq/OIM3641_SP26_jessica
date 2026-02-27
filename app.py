import os
from pathlib import Path

import streamlit as st

DATA_DIR = Path("data")
API_ENV = "GOOGLE_API_KEY"


def require(condition: bool, msg: str) -> None:
    """Small guard: show a helpful message and stop the app."""
    if not condition:
        st.error(msg)
        st.stop()


@st.cache_resource(show_spinner=True)
def build_engine():
    """Build the RAG engine (cached). Fast-fail on errors."""
    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
        from llama_index.core.settings import Settings
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        from llama_index.llms.google_genai import GoogleGenAI

        Settings.llm = GoogleGenAI(model="gemini-1.5-flash")
        Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

        docs = SimpleDirectoryReader(input_dir=str(DATA_DIR)).load_data()
        require(len(docs) > 0, "No documents were loaded from data/. Put at least one readable file in data/.")

        index = VectorStoreIndex.from_documents(docs)
        return index.as_query_engine(similarity_top_k=4)

    except Exception:
        # friendly fast-fail message (no wall of stack trace)
        raise RuntimeError(
            "RAG engine failed to initialize. Check: API key, installed packages, and that your file is readable."
        )


def main():
    st.title("RAG Chatbot (Robust MVP)")

    # 1) config + data checks (minimal)
    api_key = os.getenv(API_ENV, "").strip()
    require(bool(api_key), f"Missing API key: set environment variable `{API_ENV}`.")

    require(DATA_DIR.exists() and DATA_DIR.is_dir(), "Missing data/ directory. Create a data/ folder in your project.")

    files = [p for p in DATA_DIR.iterdir() if p.is_file()]
    require(len(files) > 0, "data/ is empty. Add at least one file (e.g., the Babson handbook PDF).")

    # 2) build RAG engine (cached) with fast-fail
    try:
        engine = build_engine()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # 3) simple UI + small input guard
    q = st.text_input("Ask a question about the document(s):")
    if st.button("Send"):
        q = (q or "").strip()
        require(len(q) > 0, "Please type a question first.")

        try:
            st.write(engine.query(q))
        except Exception:
            st.error("Query failed. Please try again (or verify your API key / model settings).")


if __name__ == "__main__":
    main()