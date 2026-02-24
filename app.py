import os

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI


DATA_DIR = "data"
PERSIST_DIR = "storage"


MAX_DOCS_FOR_FIRST_BUILD = 200


def init_settings() -> None:
    """Configure llama-index to use Gemini (not OpenAI)."""
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)


@st.cache_resource(show_spinner=False)
def build_or_load_index() -> VectorStoreIndex:
    """
    Build the index once (calls Gemini embeddings), persist to ./storage,
    then load from disk on later runs to avoid re-embedding and 429 errors.
    """
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Missing '{DATA_DIR}/' directory. Put the handbook file inside it.")

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if not documents:
        raise FileNotFoundError(f"No files found in '{DATA_DIR}/'. Add the Babson handbook file there.")

    documents = documents[:MAX_DOCS_FOR_FIRST_BUILD]

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


def rag_answer(index: VectorStoreIndex, user_query: str) -> str:
    """Run a RAG query against the handbook index and return the final answer."""
    query_engine = index.as_query_engine(similarity_top_k=4)
    response = query_engine.query(user_query)
    return str(response)


def main() -> None:
    st.set_page_config(page_title="Babson Handbook RAG Chatbot", page_icon="📘")
    st.title("📘 Babson Student Handbook — RAG Chatbot (Gemini + LlamaIndex)")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY. Add it to a .env file in your project root.")
        st.stop()

    os.environ["GEMINI_API_KEY"] = api_key
    init_settings()

    with st.spinner("Loading handbook + building/loading index..."):
        try:
            index = build_or_load_index()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.error(
                    "Still hitting Gemini 429 (quota/rate limit).\n\n"
                    "Do this:\n"
                    "1) Ensure your key has quota / billing enabled, OR\n"
                    "2) Lower MAX_DOCS_FOR_FIRST_BUILD to 20, rerun.\n\n"
                    "Once you successfully build the index once, ./storage will appear and future runs won't re-embed."
                )
                st.stop()
            raise

    st.success("Index ready ✅ (cached locally in ./storage after first successful build)")
    st.caption(
        "Tip: Keep only the handbook file in ./data to reduce weird answers. "
        f"Currently indexing first {MAX_DOCS_FOR_FIRST_BUILD} document chunks for the initial build."
    )

    user_query = st.text_input(
        "Ask a question about the Babson student handbook",
        placeholder="e.g., Can I get credit for courses taken somewhere else?",
    )
    submit = st.button("Submit")

    if submit and user_query.strip():
        with st.spinner("Thinking..."):
            answer = rag_answer(index, user_query.strip())
        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
