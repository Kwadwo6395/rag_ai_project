# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: app.py
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.embeddings import Embedder
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store
from rag.retrieval import HybridRetriever
from rag.decomposer import QueryDecomposer
from rag.generator import build_text_generator
from rag.pipeline import RAGPipeline

# Make `.env` win over a stale GEMINI_* / LLM_* exported in the shell.
load_dotenv(override=True)
st.set_page_config(page_title="Academic City RAG", layout="wide")

_APP_DIR = Path(__file__).resolve().parent
_LOGO_PATH = _APP_DIR / "assets" / "academic_city_logo.png"

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
_DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# st.secrets[...] triggers a noisy st.error() when no secrets.toml exists; cache this once.
_SECRETS_FILE_MISSING: bool | None = None


def _inject_theme_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --acity-red: #c8102e;
            --acity-black: #111111;
            --acity-white: #ffffff;
            --acity-soft: #f7f7f8;
            --acity-border: #e2e2e2;
          }
          .stApp {
            background: linear-gradient(180deg, #fff 0%, #fafafa 100%);
          }
          .header-card {
            border: 1px solid var(--acity-border);
            border-left: 8px solid var(--acity-red);
            background: var(--acity-white);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
          }
          .grade-card {
            border: 1px solid var(--acity-border);
            border-top: 4px solid var(--acity-red);
            background: var(--acity-soft);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
          }
          .grade-title {
            color: var(--acity-black);
            font-weight: 700;
            margin-bottom: 0.35rem;
          }
          .grade-sub {
            color: #333;
            font-size: 0.92rem;
          }
          .metric-chip {
            display: inline-block;
            background: var(--acity-black);
            color: var(--acity-white);
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.8rem;
            margin-right: 0.4rem;
          }
          .stButton > button {
            border-radius: 10px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _env_or_secrets(key: str) -> str | None:
    """Prefer process env (.env via load_dotenv); otherwise Streamlit secrets if a file exists."""
    global _SECRETS_FILE_MISSING
    v = os.environ.get(key, "").strip()
    if v:
        return v
    if _SECRETS_FILE_MISSING:
        return None
    try:
        from streamlit.runtime.secrets import secrets_singleton
    except ImportError:
        return None
    if _SECRETS_FILE_MISSING is None:
        # Loads secrets.toml without printing "No secrets found" when the file is absent.
        if not secrets_singleton.load_if_toml_exists():
            _SECRETS_FILE_MISSING = True
            return None
        _SECRETS_FILE_MISSING = False
    try:
        out = str(secrets_singleton[key]).strip()
        return out or None
    except KeyError:
        return None


def _resolved_gemini_model() -> str:
    m = _env_or_secrets("GEMINI_MODEL")
    return m or _DEFAULT_GEMINI_MODEL


def _resolved_groq_model() -> str:
    m = _env_or_secrets("GROQ_MODEL")
    return m or _DEFAULT_GROQ_MODEL


def _provider_from_env() -> str:
    p = (os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    return "groq" if p == "groq" else "gemini"


@st.cache_resource(show_spinner="Loading models and indices...")
def load_pipeline(llm_provider: str, llm_model: str, use_llm_decomposition: bool):
    emb = Embedder()
    vs = VectorStore.load("index/store")
    bm = BM25Store.load("index/bm25.pkl")
    retriever = HybridRetriever(emb, vs, bm)
    gemini_key = _env_or_secrets("GEMINI_API_KEY")
    groq_key = _env_or_secrets("GROQ_API_KEY")
    generator = build_text_generator(
        provider=llm_provider,
        model_id=llm_model,
        gemini_api_key=gemini_key,
        groq_api_key=groq_key,
    )
    decomposer = QueryDecomposer(
        generator=generator,
        use_llm_decomposition=use_llm_decomposition,
    )
    return RAGPipeline(retriever, decomposer, generator)


def main():
    _inject_theme_css()
    logo_col, title_col = st.columns([1, 5], vertical_alignment="center")
    with logo_col:
        if _LOGO_PATH.is_file():
            st.image(str(_LOGO_PATH), use_column_width=True)
    with title_col:
        st.markdown(
            """
            <div class="header-card">
              <h1 style="margin:0;color:#111;">Academic City RAG Assistant</h1>
              <p style="margin:0.25rem 0 0 0;color:#333;">
                Ghana elections + 2025 budget statement
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    default_idx = 1 if _provider_from_env() == "groq" else 0
    llm_provider = st.sidebar.radio(
        "LLM provider",
        ("gemini", "groq"),
        index=default_idx,
        horizontal=True,
        help="Gemini: set GEMINI_API_KEY. Groq: set GROQ_API_KEY (https://console.groq.com). "
        "Override default with LLM_PROVIDER in `.env`.",
    )
    llm_model = (
        _resolved_groq_model() if llm_provider == "groq" else _resolved_gemini_model()
    )
    st.sidebar.caption(f"Active: **{llm_provider}** · model `{llm_model}`")

    _env_llm = os.environ.get("RAG_LLM_DECOMPOSE", "").strip().lower() in (
        "1", "true", "yes",
    )
    use_llm_decompose = st.sidebar.checkbox(
        "LLM query planning (slower: 2× API calls per question)",
        value=_env_llm,
        help="Off (default): instant keyword routing + **one** LLM call for the answer. "
        "Turn on for harder multi-corpus splits.",
    )
    st.sidebar.caption(
        "Default: **1** LLM call per question (answer only). "
        "Gemini free tier is tight; Groq is often faster for chat."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Grading View")
    st.sidebar.caption("Use this panel during assessment/demo.")
    st.sidebar.markdown(
        "- [ ] Accurate answer with citations\n"
        "- [ ] Sources are relevant\n"
        "- [ ] Decomposition is reasonable\n"
        "- [ ] Prompt and trace are visible\n"
        "- [ ] Log path generated",
    )

    try:
        pipeline = load_pipeline(llm_provider, llm_model, use_llm_decompose)
    except RuntimeError as e:
        st.error(
            f"Startup error: {e}\n\n"
            "For **Gemini**, set `GEMINI_API_KEY`. For **Groq**, set `GROQ_API_KEY` in `.env` "
            "or Streamlit secrets."
        )
        return

    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.write(text)

    
    st.markdown(
        """
        <div class="grade-card">
          <div class="grade-title">Query Evaluation Guidance</div>
          <div class="grade-sub">
             Election query , Budget query, Cross-corpus query, Politics query.
            <p>Confirms citations , retrieves context quality, and pipeline trace.<p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.chat_input("Ask a question...")
    if query:
        st.session_state.history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            status_line = st.empty()
            try:
                with st.spinner(
                    "Generating answer…"
                    if not use_llm_decompose
                    else "Thinking… (LLM plan + answer; slower)"
                ):
                    result = pipeline.answer(
                        query,
                        st.session_state.history[:-1],
                        progress=status_line.caption,
                    )
            except Exception as e:
                status_line.empty()
                st.error(f"Could not complete your request:\n\n{e}")
                st.session_state.history.append(
                    ("assistant", f"Error: {e}")
                )
                st.stop()

            status_line.empty()
            st.write(result["answer"])
            with st.expander("Retrieved context"):
                for i, c in enumerate(result["chunks"]):
                    loc = c.get("page", c.get("row", "?"))
                    st.markdown(
                        f"**[{i+1}] {c.get('source', '?')}** "
                        f"(loc {loc}, rrf={c.get('rrf_score', 0):.3f})"
                    )
                    st.write(c["text"])
            with st.expander("Decomposition"):
                st.json(result["decomposition"])
            with st.expander("Full prompt sent to LLM"):
                st.code(result["prompt"])
            with st.expander("Pipeline trace"):
                st.json(result["trace"])
            st.caption(f"Log: `{result['log_path']}`")
        st.session_state.history.append(("assistant", result["answer"]))


if __name__ == "__main__":
    main()
