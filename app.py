"""
AI Research Assistant — Streamlit UI
Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-ready   { background: #d4edda; color: #155724; }
    .badge-waiting { background: #fff3cd; color: #856404; }
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #4361ee;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .chat-user     { background: #e8f4fd; border-radius: 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
    .chat-assistant{ background: #f0fdf4; border-radius: 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
def _init_state() -> None:
    if "pipeline" not in st.session_state:
        from pipeline import ResearchPipeline
        st.session_state.pipeline = ResearchPipeline()

    defaults = {
        "messages": [],          # {role, content, citations}
        "index_loaded": False,
        "doc_count": 0,
        "chunk_count": 0,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


_init_state()
pipeline = st.session_state.pipeline


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Document Ingestion")

    api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Get a free key at aistudio.google.com. Used only within this session.",
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        import config as _cfg
        _cfg.GOOGLE_API_KEY = api_key

    st.divider()

    upload_tab, dir_tab = st.tabs(["Upload PDFs", "Use Existing Index"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Drop research papers here",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if st.button("📥 Index Uploaded Files", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one file first.")
            elif not api_key:
                st.error("Please enter your Google API key.")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    for f in uploaded_files:
                        dest = Path(tmpdir) / f.name
                        dest.write_bytes(f.read())

                    with st.spinner("Indexing documents…"):
                        try:
                            n_chunks = pipeline.ingest(tmpdir)
                            st.session_state.index_loaded = True
                            st.session_state.doc_count = len(uploaded_files)
                            st.session_state.chunk_count = n_chunks
                            st.success(f"✅ Indexed {len(uploaded_files)} file(s) → {n_chunks} chunks")
                        except Exception as exc:
                            st.error(f"Ingestion failed: {exc}")

    with dir_tab:
        import config as _cfg
        st.caption(f"Index path: `{_cfg.VECTOR_STORE_PATH}`")

        if st.button("🔄 Load Saved Index", use_container_width=True):
            with st.spinner("Loading index…"):
                ok = pipeline.load_existing_index()
                if ok:
                    st.session_state.index_loaded = True
                    st.session_state.chunk_count = pipeline.chunk_count
                    st.success(f"✅ Loaded {pipeline.chunk_count} chunks from disk")
                else:
                    st.error("No saved index found. Please upload documents first.")

    st.divider()

    # Status badge
    if st.session_state.index_loaded:
        st.markdown(
            f'<span class="status-badge badge-ready">🟢 Ready · {st.session_state.chunk_count} chunks</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge badge-waiting">🟡 Waiting for documents</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Summary button
    if st.button("📋 Summarise All Documents", use_container_width=True):
        if not st.session_state.index_loaded:
            st.warning("Index documents first.")
        else:
            with st.spinner("Generating summary…"):
                summary = pipeline.summarise()
            st.markdown("### Document Summary")
            st.markdown(summary)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        pipeline.reset_memory()
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about your uploaded research papers.</div>',
    unsafe_allow_html=True,
)

# Render existing messages
for msg in st.session_state.messages:
    css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
    prefix = "**You:** " if msg["role"] == "user" else "**Assistant:** "
    st.markdown(
        f'<div class="{css_class}">{prefix}{msg["content"]}</div>',
        unsafe_allow_html=True,
    )
    if msg.get("citations"):
        with st.expander("📎 Sources", expanded=False):
            st.markdown(msg["citations"])

st.divider()

# Input row
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Ask a question",
        key="user_input",
        placeholder="e.g. What evaluation metrics are used across these papers?",
        label_visibility="collapsed",
    )
with col_btn:
    send = st.button("Send ➤", use_container_width=True)

if send and user_input.strip():
    if not st.session_state.index_loaded:
        st.warning("Please index documents before asking questions.")
    elif not api_key:
        st.error("Enter your Google API key in the sidebar.")
    else:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        with st.spinner("Thinking…"):
            try:
                result = pipeline.query(user_input.strip())
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                })
            except Exception as exc:
                st.error(f"Error: {exc}")

        st.rerun()
