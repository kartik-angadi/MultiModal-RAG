
"""
ui.py
-----
Streamlit UI for the Multimodal RAG pipeline.
 
Two-step flow:
    Step 1 — Setup & Index:
        - Upload files (PDF/DOCX/TXT/CSV/PPTX) OR upload an extracted JSON
        - If files uploaded: parse → save JSON → ingest into vector DB
        - If JSON uploaded: ingest directly (skip parsing)
        - Output JSON path shown only when files are uploaded (not needed for JSON mode)
        - Chunk size/overlap sliders shown only for non-CSV file uploads
 
    Step 2 — Query:
        - Unlocked only after indexing is complete
        - User types a question and clicks Ask
        - Shows answer, figures (rendered from base64), tables (rendered from HTML)
 
Run:
    streamlit run ui.py
"""

# ── Warning suppression ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import base64
import contextlib
import io
import os
import tempfile
import time

import streamlit as st

from main import stage_parse, stage_ingest, stage_retrieve, stage_generate, CHUNK_DEFAULTS


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
 
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e8e8e8; }
 
    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3d;
    }
 
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4a9eff;
        margin-bottom: 0.5rem;
    }
 
    .step-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #2a2f3d;
        background: #161b27;
        border: 1px solid #2a2f3d;
        border-radius: 4px;
        padding: 0.4rem 0.8rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
 
    .step-header.active { color: #4a9eff; border-color: #4a9eff; }
    .step-header.done   { color: #3ddc97; border-color: #3ddc97; }
 
    .answer-box {
        background-color: #161b27;
        border: 1px solid #2a2f3d;
        border-left: 3px solid #4a9eff;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e8e8e8;
        white-space: pre-wrap;
        font-family: 'IBM Plex Sans', sans-serif;
    }
 
    .fig-caption {
        font-size: 0.8rem;
        color: #8892a4;
        margin-top: 0.4rem;
        font-style: italic;
        line-height: 1.5;
    }
 
    .reasoning-box {
        background-color: #1a1f2e;
        border: 1px solid #2a2f3d;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #8892a4;
        font-style: italic;
    }
 
    .stage-log {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #4a9eff;
        background: #0d1117;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        margin: 0.15rem 0;
    }
 
    .indexed-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #3ddc97;
        background: #0d1f16;
        border: 1px solid #3ddc97;
        border-radius: 4px;
        padding: 0.4rem 0.8rem;
    }
 
    hr { border-color: #2a2f3d; }
 
    .table-wrapper {
        overflow-x: auto;
        background: #161b27;
        border: 1px solid #2a2f3d;
        border-radius: 6px;
        padding: 0.5rem;
    }
    .table-wrapper table { width: 100%; border-collapse: collapse; font-size: 0.85rem; color: #e8e8e8; }
    .table-wrapper th {
        background: #1e2535; color: #4a9eff;
        padding: 8px 12px; text-align: left;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem; letter-spacing: 0.05em;
        border-bottom: 1px solid #2a2f3d;
    }
    .table-wrapper td { padding: 7px 12px; border-bottom: 1px solid #1e2535; }
    .table-wrapper tr:last-child td { border-bottom: none; }
    .table-wrapper tr:hover td { background: #1a2030; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "indexed"      not in st.session_state: st.session_state.indexed      = False
if "session"      not in st.session_state: st.session_state.session      = None
if "json_path"    not in st.session_state: st.session_state.json_path    = None
if "index_info"   not in st.session_state: st.session_state.index_info   = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_TYPES = ["pdf", "docx", "txt", "csv", "pptx"]


def get_file_ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def save_uploaded_files(uploaded_files) -> list[str]:
    paths = []
    for uf in uploaded_files:
        ext = get_file_ext(uf.name)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(uf.read())
        tmp.close()
        paths.append(tmp.name)
    return paths


def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def display_base64_image(b64_string: str, caption: str, figure_id: str):
    try:
        img_bytes = base64.b64decode(b64_string)
        st.image(img_bytes, use_container_width=True)
        st.markdown(f'<p class="fig-caption"><b>{figure_id}</b> — {caption}</p>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not render {figure_id}: {e}")


def display_html_table(html: str, caption: str, table_id: str):
    st.markdown(f'<p class="section-label">{table_id}</p>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<p class="fig-caption">{caption}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="table-wrapper">{html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — API keys (always visible)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Multimodal RAG")
    st.markdown("---")

    st.markdown('<p class="section-label">API Keys</p>', unsafe_allow_html=True)
    cohere_key = st.text_input("Cohere API Key", type="password", value=os.environ.get("COHERE_API_KEY", ""))
    gemini_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

    # Show indexed status in sidebar
    if st.session_state.indexed:
        st.markdown("---")
        st.markdown(f'<div class="indexed-badge">✓ Index ready — {st.session_state.index_info}</div>', unsafe_allow_html=True)
        if st.button("Reset / Re-index", use_container_width=True):
            st.session_state.indexed    = False
            st.session_state.session    = None
            st.session_state.json_path  = None
            st.session_state.index_info = ""
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Setup & Index
# ─────────────────────────────────────────────────────────────────────────────

step1_label = "done" if st.session_state.indexed else "active"
st.markdown(f'<span class="step-header {step1_label}">Step 1 — Setup & Index</span>', unsafe_allow_html=True)

if not st.session_state.indexed:

    input_mode = st.radio(
        "input_mode",
        ["Upload files (PDF / DOCX / TXT / CSV / PPTX)", "Use extracted JSON"],
        label_visibility="collapsed",
        horizontal=True,
    )

    uploaded_files = None
    uploaded_json  = None
    chunk_size     = None
    chunk_overlap  = None
    json_output_path = None

    if input_mode == "Upload files (PDF / DOCX / TXT / CSV / PPTX)":
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
        )

        if uploaded_files:
            exts          = [get_file_ext(f.name) for f in uploaded_files]
            show_chunk_ui = any(e != ".csv" for e in exts)

            col_path, _ = st.columns([2, 1])
            with col_path:
                st.markdown('<p class="section-label">Output JSON path</p>', unsafe_allow_html=True)
                json_output_path = st.text_input(
                    "json_output_path",
                    value=os.path.join(tempfile.gettempdir(), "rag_output.json"),
                    label_visibility="collapsed",
                )

            if show_chunk_ui:
                primary_ext  = next((e for e in exts if e != ".csv"), ".pdf")
                defaults     = CHUNK_DEFAULTS.get(primary_ext, {"chunk_size": 3000, "chunk_overlap": 1000})
                default_size = defaults["chunk_size"]    or 3000
                default_ovlp = defaults["chunk_overlap"] or 1000

                st.markdown('<p class="section-label">Chunking</p>', unsafe_allow_html=True)
                st.caption(f"Defaults for {primary_ext.lstrip('.')}: size={default_size}, overlap={default_ovlp}")

                col1, col2 = st.columns(2)
                with col1:
                    chunk_size = st.slider("Chunk size", min_value=500, max_value=20000, value=default_size, step=500)
                with col2:
                    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=chunk_size // 2, value=min(default_ovlp, chunk_size // 2), step=100)

    else:
        uploaded_json = st.file_uploader("Upload extraction JSON", type=["json"])

    # Index button
    st.markdown("")
    index_btn = st.button("Index", type="primary", use_container_width=False)

    if index_btn:
        # Validate
        if not cohere_key:
            st.error("Please provide your Cohere API key in the sidebar.")
            st.stop()
        if input_mode == "Upload files (PDF / DOCX / TXT / CSV / PPTX)" and not uploaded_files:
            st.error("Please upload at least one file.")
            st.stop()
        if input_mode == "Use extracted JSON" and not uploaded_json:
            st.error("Please upload an extraction JSON file.")
            st.stop()

        status_area = st.empty()

        def log(msg: str):
            status_area.markdown(f'<div class="stage-log">▸ {msg}</div>', unsafe_allow_html=True)

        try:
            with st.spinner("Indexing..."):
                t0 = time.perf_counter()

                if input_mode == "Upload files (PDF / DOCX / TXT / CSV / PPTX)":
                    saved_paths  = save_uploaded_files(uploaded_files)
                    file_arg     = saved_paths[0] if len(saved_paths) == 1 else saved_paths
                    json_path    = json_output_path or os.path.join(tempfile.gettempdir(), "rag_output.json")

                    log("Stage 1 / 2 — Parsing documents...")
                    first_file = file_arg if isinstance(file_arg, str) else file_arg[0]
                    silent(stage_parse, first_file, json_path, chunk_size, chunk_overlap)

                    log("Stage 2 / 2 — Ingesting and embedding chunks...")
                    session = silent(stage_ingest, json_path)

                    info = f"{len(uploaded_files)} file(s) · {len(session.docs)} chunks"

                else:
                    tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                    tmp_json.write(uploaded_json.read())
                    tmp_json.close()
                    json_path = tmp_json.name

                    log("Stage 1 / 1 — Ingesting and embedding chunks...")
                    session = silent(stage_ingest, json_path)

                    info = f"JSON · {len(session.docs)} chunks"

                elapsed = time.perf_counter() - t0
                log(f"✓ Indexed in {elapsed:.1f}s")

                # Store in session state
                st.session_state.indexed    = True
                st.session_state.session    = session
                st.session_state.json_path  = json_path
                st.session_state.index_info = info

        except Exception as e:
            st.error(f"Indexing error: {e}")
            st.stop()

        st.rerun()

else:
    st.success(f"✓ Index ready — {st.session_state.index_info}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Query  (only shown after indexing)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
step2_label = "active" if st.session_state.indexed else ""
st.markdown(f'<span class="step-header {step2_label}">Step 2 — Ask a question</span>', unsafe_allow_html=True)

if not st.session_state.indexed:
    st.caption("Complete Step 1 to unlock querying.")
else:
    query   = st.text_area("query", placeholder="e.g. Explain the architecture of the model...", height=80, label_visibility="collapsed")
    ask_btn = st.button("Ask", type="primary", use_container_width=False)

    if ask_btn:
        if not query.strip():
            st.error("Please enter a question.")
            st.stop()
        if not gemini_key:
            st.error("Please provide your Gemini API key in the sidebar.")
            st.stop()

        status_area = st.empty()

        def log(msg: str):
            status_area.markdown(f'<div class="stage-log">▸ {msg}</div>', unsafe_allow_html=True)

        try:
            with st.spinner("Generating answer..."):
                t0 = time.perf_counter()

                log("Stage 1 / 2 — Retrieving relevant chunks...")
                results = silent(stage_retrieve, st.session_state.session, query, cohere_key)

                log("Stage 2 / 2 — Generating answer...")
                response = silent(stage_generate, st.session_state.json_path, query, results, gemini_key)

                elapsed = time.perf_counter() - t0
                log(f"✓ Done in {elapsed:.1f}s")

        except Exception as e:
            st.error(f"Query error: {e}")
            st.stop()

        status_area.empty()
        st.markdown("---")

        # Answer
        st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)

        # Figures
        if response.get("figures"):
            st.markdown("---")
            st.markdown('<p class="section-label">Figures</p>', unsafe_allow_html=True)
            cols = st.columns(min(len(response["figures"]), 2))
            for i, fig in enumerate(response["figures"]):
                with cols[i % 2]:
                    display_base64_image(fig["image_base64"], fig["caption"], fig["figure_id"])

        # Tables
        if response.get("tables"):
            st.markdown("---")
            st.markdown('<p class="section-label">Tables</p>', unsafe_allow_html=True)
            for tbl in response["tables"]:
                display_html_table(tbl["html"], tbl["caption"], tbl["table_id"])
                st.markdown("<br>", unsafe_allow_html=True)

        # Visual reasoning
        if response.get("visual_reasoning"):
            with st.expander("Visual selection reasoning"):
                st.markdown(f'<div class="reasoning-box">{response["visual_reasoning"]}</div>', unsafe_allow_html=True)

        # Stats
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks retrieved", len(results))
        c2.metric("Figures shown",    len(response.get("figures", [])))
        c3.metric("Tables shown",     len(response.get("tables",  [])))
        c4.metric("Time (s)",         f"{elapsed:.1f}")