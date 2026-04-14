"""
main.py
-------
End-to-end RAG pipeline: PDF/DOCX/TXT/CSV/PPTX → Extraction → Ingestion → Retrieval → Generation.

Two entry modes:
    1. Input file path → runs MainParser first, then continues with the output JSON
    2. JSON path only  → skips parsing, starts directly from ingestion (faster, for demos)

Chunk size and overlap:
    - Each file type has its own recommended defaults (defined in CHUNK_DEFAULTS below).
    - If CHUNK_SIZE / CHUNK_OVERLAP are set to None, the file-type default is used.
    - If the file type is CSV, chunk settings are ignored (CSV parser doesn't use them).
    - MainParser's own fallback is chunk_size=3000, chunk_overlap=1000 if nothing is passed.

Usage:
    Set the config block below and run:
        python main.py
"""

import os
import time

from parser.main_parser import MainParser
from ingest import IngestSession
from retriever import RetrieverSession
from response_generator import ResponseGenerator
from utils import print_response


# ─────────────────────────────────────────────────────────────────────────────
# Per-file-type chunk defaults
# Tune these based on your content. CSV is excluded — no chunking needed.
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_DEFAULTS = {
    ".pdf":  {"chunk_size": 10000, "chunk_overlap": 3000},
    ".docx": {"chunk_size": 5000,  "chunk_overlap": 1000},
    ".txt":  {"chunk_size": 5000,  "chunk_overlap": 1000},
    ".pptx": {"chunk_size": 3000,  "chunk_overlap": 500},
    ".csv":  {"chunk_size": None,  "chunk_overlap": None},   # CSV ignores these
}


# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Input: set INPUT_FILE to a PDF/DOCX/TXT/CSV/PPTX path to run full pipeline.
# Set INPUT_FILE to None to skip parsing and use an existing JSON directly.
# INPUT_FILE = r"D:\SL_Projects\Projects\AI_Python\Qwen\Data\Papers\qwen1_technical_report.pdf"
INPUT_FILE = None
JSON_PATH  = r"D:\SL_Projects\Projects\AI_Python\Multimodal-RAG\dump\output_test_sample.json"

# Chunk settings — set to None to use the per-file-type defaults above.
# For Streamlit UI these will come from sliders; None means "use the default".
CHUNK_SIZE    = None
CHUNK_OVERLAP = None

# API keys
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "xDB2CZr31oTonsAsJip3RN4gnMHlu3KgFlSfL11z")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC6e8xt1CKOfv8sPljQN79aWZHP076LiwY")

# Query
QUERY = "Explain how the design of the Qwen tokenizer differs from tokenizers used in LLaMA and XLM-R, and why these differences improve multilingual compression efficiency."


# ─────────────────────────────────────────────────────────────────────────────
# Helper: resolve chunk settings for a given file
# ─────────────────────────────────────────────────────────────────────────────

def resolve_chunk_settings(file_path: str, chunk_size: int | None, chunk_overlap: int | None) -> tuple[int | None, int | None]:
    """
    Return the chunk_size and chunk_overlap to use for this file.

    Priority:
        1. Explicit values passed in (from UI slider or config).
        2. Per-file-type defaults from CHUNK_DEFAULTS.
        3. MainParser's own defaults (chunk_size=3000, chunk_overlap=1000)
           kick in automatically when None is passed to parser.parse().

    CSV files always get (None, None) — the CSV parser ignores these.
    """
    _, ext = os.path.splitext(file_path)
    ext    = ext.lower()

    if ext == ".csv":
        return None, None

    defaults = CHUNK_DEFAULTS.get(ext, {})

    resolved_size    = chunk_size    if chunk_size    is not None else defaults.get("chunk_size")
    resolved_overlap = chunk_overlap if chunk_overlap is not None else defaults.get("chunk_overlap")

    return resolved_size, resolved_overlap


# ─────────────────────────────────────────────────────────────────────────────
# Stage functions
# ─────────────────────────────────────────────────────────────────────────────

def stage_parse(
    input_file: str,
    json_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    """Stage 1: Parse input file and write extraction JSON."""
    print("\n" + "=" * 60)
    print("STAGE 1 — PARSING")
    print("=" * 60)

    resolved_size, resolved_overlap = resolve_chunk_settings(input_file, chunk_size, chunk_overlap)

    print(f"  Input        : {input_file}")
    print(f"  Output       : {json_path}")
    print(f"  Chunk size   : {resolved_size   or 'MainParser default (3000)'}")
    print(f"  Chunk overlap: {resolved_overlap or 'MainParser default (1000)'}")

    t0     = time.perf_counter()
    parser = MainParser()

    # Pass None values through — MainParser will use its own defaults (3000 / 1000)
    kwargs = {}
    if resolved_size    is not None: kwargs["chunk_size"]    = resolved_size
    if resolved_overlap is not None: kwargs["chunk_overlap"] = resolved_overlap

    parser.parse(input_files=input_file, output_path=json_path, **kwargs)
    print(f"  Done in {time.perf_counter() - t0:.2f}s")


def stage_ingest(json_path: str):
    """Stage 2: Embed chunks and build in-memory vector DB."""
    print("\n" + "=" * 60)
    print("STAGE 2 — INGESTION")
    print("=" * 60)
    print(f"  JSON         : {json_path}")

    t0      = time.perf_counter()
    session = IngestSession(json_path).run()
    print(f"  Docs embedded: {len(session.docs)}")
    print(f"  Done in {time.perf_counter() - t0:.2f}s")
    return session


def stage_retrieve(session, query: str, cohere_api_key: str) -> list[dict]:
    """Stage 3: Retrieve and re-rank top chunks for the query."""
    print("\n" + "=" * 60)
    print("STAGE 3 — RETRIEVAL")
    print("=" * 60)
    print(f"  Query        : {query}")

    t0        = time.perf_counter()
    retriever = RetrieverSession(ingest_session=session, cohere_api_key=cohere_api_key)
    results   = retriever.query(query)
    print(f"  Chunks found : {len(results)}")
    print(f"  Done in {time.perf_counter() - t0:.2f}s")
    return results


def stage_generate(json_path: str, query: str, results: list[dict], gemini_api_key: str) -> dict:
    """Stage 4: Generate answer with relevant figures and tables."""
    print("\n" + "=" * 60)
    print("STAGE 4 — GENERATION")
    print("=" * 60)

    t0        = time.perf_counter()
    generator = ResponseGenerator(json_path=json_path, gemini_api_key=gemini_api_key)
    response  = generator.generate(query=query, retriever_results=results)
    print(f"  Answer length: {len(response['answer'])} chars")
    print(f"  Figures      : {len(response['figures'])}")
    print(f"  Tables       : {len(response['tables'])}")
    print(f"  Done in {time.perf_counter() - t0:.2f}s")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    json_path: str,
    query: str,
    cohere_api_key: str,
    gemini_api_key: str,
    input_file: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """
    Run the full pipeline and return the response dict.
    Called by main() below and later by the Streamlit UI.

    Args:
        json_path:      Path to extraction JSON (output if parsing, input if skipping).
        query:          User's question.
        cohere_api_key: Cohere API key for retrieval.
        gemini_api_key: Gemini API key for generation.
        input_file:     Path to source file (PDF/DOCX/TXT/CSV/PPTX). None = skip parsing.
        chunk_size:     Override chunk size. None = use per-file-type default.
        chunk_overlap:  Override chunk overlap. None = use per-file-type default.

    Returns:
        Response dict with keys: answer, figures, tables, include_visuals, visual_reasoning.
    """
    pipeline_start = time.perf_counter()

    # Stage 1 — Parse (skipped if no input file given)
    if input_file:
        stage_parse(input_file, json_path, chunk_size, chunk_overlap)
    else:
        print(f"\nSkipping Stage 1 — using existing JSON: {json_path}")

    # Stage 2 — Ingest
    session = stage_ingest(json_path)

    # Stage 3 — Retrieve
    results = stage_retrieve(session, query, cohere_api_key)

    # Stage 4 — Generate
    response = stage_generate(json_path, query, results, gemini_api_key)

    print(f"\nTotal pipeline time: {time.perf_counter() - pipeline_start:.2f}s")
    return response


def main():
    response = run_pipeline(
        json_path=JSON_PATH,
        query=QUERY,
        cohere_api_key=COHERE_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        input_file=INPUT_FILE,       # set to None to skip parsing
        chunk_size=CHUNK_SIZE,       # None = use CHUNK_DEFAULTS for the file type
        chunk_overlap=CHUNK_OVERLAP, # None = use CHUNK_DEFAULTS for the file type
    )
    print_response(response, QUERY)


if __name__ == "__main__":
    main()