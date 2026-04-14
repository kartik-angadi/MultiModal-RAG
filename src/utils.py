"""
utils.py
--------
Pure utility functions for the RAG pipeline.
No file I/O, no API calls — only prompt building, response parsing, and display.

Functions:
    build_prompt()    — builds the LLM prompt string from chunks and visuals
    parse_response()  — extracts answer + selected visuals from raw Gemini response
    print_response()  — pretty-prints the final response dict

Usage:
    from utils import build_prompt, parse_response, print_response
"""

# ── Config (shared with response_generator) ───────────────────────────────────

SOURCE_ID_FIELDS = {
    "papers": "paper_id",
    "documents": "document_id",
    "txt_files": "txt_file_id",
    "csv_files": "csv_file_id",
    "presentations": "ppt_id",
}


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(
        query: str,
        chunks: list[dict],
        figures: list[dict],
        tables: list[dict],
) -> str:
    """
    Build the full prompt string to send to the LLM.

    Args:
        query:   The user's question.
        chunks:  List of chunk dicts, each must have a "_text" key injected.
        figures: List of candidate figure dicts (from visuals index).
        tables:  List of candidate table dicts (from visuals index).

    Returns:
        A single prompt string ready to pass to call_gemini().
    """
    lines = [
        "You are a precise research assistant. Answer the user's query using ONLY "
        "the information in the provided context below.",
        "",
        "STRICT RULES:",
        "- Base your answer solely on the TEXT CHUNKS provided.",
        "- If the context does not fully cover the query, state what is missing.",
        "- Do not speculate or add information not present in the context.",
        # "- Cite chunk_ids, figure_ids, or table_ids inline when referencing evidence.",
        "",
        "=" * 60,
        f"USER QUERY:\n{query}",
        "=" * 60,
        "",
        "CONTEXT — TEXT CHUNKS:",
        "",
    ]

    for i, chunk in enumerate(chunks, 1):
        source_type = chunk.get("source_type", "")
        id_field = SOURCE_ID_FIELDS.get(source_type, "source_id")
        source = chunk.get(id_field, "unknown")
        score = chunk.get("relevance_score", 0)
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("_text", "[text not available]")
        lines += [
            f"[{chunk_id} | {source} | {source_type} | score: {score:.3f}]",
            text,
            "",
        ]

    if figures:
        lines += ["─" * 60, "AVAILABLE FIGURES (evaluate each for relevance):", ""]
        for fig in figures:
            lines += [
                f"  figure_id : {fig['figure_id']}",
                f"  source    : {fig['source_id']} ({fig['source_type']})",
                f"  caption   : {fig['caption']}",
                "",
            ]

    if tables:
        lines += ["─" * 60, "AVAILABLE TABLES (evaluate each for relevance):", ""]
        for tbl in tables:
            lines += [
                f"  table_id  : {tbl['table_id']}",
                f"  source    : {tbl['source_id']} ({tbl['source_type']})",
                f"  caption   : {tbl['caption']}",
                f"  content   : {tbl['html']}",
                "",
            ]

    lines += [
        "=" * 60,
        "INSTRUCTIONS:",
        "1. Write a thorough answer to the query using ONLY the text chunks above.",
        # "2. Cite evidence using chunk_ids, figure_ids, or table_ids inline.",
        "2. Call select_relevant_visuals() with:",
        "   - relevant_figure_ids: only figures whose content directly supports the answer.",
        "   - relevant_table_ids:  only tables whose content directly supports the answer.",
        "   - Return empty lists if no visuals are relevant.",
    ]

    return "\n".join(lines)


# ── Response Parser ───────────────────────────────────────────────────────────

def parse_response(
        raw_response,
        candidate_figures: list[dict],
        candidate_tables: list[dict],
) -> dict:
    """
    Extract answer text and tool call results from the raw Gemini response.

    Args:
        raw_response:      The response object returned by call_gemini().
        candidate_figures: All figures passed into the prompt.
        candidate_tables:  All tables passed into the prompt.

    Returns:
        Dict with keys: answer, figures, tables, include_visuals,
                        include_figures, include_tables, visual_reasoning.
    """
    answer_parts = []
    selected_fig_ids = set()
    selected_tbl_ids = set()
    reasoning = ""

    for part in raw_response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            answer_parts.append(part.text)

        if hasattr(part, "function_call") and part.function_call:
            if part.function_call.name == "select_relevant_visuals":
                args = part.function_call.args
                selected_fig_ids = set(args.get("relevant_figure_ids", []))
                selected_tbl_ids = set(args.get("relevant_table_ids", []))
                reasoning = str(args.get("reasoning", ""))

    selected_figures = [f for f in candidate_figures if f["figure_id"] in selected_fig_ids]
    selected_tables = [t for t in candidate_tables if t["table_id"] in selected_tbl_ids]

    return {
        "answer": "\n".join(answer_parts).strip(),
        "figures": selected_figures,
        "tables": selected_tables,
        "include_visuals": bool(selected_figures or selected_tables),
        "include_figures": bool(selected_figures),
        "include_tables": bool(selected_tables),
        "visual_reasoning": reasoning,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_response(response: dict, query: str) -> None:
    """
    Pretty-print the response dict returned by ResponseGenerator.generate().

    Args:
        response: Dict returned by ResponseGenerator.generate().
        query:    The original user query string.
    """
    print("\n" + "=" * 70)
    print(f"QUERY  : {query}")
    print("=" * 70)

    print("\nANSWER:")
    print(response["answer"])

    if response["figures"]:
        print(f"\nFIGURES ({len(response['figures'])}):")
        for fig in response["figures"]:
            print(f"  [{fig['figure_id']} | {fig['paper_id']}]")
            print(f"  Caption : {fig['caption']}")
            b64_preview = fig["image_base64"][:60] + "..." if fig["image_base64"] else "(none)"
            print(f"  Base64  : {b64_preview}")

    if response["tables"]:
        print(f"\nTABLES ({len(response['tables'])}):")
        for tbl in response["tables"]:
            print(f"  [{tbl['table_id']} | {tbl['paper_id']}]")
            print(f"  Caption : {tbl['caption']}")
            html_preview = tbl["html"][:100] + "..." if tbl["html"] else "(none)"
            print(f"  HTML    : {html_preview}")

    print(f"\nVisual reasoning : {response['visual_reasoning']}")
    print("=" * 70)
