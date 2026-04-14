"""
response_generator.py
----------------------
Generates a final response for a user query using:
- Retrieved chunks from the retriever
- Figures (base64) and tables (HTML) linked to those chunks
- Gemini Flash via llm.py for generation + visual selection

Flow:
    1. Load chunk text from extraction JSON
    2. Filter chunks by relevance score for visual collection
    3. Collect figures and tables referenced by those chunks
    4. Build a prompt with chunks + visual captions
    5. Call Gemini (via llm.py) — returns answer text + tool call
    6. Parse tool call to get only query-relevant figure/table IDs
    7. Return structured response with answer, figures (base64), tables (HTML)

Usage:
    from response_generator import ResponseGenerator

    generator = ResponseGenerator(json_path="extraction.json")
    response  = generator.generate(query="...", retriever_results=[...])

    print(response["answer"])
    for fig in response["figures"]:
        print(fig["figure_id"], fig["caption"])
    for tbl in response["tables"]:
        print(tbl["table_id"], tbl["html"])
"""

import json
import os

from llm import call_gemini, GEMINI_API_KEY
from utils import build_prompt, parse_response, SOURCE_ID_FIELDS

# ── Config ────────────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.5


# ── Utility: Load chunk text ──────────────────────────────────────────────────

def build_chunk_store(json_path: str) -> dict[str, str]:
    """
    Returns a dict mapping chunk_id -> text for all chunks in the JSON.
    Used to inject text into retriever results before sending to the LLM.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    store = {}
    for source_type, id_field in SOURCE_ID_FIELDS.items():
        for entry in data.get(source_type, []):
            for chunk in entry.get("texts", []):
                store[chunk["chunk_id"]] = chunk.get("text", "")
    return store


# ── Utility: Load figures and tables ─────────────────────────────────────────

def load_visuals_index(json_path: str) -> tuple[dict, dict]:
    """
    Returns two dicts for O(1) lookup:
        figures_index[(source_type, source_id, figure_id)] -> figure dict
        tables_index [(source_type, source_id, table_id )] -> table dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    figures_index = {}
    tables_index = {}

    for source_type, id_field in SOURCE_ID_FIELDS.items():
        for entry in data.get(source_type, []):
            sid = entry[id_field]
            for fig in entry.get("figures", []):
                key = (source_type, sid, fig["figure_id"])
                figures_index[key] = {
                    "figure_id": fig["figure_id"],
                    "caption": fig.get("caption", ""),
                    "image_base64": fig.get("image_base64", ""),
                    "source_id": sid,
                    "source_type": source_type,
                    "paper_id": sid,  # backward-compat alias
                }
            for tbl in entry.get("tables", []):
                key = (source_type, sid, tbl["table_id"])
                tables_index[key] = {
                    "table_id": tbl["table_id"],
                    "caption": tbl.get("caption", ""),
                    "html": tbl.get("html", ""),
                    "source_id": sid,
                    "source_type": source_type,
                    "paper_id": sid,  # backward-compat alias
                }

    return figures_index, tables_index


# ── Utility: Collect visuals from chunks ─────────────────────────────────────

def collect_visuals(
        chunks: list[dict],
        figures_index: dict,
        tables_index: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Collect unique figures and tables referenced by the given chunks.
    Returns (figures_list, tables_list).
    """
    seen_figures = set()
    seen_tables = set()
    figures = []
    tables = []

    for chunk in chunks:
        source_type = chunk.get("source_type", "")
        id_field = SOURCE_ID_FIELDS.get(source_type, "")
        source_id = chunk.get(id_field, "")

        for fig_id in chunk.get("referenced_figures", []):
            key = (source_type, source_id, fig_id)
            if key not in seen_figures:
                seen_figures.add(key)
                fig = figures_index.get(key)
                if fig:
                    figures.append(fig)

        for tbl_id in chunk.get("referenced_tables", []):
            key = (source_type, source_id, tbl_id)
            if key not in seen_tables:
                seen_tables.add(key)
                tbl = tables_index.get(key)
                if tbl:
                    tables.append(tbl)

    return figures, tables


# ── Main Class ────────────────────────────────────────────────────────────────

class ResponseGenerator:
    """
    Ties together chunk loading, visual collection, prompt building,
    LLM calling, and response parsing.
    """

    def __init__(self, json_path: str, gemini_api_key: str = GEMINI_API_KEY):
        self.json_path = json_path
        self.gemini_api_key = gemini_api_key
        self.chunk_store = build_chunk_store(json_path)
        self.figures_index, self.tables_index = load_visuals_index(json_path)

    def generate(self, query: str, retriever_results: list[dict]) -> dict:
        """
        Generate a response for the query.

        Args:
            query:             The user's question.
            retriever_results: List of chunk dicts from the retriever, each with:
                               chunk_id, relevance_score, source_type, referenced_figures,
                               referenced_tables, and the source-specific id field.

        Returns:
            Dict with keys: answer, figures, tables, include_visuals,
                            include_figures, include_tables, visual_reasoning.
        """
        # Inject text into all chunks
        for chunk in retriever_results:
            chunk["_text"] = self.chunk_store.get(chunk.get("chunk_id", ""), "[text not available]")

        # Filter chunks above threshold for visual collection
        visual_chunks = [
            r for r in retriever_results
            if r.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
        ]
        if not visual_chunks:
            # Fallback: use the best chunk
            visual_chunks = sorted(
                retriever_results, key=lambda x: x.get("relevance_score", 0), reverse=True
            )[:1]

        # Collect candidate visuals from high-score chunks
        candidate_figures, candidate_tables = collect_visuals(
            visual_chunks, self.figures_index, self.tables_index
        )

        # Build prompt and call LLM
        prompt = build_prompt(query, retriever_results, candidate_figures, candidate_tables)
        response = call_gemini(prompt, api_key=self.gemini_api_key)

        # Parse and return structured result
        return parse_response(response, candidate_figures, candidate_tables)


# ── Test Usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    # --- Minimal fake extraction.json for testing ---
    fake_extraction = {
        "papers": [
            {
                "paper_id": "paper_001",
                "texts": [
                    {
                        "chunk_id": "chunk_001",
                        "text": "The model was fine-tuned using LoRA on a single A100 GPU for 3 epochs with a learning rate of 1e-4.",
                    },
                    {
                        "chunk_id": "chunk_002",
                        "text": "Results showed a 12% improvement in BLEU score compared to the baseline.",
                    },
                ],
                "figures": [
                    {
                        "figure_id": "Figure 1",
                        "caption": "Training loss curve over 3 epochs.",
                        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAUA",  # fake base64
                    }
                ],
                "tables": [
                    {
                        "table_id": "Table 1",
                        "caption": "Hyperparameter settings used during fine-tuning.",
                        "html": "<table><tr><th>Param</th><th>Value</th></tr><tr><td>lr</td><td>1e-4</td></tr><tr><td>epochs</td><td>3</td></tr></table>",
                    }
                ],
            }
        ]
    }

    # --- Fake retriever results ---
    fake_retriever_results = [
        {
            "chunk_id": "chunk_001",
            "source_type": "papers",
            "paper_id": "paper_001",
            "relevance_score": 0.91,
            "referenced_figures": ["Figure 1"],
            "referenced_tables": ["Table 1"],
        },
        {
            "chunk_id": "chunk_002",
            "source_type": "papers",
            "paper_id": "paper_001",
            "relevance_score": 0.45,
            "referenced_figures": [],
            "referenced_tables": [],
        },
    ]

    # Write fake JSON to a temp file
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(fake_extraction, tmp)
        tmp_path = tmp.name

    print(f"Using temp JSON: {tmp_path}")
    print("Initialising ResponseGenerator...")

    generator = ResponseGenerator(
        json_path=tmp_path,
        gemini_api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyC6e8xt1CKOfv8sPljQN79aWZHP076LiwY"),
    )

    query = "What training methods and hyperparameters were used?"

    print(f"Running generate() for query: '{query}'\n")
    result = generator.generate(query=query, retriever_results=fake_retriever_results)

    # In your main script, you'd call print_response(result, query) from utils.py
    # Here we just confirm the keys are present
    print("Response keys    :", list(result.keys()))
    print("Answer (preview) :", result["answer"][:200])
    print("Figures returned :", len(result["figures"]))
    print("Tables  returned :", len(result["tables"]))