# Multimodal RAG: Documents That Know Their Own Figures and Tables
 
A Retrieval-Augmented Generation pipeline built for documents where text, figures, and tables aren't independent — where a paragraph says "as shown in Figure 3" and Figure 3's caption is about the paragraph's topic. Instead of flattening everything into text, this project keeps text, figures, and tables as separate, addressable objects and **links them to each other exactly where they reference one another** — so retrieval and generation can reason across modalities instead of guessing.
 
> 🎥 **Demo:** _add your demo link / screen recording here_
> 📄 **Parser internals:** [`src/parser/README.md`](src/parser/README.md)
 
---
 
## Overview
 
Point the pipeline at a PDF, DOCX, PPTX, CSV, or TXT file and it will:
 
1. Parse the document into text chunks, extracted figures (as images), and extracted tables (as structured markup) — see [`src/parser/README.md`](src/parser/README.md) for exactly how.
2. Detect every place a chunk mentions a figure or table (`"Figure 3"`, `"Table 2"`, etc.) and record that link in **both directions**: the chunk knows what it references, and the figure/table knows every chunk that mentions it.
3. Embed everything into a vector store and make it queryable through hybrid (dense + sparse) retrieval with reranking.
4. On a query, retrieve the most relevant chunks, pull in the figures/tables *linked to those specific chunks*, and hand both to an LLM — which writes the answer and explicitly selects which of the candidate visuals are actually worth returning.
The output isn't just a text answer — it's an answer accompanied by the exact chart or table it's talking about, pulled from the source document, not regenerated or guessed at.
 
---
 
## Why this is different
 
- **Visuals stay visuals.** Figures are kept as real images (base64) and tables as structured HTML/markdown, extracted directly from the document's own layout — not reconstructed or paraphrased from text. When the answer needs a chart, you get the actual chart.
- **Structure is preserved, not flattened.** Multi-column tables, merged cells, and nested headers stay intact as structured markup, so the LLM (and the user) sees the table the way it was actually laid out in the source document.
- **Every link is explicit and chunk-level.** Each text chunk carries its own `referenced_figures` / `referenced_tables`, pointing to specific, individually retrievable objects — not a vague page-level association. This means retrieval can tell exactly *which* figure a given chunk is about, even on pages with several visuals.
- **Retrieval pulls in only what's relevant.** Because links are chunk-specific rather than page- or document-wide, a retrieved chunk brings along only the visuals it actually discusses — nothing extra just because it happened to share a page with them.
- **The LLM curates, it doesn't guess.** At generation time, the model is handed the candidate visuals linked to the retrieved chunks and explicitly selects (via tool calling) which ones are worth returning for that specific query — so the final answer includes the right figure or table, not every visual that was loosely nearby.
The result: ask a question about a specific chart, and you get that exact chart back, correctly identified, alongside a properly formatted answer — grounded in the source document's real structure rather than an approximation of it.
 
---
 
## How it works
 
```
                 ┌──────────────┐
  PDF/DOCX/PPTX  │  1. PARSER   │  → structured JSON
  CSV/TXT   ───▶ │  (per type)  │     text chunks + figures (base64) +
                 └──────┬───────┘     tables (markdown/HTML), cross-linked
                        │
                        ▼
                 ┌──────────────┐
                 │  2. INGEST   │  → embeds text/figure-caption/table-caption
                 │              │     docs into a Chroma vector store
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  3. RETRIEVE │  → hybrid retrieval:
                 │              │     dense (MMR) + sparse (BM25) → rerank
                 └──────┬───────┘
                        │
                        ▼
                 ┌───────────────────┐
                 │  4. GENERATE       │ → gathers figures/tables linked to the
                 │                    │    retrieved chunks, prompts the LLM,
                 │                    │    which tool-calls which visuals to
                 └────────────────────┘    keep, and returns answer + visuals
```
 
---
 
## Features
 
- **Bidirectional cross-linking** between text chunks and the figures/tables they mention, built at parse time from the document's actual structure — not inferred later from flattened text.
- **Multi-format ingestion**: PDF, DOCX, PPTX, CSV, TXT, each with a dedicated parser.
- **Hybrid retrieval**: dense (MMR) + sparse (BM25), fused and reranked for higher precision than either alone.
- **Visual-aware generation**: the LLM receives the answer-relevant text *and* the candidate visuals linked to it, then explicitly selects (via tool calling) which figures/tables are worth returning — so you get only the visuals relevant to the actual question, not every image linked to a chunk.
- **Streamlit UI** for uploading documents, tuning chunk size/overlap per file type, indexing, and querying — no notebook required.
- **Skip-parsing mode**: point the pipeline at an already-parsed JSON to re-index and query instantly, useful for demos and iteration.
---

## Getting started

### 1. Clone and install

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

PDF/PPTX parsing with image and table extraction depends on `unstructured`'s document-processing extras, which in turn need system-level tools (Poppler for PDF rendering, Tesseract for OCR). See [`src/parser/README.md`](src/parser/README.md) for OS-level setup notes.

### 2. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# .env
COHERE_API_KEY=your-cohere-api-key
GEMINI_API_KEY=your-gemini-api-key
```

- **`COHERE_API_KEY`** — used for embeddings and reranking during ingestion/retrieval.
- **`GEMINI_API_KEY`** — used for answer generation and visual selection.

`.env` is loaded automatically wherever the pipeline runs (see `.env.example` for the exact keys expected). Keep `.env` out of version control — only commit `.env.example`.

### 3. Run the pipeline

**Option A — Streamlit UI (recommended):**
```bash
streamlit run src/ui.py
```
Upload a PDF/DOCX/PPTX/CSV/TXT (or a pre-parsed JSON), index it, then ask questions in the query box.

**Option B — Script:**
Edit the config block at the top of `src/main.py` (`INPUT_FILE`, `JSON_PATH`, `QUERY`), then:
```bash
python src/main.py
```

---

## Project structure

```
.
├── README.md                    # You are here
├── requirements.txt
├── .env.example                 # Template for COHERE_API_KEY / GEMINI_API_KEY
└── src/
    ├── main.py                  # End-to-end pipeline orchestration (parse → ingest → retrieve → generate)
    ├── ui.py                    # Streamlit app
    ├── ingest.py                # Parsed JSON → embedded Chroma vector store
    ├── retriever.py             # Hybrid (dense + sparse) retrieval + rerank
    ├── response_generator.py    # Gathers linked visuals, builds the prompt, calls the LLM
    ├── llm.py                   # LLM client + visual-selection tool definition
    ├── utils.py                 # Prompt building, response parsing, display helpers
    └── parser/
        ├── main_parser.py       # Dispatches to the right extractor by file extension
        ├── parse_pdf.py         # PDF → text/figures/tables, cross-linked
        ├── parse_pptx.py        # PPTX → text/figures/tables per slide, cross-linked
        ├── parse_docx.py        # DOCX → chunked text
        ├── parse_txt.py         # TXT → chunked text
        ├── parse_csv.py         # CSV → row-level records
        └── README.md            # Deep-dive into the parsing/extraction logic
```


