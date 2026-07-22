# Parser: Document → Cross-Linked Structured JSON

This module turns a raw document (PDF, DOCX, PPTX, CSV, or TXT) into a single structured JSON file containing text chunks, extracted figures, and extracted tables — with **text chunks and visuals cross-referenced by ID** so downstream retrieval and generation can pull in exactly the right figure or table for a given piece of text.

This README covers the extraction logic. For how the resulting JSON is embedded, retrieved, and used for generation, see the [main README](../../README.md).

---

## Entry point: `main_parser.py`

`MainParser.parse()` is the single entry point for every file type:

```python
from parser.main_parser import MainParser

parser = MainParser()
parser.parse(
    input_files="paper.pdf",              # str or list[str]
    output_path="output/extraction.json",
    chunk_size=10000,
    chunk_overlap=3000,
)
```

It looks at the extension of the **first** file in `input_files` and dispatches to the matching extractor:

| Extension | Extractor | Figures/tables extracted? |
|---|---|---|
| `.pdf` | `MultimodalPDFExtractor` (`parse_pdf.py`) | ✅ |
| `.pptx` | `PptxExtractor` (`parse_pptx.py`) | ✅ |
| `.docx` | `DocxExtractor` (`parse_docx.py`) | ❌ (text only, for now) |
| `.txt` | `TxtExtractor` (`parse_txt.py`) | ❌ (text only) |
| `.csv` | `CSVExtractor` (`parse_csv.py`) | N/A (row-based, not chunked) |

> ⚠️ All files passed in one call are assumed to be the **same file type** — the extension of `input_files[0]` decides the extractor for the whole batch.

Every extractor writes one merged JSON file and also returns the same dict in memory.

---

## PDF extraction (`parse_pdf.py`) — the core of the multimodal pipeline

This is the most involved extractor, since PDFs are where figure/table/text cross-referencing matters most (research papers, reports).

### Step by step

1. **Partition** — `unstructured.partition.pdf.partition_pdf()` runs with `strategy="hi_res"`, table structure inference, and image extraction (`extract_image_block_to_payload=True`, so images come back as base64 directly on the element). If hi-res partitioning fails (e.g. missing OCR/layout deps), it falls back to `strategy="fast"` without image/table extraction.
2. **Header/footer detection** — any short line (≤60 chars) that repeats 3+ times across the document is flagged as a running header/footer and dropped.
3. **Noise filtering** — drops `Header`/`Footer`/`PageBreak` elements, near-empty text, bare page numbers, and lines that look like a table of contents (≥60% of lines matching a dot-leader pattern like `..... 12`).
4. **Text extraction** — remaining elements are stitched into a single corpus:
   - `Title` elements start a new paragraph.
   - `ListItem` elements are grouped into bullet blocks.
   - Otherwise, a text element continuing a lowercase sentence fragment from the previous block is merged into it (handles PDF text broken mid-sentence by layout).
   - The corpus is then split with `RecursiveCharacterTextSplitter` (`chunk_size`/`chunk_overlap` as passed in) into chunks, each assigned a `chunk_id`.
5. **Figure extraction** — every `Image` element is matched to a nearby caption (see [caption matching](#caption-matching) below). If a caption matching `Figure <n>` is found, the image's base64 payload + caption are stored under `figure_id = "Figure <n>"`. Captions found without a matching image (e.g. image extraction failed) are still recorded with an empty `image_base64`.
6. **Table extraction** — every `Table` element is matched to a nearby `Table <n>` caption the same way. The table's `text_as_html` metadata (from `infer_table_structure=True`) is stored as `html`, falling back to plain text if HTML isn't available.
7. **Cross-linking** — every text chunk is scanned with regex for `Figure <n>` / `Fig. <n>` and `Table <n>` mentions. For each match that corresponds to a real extracted figure/table:
   - the figure/table ID is added to the chunk's `referenced_figures` / `referenced_tables`, and
   - the chunk's ID is added to that figure/table's `mentioned_in_chunks`.

   This is what makes the link bidirectional: a chunk knows what it references, and each figure/table knows every chunk that talks about it.

### Caption matching

`_find_caption()` searches up to 6 elements forward, then up to 6 elements backward, from the image/table element, stopping early if it hits another structural element (`Image`/`Table`) first. It accepts the first element whose text matches `^Figure \d+` / `^Table \d+` (case-insensitive, allowing letters like `Figure 3a`). This handles both captions placed directly below a figure and captions placed above it.

### Output schema (PDF)

```json
{
  "papers": [
    {
      "paper_id": "paper_1",
      "source": "path/to/file.pdf",
      "texts": [
        {
          "paper_id": "paper_1",
          "chunk_id": "chunk_0",
          "text": "…paragraph text mentioning Figure 1 and Table 2…",
          "referenced_figures": ["Figure 1"],
          "referenced_tables": ["Table 2"]
        }
      ],
      "figures": [
        {
          "paper_id": "paper_1",
          "figure_id": "Figure 1",
          "image_base64": "iVBORw0KGgoAAAANS…",
          "caption": "Figure 1: Model architecture overview.",
          "mentioned_in_chunks": ["chunk_0", "chunk_3"]
        }
      ],
      "tables": [
        {
          "paper_id": "paper_1",
          "table_id": "Table 2",
          "html": "<table><tr><th>Param</th><th>Value</th></tr>…</table>",
          "caption": "Table 2: Hyperparameter settings.",
          "mentioned_in_chunks": ["chunk_0"]
        }
      ]
    }
  ]
}
```

Multiple input PDFs produce multiple entries in `"papers"`, each with its own `paper_id` (`paper_1`, `paper_2`, …). A file that raises an exception during processing is skipped, not fatal to the batch.

---

## PPTX extraction (`parse_pptx.py`)

Same core idea as the PDF extractor, adapted for slide-based documents:

- `unstructured.partition.pptx.partition_pptx()` runs with table-structure inference and image-to-base64 extraction.
- Elements are grouped by `slide_number` (from element metadata) before anything else, since figures/tables should only be matched to captions and text **on the same slide**.
- Per slide: text is stitched (titles/bullets handled the same way as the PDF extractor, minus the lowercase-continuation merge), then chunked with `RecursiveCharacterTextSplitter`.
- Figures and tables are extracted and caption-matched **within each slide's element list**, so a caption on slide 4 can never be matched to an image on slide 7.
- Cross-linking works identically to the PDF extractor — bidirectional `referenced_figures`/`referenced_tables` ↔ `mentioned_in_chunks`.

### Output schema (PPTX)

```json
{
  "presentations": [
    {
      "ppt_id": "ppt_1",
      "source": "path/to/deck.pptx",
      "texts": [
        {
          "ppt_id": "ppt_1",
          "chunk_id": "chunk_3_0",
          "slide_number": 3,
          "text": "…",
          "referenced_figures": ["Figure 1"],
          "referenced_tables": []
        }
      ],
      "figures": [
        {
          "ppt_id": "ppt_1",
          "figure_id": "Figure 1",
          "slide_number": 3,
          "image_base64": "…",
          "caption": "Figure 1: Q3 revenue by region.",
          "mentioned_in_chunks": ["chunk_3_0"]
        }
      ],
      "tables": [ /* same shape, with table_id + html */ ]
    }
  ]
}
```

---

## DOCX extraction (`parse_docx.py`)

Simpler, text-only pipeline: `unstructured.partition.docx.partition_docx()` extracts all elements, their text is concatenated, and the result is chunked with `RecursiveCharacterTextSplitter`. No figure/table extraction or cross-linking yet — this is a natural extension point if DOCX-embedded images/tables become a priority.

```json
{
  "documents": [
    {
      "document_id": "document_1",
      "source": "path/to/file.docx",
      "texts": [
        {"document_id": "document_1", "chunk_id": "document_1_chunk_1", "text": "…"}
      ]
    }
  ]
}
```

---

## TXT extraction (`parse_txt.py`)

Reads the file as UTF-8 and chunks it directly with `RecursiveCharacterTextSplitter` — no partitioning step needed.

```json
{
  "txt_files": [
    {
      "txt_id": "txt_1",
      "source": "path/to/file.txt",
      "texts": [
        {"txt_id": "txt_1", "chunk_id": "txt_1_chunk_1", "text": "…"}
      ]
    }
  ]
}
```

---

## CSV extraction (`parse_csv.py`)

CSVs aren't chunked by character count — instead, every row becomes its own record via `pandas`, preserving the original column structure as a dict:

```json
{
  "csv_files": [
    {
      "csv_id": "csv_1",
      "source": "path/to/file.csv",
      "rows": [
        {"csv_id": "csv_1", "row_id": "csv_1_row_1", "data": {"col_a": "value", "col_b": 42}}
      ]
    }
  ]
}
```

---

## Regex reference

Used by both the PDF and PPTX extractors for caption detection and in-text reference detection:

| Pattern | Purpose |
|---|---|
| `^Figure\s+(\d+[a-zA-Z]?)[\s:.\-–—](.*)` | Matches a caption starting with "Figure 3:", "Figure 3a -", etc. |
| `^Table\s+(\d+[a-zA-Z]?)[\s:.\-–—](.*)` | Same, for table captions. |
| `\b(?:Figure\|Fig\.?)\s+(\d+[a-zA-Z]?)` | Matches in-text mentions like "see Figure 3" or "Fig. 3a". |
| `\bTable\s+(\d+[a-zA-Z]?)` | Matches in-text mentions like "as shown in Table 2". |

These are intentionally simple and English-caption-oriented — documents using different captioning conventions (e.g. "Fig 3" without a period, or non-English labels) won't cross-link automatically and are a good place to extend the regex set.

---

## Dependencies & setup notes

- **`unstructured`** does the heavy lifting for PDF/PPTX/DOCX partitioning. For PDFs/PPTXs with `strategy="hi_res"` (layout + image extraction), it needs system-level dependencies — Poppler (PDF rendering) and Tesseract OCR. Install the `unstructured[all-docs]` extra plus the corresponding system packages for your OS.
- On **Windows**, `parse_pdf.py` and `parse_pptx.py` point `unstructured_pytesseract` at a hardcoded Tesseract path by default. Override it with the `TESSERACT_CMD` environment variable instead of editing the source:
  ```bash
  set TESSERACT_CMD=C:\path\to\tesseract.exe
  ```
- **`langchain-text-splitters`** provides `RecursiveCharacterTextSplitter`, used by every extractor except CSV.
- **`pandas`** is only needed for CSV extraction.

---

## Choosing chunk size / overlap

Each extractor exposes `chunk_size` / `chunk_overlap` on its constructor. `main.py` (in the parent pipeline) defines suggested per-file-type defaults:

| File type | chunk_size | chunk_overlap |
|---|---|---|
| PDF | 10000 | 3000 |
| DOCX | 5000 | 1000 |
| TXT | 5000 | 1000 |
| PPTX | 3000 | 500 |
| CSV | — (row-based, ignored) | — |

Larger chunks (PDF) keep more surrounding context around each figure/table reference, which helps the generation step reason about *why* a visual matters. Smaller chunks (PPTX) match the naturally short, self-contained nature of slide content.

---

## Extending to a new file type

1. Create `parse_<type>.py` with an extractor class exposing `.extract(file_paths, output_path) -> dict`.
2. Follow the existing output convention: a top-level key naming the source type (e.g. `"spreadsheets"`), a list of per-file entries, each with a unique ID field.
3. If the format can contain figures/tables referenced from text, reuse the caption-matching (`_find_caption`) and cross-linking (`_cross_link`) pattern from `parse_pdf.py` / `parse_pptx.py` for consistency.
4. Register the new extension in `MainParser.parse()` in `main_parser.py`.
5. Add the new source type's ID field to `SOURCE_ID_FIELDS` in `ingest.py` and `utils.py` so it flows through ingestion and generation automatically.