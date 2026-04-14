import os
import re
import json
import platform
from collections import Counter
from typing import List, Dict, Any, Tuple

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Tesseract (Windows) ─────────────────────────────────────────────

if platform.system() == "Windows":
    import unstructured_pytesseract

    DEFAULT_TESS = r"C:\Users\SL00088.SANDLOGIC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    unstructured_pytesseract.pytesseract.tesseract_cmd = os.environ.get(
        "TESSERACT_CMD", DEFAULT_TESS
    )


# ── Regex definitions (UNCHANGED) ────────────────────────────────────

_FIG_CAP_RE  = re.compile(r"^Figure\s+(\d+[a-zA-Z]?)[\s:\.\-–—](.*)", re.IGNORECASE | re.DOTALL)
_TBL_CAP_RE  = re.compile(r"^Table\s+(\d+[a-zA-Z]?)[\s:\.\-–—](.*)",  re.IGNORECASE | re.DOTALL)
_REF_FIG_RE  = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
_REF_TBL_RE  = re.compile(r"\bTable\s+(\d+[a-zA-Z]?)", re.IGNORECASE)

_TOC_RE      = re.compile(r"\.{4,}|\s\.\s\.\s\.")
_BARE_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_SHORT_LINE_RE = re.compile(r"^.{1,60}$")

_STRUCTURAL   = {"Image", "Table", "FigureCaption"}
_HARD_DROP    = {"Header", "Footer", "PageBreak"}
_CAP_ONLY_TYPES = {"FigureCaption"}

# ─────────────────────────────────────────────────────────────────────
# Main Extractor
# ─────────────────────────────────────────────────────────────────────

class MultimodalPDFExtractor:

    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 500):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ───────────────────────────────────────────────────

    def extract(self, pdf_paths: List[str], output_path: str) -> Dict:

        papers = []

        for idx, pdf_path in enumerate(pdf_paths):
            paper_id = f"paper_{idx + 1}"

            try:
                paper = self._process_single(pdf_path, paper_id)
                papers.append(paper)
            except Exception:
                continue

        merged = {"papers": papers}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        return merged

    # ── Single PDF pipeline ──────────────────────────────────────────

    def _process_single(self, pdf_path: str, paper_id: str) -> Dict:

        raw = self._partition(pdf_path)

        repeats = self._find_repeated_short_lines(raw)

        clean = self._filter_noise(raw, repeats)

        texts = self._extract_texts(clean, paper_id)
        figures = self._extract_figures(clean, paper_id)
        tables = self._extract_tables(clean, paper_id)

        self._cross_link(texts, figures, tables)

        return {
            "paper_id": paper_id,
            "source": pdf_path,
            "texts": texts,
            "figures": figures,
            "tables": tables,
        }

    # ── Partition PDF ─────────────────────────────────────────────────

    def _partition(self, pdf_path: str) -> List[Any]:

        kw = dict(
            filename=pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            multipage_sections=True,
        )

        try:
            return partition_pdf(**kw)
        except Exception:
            return partition_pdf(
                filename=pdf_path,
                infer_table_structure=True,
                strategy="fast",
                multipage_sections=True,
            )

    # ── Detect repeated short lines (headers/footers) ─────────────────

    @staticmethod
    def _find_repeated_short_lines(elements: List[Any]) -> set:

        line_counter: Counter = Counter()

        for el in elements:

            et = type(el).__name__
            if et in _STRUCTURAL or et in _HARD_DROP:
                continue

            text = (el.text if hasattr(el, "text") else str(el)).strip()

            if text and _SHORT_LINE_RE.match(text) and "\n" not in text:
                line_counter[text] += 1

        return {text for text, count in line_counter.items() if count >= 3}

    # ── Noise filter ─────────────────────────────────────────────────

    def _filter_noise(self, elements: List[Any], repeats: set) -> List[Any]:

        out = []

        for el in elements:

            et = type(el).__name__
            text = (el.text if hasattr(el, "text") else str(el)).strip()

            if et in _HARD_DROP:
                continue

            if et in _STRUCTURAL:
                out.append(el)
                continue

            if len(text) <= 3:
                continue

            if _BARE_PAGE_RE.match(text):
                continue

            if text in repeats:
                continue

            lines = [l.strip() for l in text.splitlines() if l.strip()]
            toc_lines = sum(1 for l in lines if _TOC_RE.search(l))

            if lines and toc_lines / len(lines) >= 0.6:
                continue

            out.append(el)

        return out

    # ── Text extraction ──────────────────────────────────────────────

    def _extract_texts(self, elements: List[Any], paper_id: str) -> List[Dict]:

        parts: List[str] = []

        for el in elements:

            et = type(el).__name__
            text = (el.text if hasattr(el, "text") else str(el)).strip()

            if et in _STRUCTURAL or not text:
                continue

            if et == "Title":
                parts.append(f"\n{text}")

            elif et == "ListItem":

                if parts and parts[-1].startswith("• "):
                    parts[-1] += f"\n• {text}"
                else:
                    parts.append(f"• {text}")

            else:

                if (
                    parts
                    and text[0].islower()
                    and parts[-1]
                    and parts[-1][-1] not in ".!?:"
                    and not parts[-1].startswith("• ")
                    and et not in ("Title",)
                ):
                    parts[-1] += " " + text
                else:
                    parts.append(text)

        corpus = "\n\n".join(parts)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        return [
            {
                "paper_id": paper_id,
                "chunk_id": f"chunk_{i}",
                "text": c.strip(),
                "referenced_figures": [],
                "referenced_tables": [],
            }
            for i, c in enumerate(splitter.split_text(corpus))
            if c.strip()
        ]

    # ── Figure extraction ─────────────────────────────────────────────

    def _extract_figures(self, elements: List[Any], paper_id: str) -> List[Dict]:

        n = len(elements)
        figures = {}
        used_idx = set()

        for i, el in enumerate(elements):

            if type(el).__name__ != "Image":
                continue

            caption, cap_idx = self._find_caption(elements, i, n, used_idx, "figure")

            if not caption:
                continue

            m = _FIG_CAP_RE.match(caption)
            fig_id = f"Figure {m.group(1)}"

            used_idx.add(cap_idx)

            if fig_id not in figures:
                figures[fig_id] = {
                    "paper_id": paper_id,
                    "figure_id": fig_id,
                    "image_base64": _b64(el),
                    "caption": caption,
                    "mentioned_in_chunks": [],
                }

        for i, el in enumerate(elements):

            if i in used_idx or type(el).__name__ != "FigureCaption":
                continue

            caption = (el.text if hasattr(el, "text") else str(el)).strip()
            m = _FIG_CAP_RE.match(caption)

            if not m:
                continue

            fig_id = f"Figure {m.group(1)}"

            if fig_id not in figures:
                figures[fig_id] = {
                    "paper_id": paper_id,
                    "figure_id": fig_id,
                    "image_base64": "",
                    "caption": caption,
                    "mentioned_in_chunks": [],
                }

        result = list(figures.values())
        result.sort(key=lambda f: _num_key(f["figure_id"]))
        return result

    # ── Table extraction ──────────────────────────────────────────────

    def _extract_tables(self, elements: List[Any], paper_id: str) -> List[Dict]:

        n = len(elements)
        tables = {}
        used_idx = set()

        for i, el in enumerate(elements):

            if type(el).__name__ != "Table":
                continue

            caption, cap_idx = self._find_caption(elements, i, n, used_idx, "table")

            if not caption:
                continue

            m = _TBL_CAP_RE.match(caption)
            tbl_id = f"Table {m.group(1)}"

            used_idx.add(cap_idx)

            html = (
                getattr(getattr(el, "metadata", None), "text_as_html", None)
                or (el.text if hasattr(el, "text") else str(el))
            )

            if tbl_id not in tables:
                tables[tbl_id] = {
                    "paper_id": paper_id,
                    "table_id": tbl_id,
                    "html": html,
                    "caption": caption,
                    "mentioned_in_chunks": [],
                }

        result = list(tables.values())
        result.sort(key=lambda t: _num_key(t["table_id"]))
        return result

    # ── Caption search (UNCHANGED) ────────────────────────────────────

    def _find_caption(
        self,
        elements: List[Any],
        anchor: int,
        n: int,
        used_idx: set,
        kind: str,
    ) -> Tuple[str, int]:

        cap_re = _FIG_CAP_RE if kind == "figure" else _TBL_CAP_RE
        structural = {"Image", "Table"}

        for direction in (
            range(anchor + 1, min(anchor + 7, n)),
            range(anchor - 1, max(anchor - 7, -1), -1),
        ):
            for j in direction:

                if j in used_idx:
                    continue

                et = type(elements[j]).__name__
                text = (elements[j].text if hasattr(elements[j], "text") else str(elements[j])).strip()

                if et in structural:
                    break

                if kind == "figure" and et == "FigureCaption":
                    if cap_re.match(text):
                        return text, j
                    used_idx.add(j)
                    continue

                if cap_re.match(text):
                    return text, j

        return "", -1

    # ── Cross linking ─────────────────────────────────────────────────

    def _cross_link(self, texts: List[Dict], figures: List[Dict], tables: List[Dict]):

        fig_idx = {f["figure_id"]: f for f in figures}
        tbl_idx = {t["table_id"]: t for t in tables}

        for chunk in texts:

            cid = chunk["chunk_id"]
            text = chunk["text"]

            for num in _REF_FIG_RE.findall(text):

                fid = f"Figure {num}"

                if fid in fig_idx:

                    if fid not in chunk["referenced_figures"]:
                        chunk["referenced_figures"].append(fid)

                    if cid not in fig_idx[fid]["mentioned_in_chunks"]:
                        fig_idx[fid]["mentioned_in_chunks"].append(cid)

            for num in _REF_TBL_RE.findall(text):

                tid = f"Table {num}"

                if tid in tbl_idx:

                    if tid not in chunk["referenced_tables"]:
                        chunk["referenced_tables"].append(tid)

                    if cid not in tbl_idx[tid]["mentioned_in_chunks"]:
                        tbl_idx[tid]["mentioned_in_chunks"].append(cid)



# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _type_counts(elements: List[Any]) -> str:
    c = Counter(type(el).__name__ for el in elements)
    return ", ".join(f"{k}={v}" for k, v in c.most_common())


def _num_key(id_str: str) -> int:
    m = re.search(r"(\d+)", id_str)
    return int(m.group(1)) if m else 9999


def _b64(el) -> str:
    try:
        return el.metadata.image_base64 or ""
    except AttributeError:
        return ""


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":

    pdf_paths = [
        r"D:\SL_Projects\Projects\AI_Python\Qwen\Data\Papers\qwen1_technical_report.pdf"
    ]

    output_path = r"D:\SL_Projects\Projects\AI_Python\Multimodal-RAG\dump\qwen1_paper_dump2.json"

    extractor = MultimodalPDFExtractor(chunk_size=10000, chunk_overlap=3000)
    extractor.extract(pdf_paths, output_path)