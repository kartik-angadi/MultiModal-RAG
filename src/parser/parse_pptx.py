import os
import re
import json
import platform
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

from unstructured.partition.pptx import partition_pptx
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------------------------
# Tesseract (Windows)
# ------------------------------------------------
if platform.system() == "Windows":
    import unstructured_pytesseract

    _DEFAULT = r"C:\Users\SL00088.SANDLOGIC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    unstructured_pytesseract.pytesseract.tesseract_cmd = os.environ.get(
        "TESSERACT_CMD", _DEFAULT
    )


# ------------------------------------------------
# Regex
# ------------------------------------------------
_FIG_CAP_RE = re.compile(r"^Figure\s+(\d+[a-zA-Z]?)[\s:\.\-–—](.*)", re.I | re.S)
_TBL_CAP_RE = re.compile(r"^Table\s+(\d+[a-zA-Z]?)[\s:\.\-–—](.*)", re.I | re.S)

_REF_FIG_RE = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+[a-zA-Z]?)", re.I)
_REF_TBL_RE = re.compile(r"\bTable\s+(\d+[a-zA-Z]?)", re.I)

_SHORT_LINE_RE = re.compile(r"^.{1,60}$")

_STRUCTURAL = {"Image", "Table", "FigureCaption"}
_HARD_DROP = {"Header", "Footer", "PageBreak"}


# =========================================================
# Main Extractor
# =========================================================
class PptxExtractor:

    def __init__(self, chunk_size=1500, chunk_overlap=300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -----------------------------------------------------
    # Public entry
    # -----------------------------------------------------
    def extract(self, file_paths: List[str], output_path: str) -> Dict:

        presentations = []

        for idx, file_path in enumerate(file_paths):

            ppt_id = f"ppt_{idx + 1}"
            print(f"\nProcessing: {file_path}")
            print("ppt_id:", ppt_id)

            try:
                presentations.append(self._process_single(file_path, ppt_id))
            except Exception as e:
                print("Skipping file:", e)

        result = {"presentations": presentations}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\nSaved:", output_path)

        return result

    # -----------------------------------------------------
    # Process single PPT
    # -----------------------------------------------------
    def _process_single(self, file_path: str, ppt_id: str) -> Dict:

        print("Partitioning pptx...")

        raw = partition_pptx(
            filename=file_path,
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )

        repeats = _find_repeated_short_lines(raw)
        clean = _filter_noise(raw, repeats)

        slides = _group_by_slide(clean)

        texts = self._extract_texts(slides, ppt_id)
        figures = self._extract_figures(slides, ppt_id)
        tables = self._extract_tables(slides, ppt_id)

        _cross_link(texts, figures, tables)

        return {
            "ppt_id": ppt_id,
            "source": file_path,
            "texts": texts,
            "figures": figures,
            "tables": tables,
        }

    # -----------------------------------------------------
    # Text extraction
    # -----------------------------------------------------
    def _extract_texts(self, slides: Dict[int, List[Any]], ppt_id: str) -> List[Dict]:

        chunks = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        for slide_num in sorted(slides):

            parts = []

            for el in slides[slide_num]:

                et = type(el).__name__
                text = _text(el)

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
                    parts.append(text)

            if not parts:
                continue

            slide_text = "\n\n".join(parts)

            for i, chunk in enumerate(splitter.split_text(slide_text)):

                if not chunk.strip():
                    continue

                chunks.append(
                    {
                        "ppt_id": ppt_id,
                        "chunk_id": f"chunk_{slide_num}_{i}",
                        "slide_number": slide_num,
                        "text": chunk.strip(),
                        "referenced_figures": [],
                        "referenced_tables": [],
                    }
                )

        return chunks

    # -----------------------------------------------------
    # Figures
    # -----------------------------------------------------
    def _extract_figures(self, slides, ppt_id):

        figures = {}

        for slide_num in sorted(slides):

            elements = slides[slide_num]
            n = len(elements)
            used = set()

            for i, el in enumerate(elements):

                if type(el).__name__ != "Image":
                    continue

                caption, cap_idx = _find_caption(elements, i, n, used, "figure")

                if not caption:
                    continue

                m = _FIG_CAP_RE.match(caption)
                fid = f"Figure {m.group(1)}"

                used.add(cap_idx)

                if fid not in figures:

                    figures[fid] = {
                        "ppt_id": ppt_id,
                        "figure_id": fid,
                        "slide_number": slide_num,
                        "image_base64": _b64(el),
                        "caption": caption,
                        "mentioned_in_chunks": [],
                    }

        return sorted(figures.values(), key=lambda f: (f["slide_number"], _num_key(f["figure_id"])))

    # -----------------------------------------------------
    # Tables
    # -----------------------------------------------------
    def _extract_tables(self, slides, ppt_id):

        tables = {}

        for slide_num in sorted(slides):

            elements = slides[slide_num]
            n = len(elements)
            used = set()

            for i, el in enumerate(elements):

                if type(el).__name__ != "Table":
                    continue

                caption, cap_idx = _find_caption(elements, i, n, used, "table")

                if not caption:
                    continue

                m = _TBL_CAP_RE.match(caption)
                tid = f"Table {m.group(1)}"

                used.add(cap_idx)

                html = getattr(getattr(el, "metadata", None), "text_as_html", None) or _text(el)

                if tid not in tables:

                    tables[tid] = {
                        "ppt_id": ppt_id,
                        "table_id": tid,
                        "slide_number": slide_num,
                        "html": html,
                        "caption": caption,
                        "mentioned_in_chunks": [],
                    }

        return sorted(tables.values(), key=lambda t: (t["slide_number"], _num_key(t["table_id"])))


# =========================================================
# Utilities
# =========================================================
def _text(el):
    return (el.text if hasattr(el, "text") else str(el)).strip()


def _b64(el):
    try:
        return el.metadata.image_base64 or ""
    except AttributeError:
        return ""


def _num_key(s):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 9999


def _group_by_slide(elements):

    slides = defaultdict(list)

    for el in elements:

        slide_num = getattr(getattr(el, "metadata", None), "page_number", 1) or 1
        slides[slide_num].append(el)

    return dict(slides)


def _find_repeated_short_lines(elements):

    counter = Counter()

    for el in elements:

        if type(el).__name__ in _STRUCTURAL | _HARD_DROP:
            continue

        t = _text(el)

        if t and _SHORT_LINE_RE.match(t) and "\n" not in t:
            counter[t] += 1

    return {t for t, c in counter.items() if c >= 3}


def _filter_noise(elements, repeats):

    clean = []

    for el in elements:

        et = type(el).__name__
        text = _text(el)

        if et in _HARD_DROP:
            continue

        if et in _STRUCTURAL:
            clean.append(el)
            continue

        if len(text) <= 3 or text in repeats:
            continue

        clean.append(el)

    return clean


def _find_caption(elements, anchor, n, used, kind):

    cap_re = _FIG_CAP_RE if kind == "figure" else _TBL_CAP_RE

    for direction in (
        range(anchor + 1, min(anchor + 7, n)),
        range(anchor - 1, max(anchor - 7, -1), -1),
    ):

        for j in direction:

            if j in used:
                continue

            et = type(elements[j]).__name__
            text = _text(elements[j])

            if et in {"Image", "Table"}:
                break

            if cap_re.match(text):
                return text, j

    return "", -1


def _cross_link(texts, figures, tables):

    fig_idx = {f["figure_id"]: f for f in figures}
    tbl_idx = {t["table_id"]: t for t in tables}

    for chunk in texts:

        cid = chunk["chunk_id"]

        for num in _REF_FIG_RE.findall(chunk["text"]):

            fid = f"Figure {num}"

            if fid in fig_idx:

                if fid not in chunk["referenced_figures"]:
                    chunk["referenced_figures"].append(fid)

                if cid not in fig_idx[fid]["mentioned_in_chunks"]:
                    fig_idx[fid]["mentioned_in_chunks"].append(cid)

        for num in _REF_TBL_RE.findall(chunk["text"]):

            tid = f"Table {num}"

            if tid in tbl_idx:

                if tid not in chunk["referenced_tables"]:
                    chunk["referenced_tables"].append(tid)

                if cid not in tbl_idx[tid]["mentioned_in_chunks"]:
                    tbl_idx[tid]["mentioned_in_chunks"].append(cid)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    file_paths = [
        r"presentation1.pptx",
        r"presentation2.pptx",
    ]

    output_path = "ppt_extraction.json"

    extractor = PptxExtractor(chunk_size = 1500, chunk_overlap= 500)

    result = extractor.extract(file_paths, output_path)

    print("\nExtraction complete.")