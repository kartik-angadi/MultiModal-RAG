import os
import json
from typing import List, Dict

from unstructured.partition.docx import partition_docx
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocxExtractor:

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------
    # Public entry
    # ------------------------------------------------
    def extract(self, file_paths: List[str], output_path: str) -> Dict:

        documents = []

        for idx, file_path in enumerate(file_paths):

            doc_id = f"document_{idx+1}"

            print("\nProcessing:", file_path)
            print("document_id:", doc_id)

            try:
                documents.append(self._process_single(file_path, doc_id))
            except Exception as e:
                print("Skipping file:", e)

        result = {"documents": documents}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\nSaved:", output_path)

        return result

    # ------------------------------------------------
    # Process single docx
    # ------------------------------------------------
    def _process_single(self, file_path: str, doc_id: str):

        elements = partition_docx(filename=file_path)

        texts = []

        full_text = []

        for el in elements:

            text = getattr(el, "text", "")

            if text and text.strip():
                full_text.append(text.strip())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_text("\n".join(full_text))

        for i, chunk in enumerate(chunks):

            texts.append(
                {
                    "document_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{i+1}",
                    "text": chunk,
                }
            )

        return {
            "document_id": doc_id,
            "source": file_path,
            "texts": texts,
        }


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    file_paths = [
        "sample.docx"
    ]

    output_path = "doc_extraction.json"

    extractor = DocxExtractor(chunk_size = 5000, chunk_overlap= 1500)

    result = extractor.extract(file_paths, output_path)

    print("\nExtraction complete.")