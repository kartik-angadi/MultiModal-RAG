import os
import json
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TxtExtractor:

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------
    # Public entry
    # ------------------------------------------------
    def extract(self, file_paths: List[str], output_path: str) -> Dict:

        txt_files = []

        for idx, file_path in enumerate(file_paths):

            txt_id = f"txt_{idx+1}"

            print("\nProcessing:", file_path)
            print("txt_id:", txt_id)

            try:
                txt_files.append(self._process_single(file_path, txt_id))
            except Exception as e:
                print("Skipping file:", e)

        result = {"txt_files": txt_files}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\nSaved:", output_path)

        return result

    # ------------------------------------------------
    # Process single TXT
    # ------------------------------------------------
    def _process_single(self, file_path: str, txt_id: str):

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_text(text)

        texts = []

        for i, chunk in enumerate(chunks):

            texts.append(
                {
                    "txt_id": txt_id,
                    "chunk_id": f"{txt_id}_chunk_{i+1}",
                    "text": chunk,
                }
            )

        return {
            "txt_id": txt_id,
            "source": file_path,
            "texts": texts,
        }


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    file_paths = [
        "sample.txt"
    ]

    output_path = "txt_extraction.json"

    extractor = TxtExtractor(chunk_size= 3000, chunk_overlap= 1000)

    result = extractor.extract(file_paths, output_path)

    print("\nExtraction complete.")