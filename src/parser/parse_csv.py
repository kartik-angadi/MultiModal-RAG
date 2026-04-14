import os
import json
import pandas as pd
from typing import List, Dict


class CSVExtractor:

    # ------------------------------------------------
    # Public entry
    # ------------------------------------------------
    def extract(self, file_paths: List[str], output_path: str) -> Dict:

        csv_files = []

        for idx, file_path in enumerate(file_paths):

            csv_id = f"csv_{idx+1}"

            print("\nProcessing:", file_path)
            print("csv_id:", csv_id)

            try:
                csv_files.append(self._process_single(file_path, csv_id))
            except Exception as e:
                print("Skipping file:", e)

        result = {"csv_files": csv_files}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\nSaved:", output_path)

        return result

    # ------------------------------------------------
    # Process single CSV
    # ------------------------------------------------
    def _process_single(self, file_path: str, csv_id: str):

        df = pd.read_csv(file_path)

        rows = []

        for idx, row in df.iterrows():

            rows.append(
                {
                    "csv_id": csv_id,
                    "row_id": f"{csv_id}_row_{idx+1}",
                    "data": row.to_dict(),
                }
            )

        return {
            "csv_id": csv_id,
            "source": file_path,
            "rows": rows,
        }


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    file_paths = [
        "sample.csv"
    ]

    output_path = "csv_extraction.json"

    extractor = CSVExtractor()

    result = extractor.extract(file_paths, output_path)

    print("\nExtraction complete.")