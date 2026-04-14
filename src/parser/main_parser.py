import os

from .parse_csv import CSVExtractor
from .parse_pdf import MultimodalPDFExtractor
from .parse_txt import TxtExtractor
from .parse_docx import DocxExtractor
from .parse_pptx import PptxExtractor


class MainParser:

    def __init__(self):
        pass

    def parse(self, input_files, output_path, chunk_size=3000, chunk_overlap=1000):

        # Ensure input is always a list
        if isinstance(input_files, str):
            input_files = [input_files]

        # Get file extension from the file
        _, ext = os.path.splitext(input_files[0])
        ext = ext.lower()

        # Select parser based on the file type
        if ext == ".csv":
            extractor = CSVExtractor()

        elif ext == ".pdf":
            extractor = MultimodalPDFExtractor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif ext == ".txt":
            extractor = TxtExtractor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif ext == ".docx":
            extractor = DocxExtractor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif ext == ".pptx":
            extractor = PptxExtractor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        else:
            print("Unsupported file type")
            return

        # Run extraction
        extractor.extract(input_files, output_path)

        print("\nExtraction complete.")


# ─────────────────────────────────────────
# Sample usage
if __name__ == "__main__":

    parser = MainParser()

    parser.parse(
        input_files=r"D:\SL_Projects\Projects\AI_Python\Qwen\Data\Papers\qwen1_technical_report.pdf",   # or ["file1.pdf", "file2.pdf"]
        output_path=r"D:\SL_Projects\Projects\AI_Python\Multimodal-RAG\dump\output_test_sample.json",
        chunk_size=10000,
        chunk_overlap=3000
    )