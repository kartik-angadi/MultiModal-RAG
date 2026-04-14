import os
import json
import uuid
import shutil
import atexit
import tempfile

from typing import List, Dict

from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document


# ------------------------------------------------
# Config
# ------------------------------------------------

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "xDB2CZr31oTonsAsJip3RN4gnMHlu3KgFlSfL11z")
COLLECTION_NAME = "multimodal_rag"
ID_KEY = "doc_id"

SOURCE_ID_FIELDS = {
    "papers": "paper_id",
    "documents": "document_id",
    "txt_files": "txt_file_id",
    "csv_files": "csv_file_id",
    "presentations": "ppt_id",
}


# ------------------------------------------------
# Document Builder
# ------------------------------------------------

class DocumentBuilder:

    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    def _serialize(self, value: List) -> str:
        return json.dumps(value)

    def _build_text_docs(self, chunks, source_type, id_field, source_id):
        docs = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            metadata = {
                ID_KEY: self._generate_id(),
                "source_type": source_type,
                "content_type": "text",
                id_field: source_id,
                "chunk_id": chunk["chunk_id"],
                "referenced_figures": self._serialize(chunk.get("referenced_figures", [])),
                "referenced_tables": self._serialize(chunk.get("referenced_tables", [])),
            }

            if "slide_number" in chunk:
                metadata["slide_number"] = chunk["slide_number"]

            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def _build_figure_docs(self, figures, source_type, id_field, source_id):
        docs = []

        for fig in figures:
            caption = fig.get("caption", "").strip()
            if not caption:
                continue

            metadata = {
                ID_KEY: self._generate_id(),
                "source_type": source_type,
                "content_type": "figure",
                id_field: source_id,
                "figure_id": fig["figure_id"],
                "mentioned_in_chunks": self._serialize(fig.get("mentioned_in_chunks", [])),
            }

            if "slide_number" in fig:
                metadata["slide_number"] = fig["slide_number"]

            docs.append(Document(page_content=caption, metadata=metadata))

        return docs

    def _build_table_docs(self, tables, source_type, id_field, source_id):
        docs = []

        for tbl in tables:
            caption = tbl.get("caption", "").strip()
            if not caption:
                continue

            metadata = {
                ID_KEY: self._generate_id(),
                "source_type": source_type,
                "content_type": "table",
                id_field: source_id,
                "table_id": tbl["table_id"],
                "mentioned_in_chunks": self._serialize(tbl.get("mentioned_in_chunks", [])),
            }

            if "slide_number" in tbl:
                metadata["slide_number"] = tbl["slide_number"]

            docs.append(Document(page_content=caption, metadata=metadata))

        return docs

    def build_documents(self, data: Dict) -> List[Document]:
        all_docs = []

        for source_type, id_field in SOURCE_ID_FIELDS.items():

            for entry in data.get(source_type, []):
                source_id = entry[id_field]

                all_docs += self._build_text_docs(
                    entry.get("texts", []), source_type, id_field, source_id
                )

                all_docs += self._build_figure_docs(
                    entry.get("figures", []), source_type, id_field, source_id
                )

                all_docs += self._build_table_docs(
                    entry.get("tables", []), source_type, id_field, source_id
                )

        return all_docs


# ------------------------------------------------
# Ingest Session
# ------------------------------------------------

class IngestSession:

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.vectorstore = None
        self.docs = []
        self.persist_dir = None

    def _load_json(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_embeddings(self):
        return CohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=COHERE_API_KEY,
        )

    def _create_temp_db(self):
        temp_dir = tempfile.mkdtemp(prefix="chroma_rag_")
        atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
        return temp_dir

    def run(self):

        data = self._load_json()

        builder = DocumentBuilder()
        self.docs = builder.build_documents(data)

        if not self.docs:
            raise ValueError("No documents found in JSON")

        self.persist_dir = self._create_temp_db()

        self.vectorstore = Chroma.from_documents(
            documents=self.docs,
            embedding=self._get_embeddings(),
            collection_name=COLLECTION_NAME,
            persist_directory=self.persist_dir,
        )

        return self

    def cleanup(self):
        if self.persist_dir and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        self.vectorstore = None
        self.docs = []


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    json_path = "extraction.json"

    session = IngestSession(json_path).run()

    print("Documents:", len(session.docs))
    print("Vector DB created")