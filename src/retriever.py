import json
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document


# ------------------------------------------------
# Config
# ------------------------------------------------

MMR_FETCH_K = 10
MMR_TOP_K = 6
MMR_LAMBDA = 0.7

BM25_TOP_K = 6

DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

RERANK_TOP_N = 5

LIST_FIELDS = {"referenced_figures", "referenced_tables", "mentioned_in_chunks"}


# ------------------------------------------------
# Retriever
# ------------------------------------------------

class RetrieverSession:

    def __init__(self, ingest_session, cohere_api_key: str):
        self.session = ingest_session
        self.cohere_api_key = cohere_api_key
        self.pipeline = None

    # ----------------------------
    # Build retrievers
    # ----------------------------

    def _get_mmr(self):
        return self.session.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": MMR_TOP_K,
                "fetch_k": MMR_FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            },
        )

    def _get_bm25(self):
        retriever = BM25Retriever.from_documents(self.session.docs)
        retriever.k = BM25_TOP_K
        return retriever

    def _get_rrf(self):
        return EnsembleRetriever(
            retrievers=[self._get_mmr(), self._get_bm25()],
            weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
        )

    def _get_reranker(self):
        return CohereRerank(
            cohere_api_key=self.cohere_api_key,
            model="rerank-multilingual-v3.0",
            top_n=RERANK_TOP_N,
        )

    def _build_pipeline(self):

        if self.session.vectorstore is None:
            raise ValueError("Run ingest first")

        self.pipeline = ContextualCompressionRetriever(
            base_retriever=self._get_rrf(),
            base_compressor=self._get_reranker(),
        )

    # ----------------------------
    # Helpers
    # ----------------------------

    def _deserialize(self, metadata: Dict) -> Dict:

        data = dict(metadata)

        for key in LIST_FIELDS:
            if key in data and isinstance(data[key], str):
                data[key] = json.loads(data[key])

        return data

    def _filter(self, docs, source_type=None, content_type=None):

        if source_type:
            docs = [d for d in docs if d.metadata.get("source_type") == source_type]

        if content_type:
            docs = [d for d in docs if d.metadata.get("content_type") == content_type]

        return docs

    # ----------------------------
    # Public API
    # ----------------------------

    def query(self, query: str, filter_source_type=None, filter_content_type=None):

        if self.pipeline is None:
            self._build_pipeline()

        docs = self.pipeline.invoke(query)

        docs = self._filter(docs, filter_source_type, filter_content_type)

        return [self._deserialize(doc.metadata) for doc in docs]

    def reset(self):
        self.pipeline = None

    @property
    def config(self):
        return {
            "mmr_fetch_k": MMR_FETCH_K,
            "mmr_top_k": MMR_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "dense_weight": DENSE_WEIGHT,
            "sparse_weight": SPARSE_WEIGHT,
            "rerank_top_n": RERANK_TOP_N,
        }


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    from ingest import IngestSession

    session = IngestSession("extraction.json").run()

    retriever = RetrieverSession(
        ingest_session=session,
        cohere_api_key="xDB2CZr31oTonsAsJip3RN4gnMHlu3KgFlSfL11z",
    )

    results = retriever.query("What is the architecture?")

    print("\nResults:")
    for r in results:
        print(r)