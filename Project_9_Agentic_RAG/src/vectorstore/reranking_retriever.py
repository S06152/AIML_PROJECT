from src.vectorstore.reranker import DocumentReranker
from src.utils.logger import logging


class RerankingRetriever:
    """
    Combines hybrid retrieval and Cross-Encoder reranking.

    Pipeline:
        Query
          |
        EnsembleRetriever
          |
        Candidate Documents
          |
        DocumentReranker
          |
        Final Top-K Documents
    """

    def __init__(
        self,
        base_retriever,
        reranker: DocumentReranker,
        candidate_k: int = 20
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def invoke(self, query: str, config=None, **kwargs):

        if not query or not query.strip():
            return []

        # Step 1: Retrieve candidate documents using hybrid search
        candidate_documents = self.base_retriever.invoke(
            query,
            config=config,
            **kwargs
        )

        # Step 2: Limit the candidate document pool
        candidate_documents = candidate_documents[:self.candidate_k]

        logging.info(
            "Hybrid retrieval returned %d candidate documents",
            len(candidate_documents)
        )

        # Step 3: Rerank candidate documents
        reranked_documents = self.reranker.rerank(
            query=query,
            documents=candidate_documents
        )

        # Step 4: Return final ranked documents
        return reranked_documents