from typing import List

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.utils.logger import logging


class DocumentReranker:
    """
    Cross-Encoder based document reranker.

    Responsibilities:
    1. Receive candidate documents.
    2. Calculate query-document relevance scores.
    3. Sort documents by relevance.
    4. Return the final top-K documents.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5
    ):
        self.top_k = top_k

        logging.info("Loading reranker model: %s", model_name)

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:

        if not documents:
            return []

        print("\n========== 3. RETRIEVED DOCUMENTS (BEFORE RERANKING) ==========")

        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content)
            print("Metadata:", doc.metadata)

        # Create query-document pairs
        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        # Calculate relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to documents
        ranked_documents = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        # Return final top-K documents
        final_documents = []

        for doc, score in ranked_documents[:self.top_k]:

            doc.metadata["rerank_score"] = float(score)

            final_documents.append(doc)
        
        print("\n========== 4. RERANKED DOCUMENTS (AFTER RERANKING) ==========")

        for i, doc in enumerate(final_documents, 1):
            print(f"\nDocument {i}:")
            print("Rerank Score:", doc.metadata.get("rerank_score"))
            print(doc.page_content)
            print("Metadata:", doc.metadata)

        logging.info(
            "Reranking completed: %d candidates -> %d documents",
            len(documents),
            len(final_documents)
        )

        return final_documents