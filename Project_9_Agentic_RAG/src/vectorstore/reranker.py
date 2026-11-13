import sys
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.utils.logger import logging
from src.utils.exception import CustomException
import warnings
warnings.filterwarnings("ignore")

class DocumentReranker:
    """
    Cross-Encoder based document reranker.

    Responsibilities:
    1. Receive candidate documents.
    2. Calculate query-document relevance scores.
    3. Sort documents by relevance.
    4. Return the final top-K documents.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k : int = 5) -> None:
        try:
            if top_k <= 0:
                raise ValueError("top_k must be greater than zero.")

            self.top_k = top_k

            logging.info("Loading reranker model: %s", model_name)

            self.model = CrossEncoder(model_name)

            logging.info("Cross-Encoder reranker initialized successfully.")

        except Exception as e:
            logging.exception("Failed to initialize document reranker.")
            raise CustomException(e, sys)

    def rerank(self, query : str, documents : List[Document]) -> List[Document]:
        try:
            if not query or not query.strip():
                logging.warning("Empty query received for reranking.")
                return []

            if not documents:
                logging.warning("No documents available for reranking.")
                return []

            logging.info("Starting reranking for %d candidate documents.", len(documents))

            # Create query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]

            # Calculate relevance scores
            scores = self.model.predict(pairs)

            # Attach scores and sort documents
            ranked_documents = sorted(
                zip(documents, scores),
                key = lambda item: float(item[1]),
                reverse = True
            )

            # Return final top-K documents
            final_documents = []

            for doc, score in ranked_documents[:self.top_k]:
                # Copy metadata to avoid modifying the original document
                reranked_doc = Document(
                    page_content = doc.page_content,
                    metadata = {
                        **doc.metadata,
                        "rerank_score": float(score)
                    }
                )

                final_documents.append(reranked_doc)

            logging.info("Reranking completed: %d candidates -> %d documents.", len(documents), len(final_documents))

            return final_documents

        except Exception as e:
            logging.exception("Document reranking failed.")
            raise CustomException(e, sys)