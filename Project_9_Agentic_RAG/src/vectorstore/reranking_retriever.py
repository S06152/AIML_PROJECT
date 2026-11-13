import sys
from typing import List, Optional
from langchain_core.documents import Document
from src.vectorstore.reranker import DocumentReranker
from src.utils.logger import logging
from src.utils.exception import CustomException
import warnings
warnings.filterwarnings("ignore")

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

    def __init__(self, base_retriever, reranker : DocumentReranker, candidate_k : int = 20) -> None:
        try:
            if candidate_k <= 0:
                raise ValueError("candidate_k must be greater than zero.")

            self.base_retriever = base_retriever
            self.reranker = reranker
            self.candidate_k = candidate_k

            logging.info("RerankingRetriever initialized with candidate_k = %d", candidate_k)

        except Exception as e:
            logging.exception("RerankingRetriever initialization failed.")
            raise CustomException(e, sys)

    def invoke(self, query : str, config : Optional[dict] = None, **kwargs) -> List[Document]:
        try:
            if not query or not query.strip():
                logging.warning("Empty query received by RerankingRetriever.")
                return []

            logging.info("Starting hybrid retrieval and reranking.")

            # Step 1: Retrieve candidate documents using hybrid search
            candidate_documents = self.base_retriever.invoke(query, config = config, **kwargs )

            # Step 2: Limit the candidate document pool
            candidate_documents = candidate_documents[:self.candidate_k]

            logging.info("Hybrid retrieval returned %d candidate documents.", len(candidate_documents))

            if not candidate_documents:
                logging.warning("No candidate documents found.")
                return []

            # Step 3: Rerank candidate documents
            reranked_documents = self.reranker.rerank(query = query, documents = candidate_documents)

            # Step 4: Return final ranked documents
            logging.info("Reranking completed. Returning %d documents.", len(reranked_documents))

            return reranked_documents

        except Exception as e:
            logging.exception("Hybrid retrieval and reranking failed.")
            raise CustomException(e, sys)
