from src.embeddings import get_embeddings
from src.config import COLLECTION_NAME
from src.vectorstores import get_qdrant_client

def retrieve_docs(query: str, top_k=5):
    # Get the Qdrant client
    client = get_qdrant_client()

    query_vector = get_embeddings([query])[0]

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )

    print("Retriever module loaded successfully.")
    return [hit.payload["text"] for hit in search_result]
    
    
