from qdrant_client import QdrantClient
import os

def get_qdrant_client():
    apiqdrant = os.getenv('QDRANT')
    qdrant_url = os.getenv('QDRANT_URL')
    return QdrantClient(url=qdrant_url, api_key=apiqdrant)

def search_qdrant(query, embedding_model, collection_name):
    qdrant_client = get_qdrant_client()
    query_embedding = list(embedding_model.embed([query]))[0]
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3,
    )
    return search_results
