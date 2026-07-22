"""
vector_store.py — Local embedded Qdrant vector database.
Runs entirely on disk (.qdrant_data/), no server process, no Docker.
"""
import hashlib
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

VECTOR_SIZE = 384  # matches all-MiniLM-L6-v2's output dimension
COLLECTION = "paper_embeddings"

_client = QdrantClient(path=".qdrant_data")

if not _client.collection_exists(COLLECTION):
    _client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def _text_to_id(text: str) -> str:
    h = hashlib.sha256(text.encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


def get_cached(text: str) -> list[float] | None:
    point_id = _text_to_id(text)
    result = _client.retrieve(collection_name=COLLECTION, ids=[point_id], with_vectors=True)
    if result:
        return result[0].vector
    return None


def store(text: str, vector: list[float], payload: dict | None = None):
    point_id = _text_to_id(text)
    _client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=point_id, vector=vector, payload=payload or {"text": text[:500]})],
    )


def search_similar(query_vector: list[float], top_k: int = 10, paper_ids: list[str] | None = None):
    query_filter = None
    if paper_ids:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        query_filter = Filter(must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))])

    results = _client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )
    return results.points