"""ZenML steps and pipeline for the Inteli RAG inference service."""

from __future__ import annotations

import os
from typing import List, Optional

from qdrant_client import QdrantClient
from zenml import pipeline, step
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "inteli_admission_chunks")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
SCORE_THRESHOLD = os.getenv("RAG_SCORE_THRESHOLD")

_embedder: Optional[SentenceTransformer] = None
_qdrant_client: Optional[QdrantClient] = None


def _get_embedder() -> SentenceTransformer:
    """Load the sentence-transformer once to avoid re-instantiating it per run."""
    global _embedder  # pylint: disable=global-statement
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _qdrant_client  # pylint: disable=global-statement
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def _parse_score_threshold() -> Optional[float]:
    """Convert SCORE_THRESHOLD to float when provided."""
    if not SCORE_THRESHOLD:
        return None
    try:
        return float(SCORE_THRESHOLD)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            "RAG_SCORE_THRESHOLD must be a float-compatible value."
        ) from exc


@step(enable_cache=False)
def query_embedding_step(query: str) -> List[float]:
    """Encode the user query into an embedding vector."""
    if not query:
        raise ValueError("query_embedding_step recebeu uma query vazia.")
    embedder = _get_embedder()
    return embedder.encode(query).tolist()


@step(enable_cache=False)
def retrieval_from_qdrant_step(
    query_embedding: List[float],
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Retrieve the top-k most similar chunks from Qdrant."""
    if not query_embedding:
        raise ValueError("retrieval_from_qdrant_step recebeu embedding vazio.")

    client = _get_qdrant_client()
    threshold = _parse_score_threshold()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        score_threshold=threshold,
    )

    contexts: List[str] = []
    for idx, point in enumerate(results, start=1):
        payload = point.payload or {}
        content = payload.get("content")
        if not content:
            continue
        metadata = payload.get("metadata") or {}
        header_parts = [f"Trecho {idx}"]
        section = metadata.get("section") or metadata.get("section_context")
        if section:
            header_parts.append(f"seção: {section}")
        page = metadata.get("page_number")
        if page is not None:
            header_parts.append(f"página: {page}")
        chunk_id = payload.get("chunk_id") or metadata.get("chunk_id")
        if chunk_id:
            header_parts.append(f"id: {chunk_id}")
        header = " | ".join(header_parts)
        contexts.append(f"{header}\n{content}")

    if not contexts:
        raise RuntimeError(
            "Nenhum contexto relevante foi retornado pela busca no Qdrant."
        )

    return "\n\n".join(contexts)


@pipeline
def rag_inference_pipeline(query: str) -> str:
    """Pipeline que encadeia embed + retrieval e retorna o contexto concatenado."""
    query_embedding = query_embedding_step(query=query)
    context = retrieval_from_qdrant_step(query_embedding=query_embedding)
    return context
