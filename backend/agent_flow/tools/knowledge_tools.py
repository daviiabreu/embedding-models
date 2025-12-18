
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Optional, Dict, List

from dotenv import load_dotenv
from google.adk.tools.tool_context import ToolContext
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.agent_flow.utils.cache import (
    cache_rag_result,
    cached_rag_query,
    get_embedding_cache,
    make_cache_key,
)
from backend.agent_flow.utils.retry import retry

logger = logging.getLogger(__name__)


AGENT_FLOW_DIR = Path(__file__).resolve().parents[1]
load_dotenv(AGENT_FLOW_DIR / ".env", override=False)


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "inteli-documents-embeddings")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "30"))
DEFAULT_ADJACENT_LIMIT = int(os.getenv("RAG_ADJACENT_LIMIT", "10"))
ADJACENCY_FIELD = os.getenv("RAG_ADJACENCY_FIELD", "adjacent_ids")
SCORE_THRESHOLD = os.getenv("RAG_SCORE_THRESHOLD")
INCLUDE_EMBEDDINGS = os.getenv("RAG_INCLUDE_EMBEDDINGS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _parse_score_threshold() -> float | None:
    if not SCORE_THRESHOLD:
        return None
    try:
        return float(SCORE_THRESHOLD)
    except ValueError as exc:
        raise ValueError(
            "RAG_SCORE_THRESHOLD must be a float-compatible value."
        ) from exc


def _stringify_point_id(point_id: Any) -> str:
    if isinstance(point_id, bytes):
        return point_id.decode("utf-8", errors="ignore")
    return str(point_id)


def _convert_to_qdrant_id(value: Any) -> Any:
    if isinstance(value, (int,)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        if stripped.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return stripped
        return stripped
    return value


def _prepare_vector(vector: Any) -> Any:
    if vector is None:
        return None
    if isinstance(vector, list):
        return vector
    if isinstance(vector, dict):
        prepared: dict[str, Any] = {}
        for key, value in vector.items():
            if isinstance(value, list):
                prepared[key] = value
            elif isinstance(value, SequenceABC):
                prepared[key] = list(value)
            else:
                prepared[key] = value
        return prepared
    if isinstance(vector, SequenceABC):
        return list(vector)
    return vector


def _extract_adjacency_candidates(
    payload: dict[str, Any], metadata: dict[str, Any]
) -> Any:
    for key in (ADJACENCY_FIELD, "adjacent_ids", "neighbors", "edges"):
        value = payload.get(key)
        if value is None:
            value = metadata.get(key)
        if value is not None:
            return value
    return None


def _normalize_adjacency_ids(
    adjacency_raw: Any, limit: int
) -> tuple[list[str], list[Any]]:
    if adjacency_raw is None or limit <= 0:
        return [], []

    if isinstance(adjacency_raw, str):
        tokens = [token.strip() for token in adjacency_raw.replace(";", ",").split(",")]
    elif isinstance(adjacency_raw, SequenceABC):
        tokens = list(adjacency_raw)
    else:
        tokens = [adjacency_raw]

    normalized_strings: list[str] = []
    qdrant_ids: list[Any] = []
    for token in tokens:
        if token is None:
            continue
        token_str = str(token).strip()
        if not token_str:
            continue
        normalized_strings.append(token_str)
        qdrant_ids.append(_convert_to_qdrant_id(token))
        if len(normalized_strings) >= limit:
            break
    return normalized_strings, qdrant_ids


def _retrieve_adjacency_payloads(
    client: QdrantClient, adjacency_ids: Sequence[Any]
) -> dict[str, dict[str, Any]]:
    if not adjacency_ids:
        return {}

    unique_ids: list[Any] = []
    seen: set[str] = set()
    for candidate in adjacency_ids:
        normalized = _stringify_point_id(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_ids.append(candidate)

    if not unique_ids:
        return {}

    records = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=unique_ids,
        with_payload=True,
        with_vectors=INCLUDE_EMBEDDINGS,
    )

    adjacency_map: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.payload or {}
        metadata = payload.get("metadata") or {}
        key = _stringify_point_id(record.id)
        adjacency_entry: dict[str, Any] = {
            "id": key,
            "score": None,
            "content": payload.get("content"),
            "metadata": metadata,
        }
        if INCLUDE_EMBEDDINGS:
            vector = _prepare_vector(getattr(record, "vector", None))
            if vector is not None:
                adjacency_entry["embedding"] = vector
        adjacency_map[key] = adjacency_entry

    return adjacency_map


def _extract_query_points(results: Any) -> list[Any]:
    if hasattr(results, "points"):
        payload = getattr(results, "points")
        return list(payload or [])
    if isinstance(results, dict):
        payload = results.get("points")
        if payload is None:
            return []
        return list(payload)
    if isinstance(results, SequenceABC) and not isinstance(results, (str, bytes)):
        return list(results)
    return [results]


def _resolve_scored_point(point: Any) -> Any:
    if isinstance(point, tuple) and point:
        candidate = point[0]
        if hasattr(candidate, "payload") or isinstance(candidate, dict):
            return candidate
    return point


# Global cache for embedding model (P0-4: Cache embedding model)
_embedding_model_cache = None


def get_embedding_model():
    """Get cached embedding model instance. Loads model on first call, then reuses."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model_cache


def query_embedding(query: str) -> list[float]:
    """Convert text query into embedding vector using sentence transformer model. Returns dense vector representation for semantic search."""
    if not query:
        raise ValueError("query_embedding_step recebeu uma query vazia.")

    # Check embedding cache first
    cache = get_embedding_cache()
    cache_key = make_cache_key("embedding", query.lower().strip())
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug(f"Embedding cache hit for query: {query[:50]}...")
        return cached

    # Generate new embedding
    model = get_embedding_model()
    embedding = model.encode(query).tolist()

    # Cache the result
    cache.set(cache_key, embedding)
    return embedding


@retry(
    max_attempts=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)
def retrieval_from_qdrant(
    query_embedding: list[float],
    top_k: int = DEFAULT_TOP_K,
    adjacency_limit: int = DEFAULT_ADJACENT_LIMIT,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Search Qdrant vector database for relevant documents using embedding similarity. 
    
    Args:
        query_embedding: Dense vector embedding of the search query
        top_k: Maximum number of results to return
        adjacency_limit: Maximum number of adjacent nodes to retrieve per result
        metadata_filter: Optional dict of metadata field filters (e.g., {"project": "Projeto 7"})
    
    Returns top-k results with adjacency expansion for graph-based retrieval."""
    if not query_embedding:
        raise ValueError("retrieval_from_qdrant_step recebeu embedding vazio.")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    threshold = _parse_score_threshold()
    
    # Build Qdrant filter from metadata_filter dict
    qdrant_filter = None
    if metadata_filter:
        conditions = []
        for key, value in metadata_filter.items():
            # Construct the nested path for metadata fields
            field_path = f"metadata.{key}"
            conditions.append(
                FieldCondition(
                    key=field_path,
                    match=MatchValue(value=value)
                )
            )
        if conditions:
            qdrant_filter = Filter(must=conditions)

    query_result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        using="dense",
        limit=top_k,
        with_payload=True,
        with_vectors=INCLUDE_EMBEDDINGS,
        score_threshold=threshold,
        query_filter=qdrant_filter,
    )
    scored_points = _extract_query_points(query_result)
    retrieved_nodes: list[dict[str, Any]] = []
    adjacency_lookup: dict[str, list[str]] = {}
    adjacency_requests: list[Any] = []
    
    # STRICT FILTERING: If filtering by academic_year, exclude chunks with different or null years
    strict_year_filter = None
    if metadata_filter and "academic_year" in metadata_filter:
        strict_year_filter = metadata_filter["academic_year"]
        logger.info(f"🚨 STRICT YEAR FILTERING ENABLED: Only keeping chunks with academic_year='{strict_year_filter}'")

    for point in scored_points:
        point = _resolve_scored_point(point)
        if hasattr(point, "payload"):
            payload = point.payload or {}
        elif isinstance(point, dict):
            payload = point.get("payload") or point
        else:
            payload = {}
        content = payload.get("content")
        if not content:
            continue
        metadata = payload.get("metadata") or {}
        
        # STRICT YEAR FILTER: Skip chunks with wrong or missing academic_year
        if strict_year_filter:
            chunk_year = metadata.get("academic_year")
            if chunk_year != strict_year_filter:
                # Skip this chunk - wrong year or null
                logger.debug(f"Skipping chunk with academic_year='{chunk_year}' (wanted '{strict_year_filter}')")
                continue
            
            # EXTRA FILTER FOR 4º ANO: Skip chunks mentioning numbered projects or generic course descriptions
            if strict_year_filter == "4º ano":
                import re
                
                # Skip chunks mentioning "Projeto X" where X is a number
                if re.search(r'[Pp]rojeto\s+\d+', content):
                    logger.debug(f"Skipping 4º ano chunk that mentions numbered project (belongs to years 1-3)")
                    continue
                
                # Skip ANY chunk mentioning technologies/topics from years 1-3
                # These are generic course descriptions, NOT specific to 4º ano
                forbidden_patterns = [
                    r'blockchain',
                    r'business\s+intelligence',
                    r'\bBI\b',
                    r'inteligência\s+artificial',
                    r'\b[Ii]A\b',  # IA as standalone word
                    r'IoT',
                    r'internet\s+das\s+coisas',
                    r'machine\s+learning',
                    r'aprendizado\s+de\s+máquina',
                    r'desenvolver\s+software',
                    r'automatizar\s+processos',
                    r'manipulação.*dados',
                    r'processamento\s+de\s+dados',
                    r'programação\s+e\s+desenvolvimento',
                    r'você\s+vai\s+aprender\s+a',
                    r'o\s+que\s+você\s+(vai|irá)\s+aprender'
                ]
                
                # If ANY forbidden pattern matches, skip this chunk
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in forbidden_patterns):
                    logger.debug(f"Skipping 4º ano chunk with generic course description (mentions technologies/skills from other years)")
                    continue
        
        raw_id = getattr(point, "id", None)
        if raw_id is None and isinstance(point, dict):
            raw_id = point.get("id")
        node_id = _stringify_point_id(raw_id)

        raw_score = getattr(point, "score", None)
        if raw_score is None and isinstance(point, dict):
            raw_score = point.get("score")
        entry: dict[str, Any] = {
            "id": node_id,
            "score": raw_score,
            "content": content,
            "metadata": metadata,
        }
        if INCLUDE_EMBEDDINGS:
            vector = getattr(point, "vector", None)
            if vector is None and isinstance(point, dict):
                vector = point.get("vector")
            if vector is not None:
                entry["embedding"] = vector

        adjacency_raw = _extract_adjacency_candidates(payload, metadata)
        adjacency_ids, adjacency_qdrant_ids = _normalize_adjacency_ids(
            adjacency_raw,
            adjacency_limit,
        )
        if adjacency_ids:
            adjacency_lookup[node_id] = adjacency_ids
            adjacency_requests.extend(adjacency_qdrant_ids)

        retrieved_nodes.append(entry)

    # Se não encontrou resultados, retorna lista vazia ao invés de crashar
    # O agente vai responder educadamente que não tem informação
    if not retrieved_nodes:
        logger.warning("⚠️ Nenhum contexto relevante encontrado no Qdrant")
        return []

    adjacency_data = _retrieve_adjacency_payloads(client, adjacency_requests)
    for entry in retrieved_nodes:
        adjacency_ids = adjacency_lookup.get(entry["id"], [])
        entry["adjacent"] = [
            adjacency_data[adj_id]
            for adj_id in adjacency_ids
            if adj_id in adjacency_data
        ]

    return retrieved_nodes


def _format_context_block(node: dict[str, Any], index: int) -> str:
    metadata = node.get("metadata") or {}
    header_parts = [f"Trecho {index}"]
    
    # Add project information (CRITICAL for course queries)
    project = metadata.get("project")
    if project:
        header_parts.append(f"projeto: {project}")
    
    # Add academic year
    academic_year = metadata.get("academic_year")
    if academic_year:
        header_parts.append(f"ano: {academic_year}")
    
    # Add section info
    section = metadata.get("section") or metadata.get("section_context")
    if section:
        header_parts.append(f"seção: {section}")
    
    # Add subsection if available
    subsection = metadata.get("subsection")
    if subsection:
        header_parts.append(f"subseção: {subsection}")
    
    page = metadata.get("page_number")
    if page is not None:
        header_parts.append(f"página: {page}")
    chunk_id = metadata.get("chunk_id") or node.get("id")
    if chunk_id:
        header_parts.append(f"id: {chunk_id}")
    header = " | ".join(header_parts)

    lines = [header, node.get("content") or ""]
    adjacency = node.get("adjacent") or []
    if adjacency:
        adjacency_lines = []
        for adj in adjacency:
            adj_meta = adj.get("metadata") or {}
            adj_section = adj_meta.get("section") or adj_meta.get("section_context")
            label = adj_section or adj.get("id")
            adjacency_lines.append(f"- {label}: {adj.get('content')}")
        lines.append("Adjacentes:\n" + "\n".join(adjacency_lines))

    return "\n".join([line for line in lines if line])


def build_graph_rag_payload(
    query: str,
    query_embedding: list[float],
    retrieved_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format retrieved documents into structured RAG payload. Combines query, embeddings, and retrieved nodes into formatted context text."""
    context_blocks = [
        _format_context_block(node, idx)
        for idx, node in enumerate(retrieved_nodes, start=1)
    ]
    context_text = "\n\n".join(context_blocks)

    return {
        "query": query,
        "query_embedding": query_embedding,
        "results": retrieved_nodes,
        "result_count": len(retrieved_nodes),
        "context": context_text,
    }


def _extract_metadata_filters_from_query(query: str) -> Optional[Dict[str, Any]]:
    """Extract metadata filters from natural language query.
    
    Detects patterns like:
    - "Projeto 7" / "projeto sete" -> {"project": "Projeto 7"}
    - "Módulo 5" / "modulo cinco" -> {"project": "Projeto 5"} (maps module to project)
    - "2º ano" / "segundo ano" -> {"academic_year": "2º ano"}
    - "professor Bryan Kano" -> {"professor_name": "Bryan Kano"}
    
    Returns None if no filters detected.
    """
    import re
    
    filters = {}
    query_lower = query.lower()
    
    # Pattern: "projeto X" (where X is a digit or number word)
    project_match = re.search(r'projeto\s+(\d+)', query_lower)
    if project_match:
        project_num = project_match.group(1)
        filters["project"] = f"Projeto {project_num}"
        logger.info(f"Detected project filter: Projeto {project_num}")
    
    # Pattern: "módulo X" (maps to Projeto X)
    module_match = re.search(r'm[oó]dulo\s+(\d+)', query_lower)
    if module_match:
        module_num = module_match.group(1)
        filters["project"] = f"Projeto {module_num}"
        logger.info(f"Detected module filter (mapped to project): Projeto {module_num}")
    
    # Pattern: "Xº ano" / "X ano"
    year_match = re.search(r'(\d+)[oº°]?\s*ano', query_lower)
    if year_match:
        year_num = year_match.group(1)
        filters["academic_year"] = f"{year_num}º ano"
        logger.info(f"Detected academic year filter: {year_num}º ano")
    
    # Pattern: Professor names (look for capitalized names after "professor")
    # Examples: "professor Bryan Kano", "a professora Ana Cristina"
    # NOTE: Only apply filter if we have at least 2 words (first + last name)
    # If only first name, let semantic search handle it without filter
    professor_pattern = r'professor[a]?\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:de|da|dos|das|do)\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:de|da|dos|das|do)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*)'
    professor_match = re.search(professor_pattern, query, re.IGNORECASE)
    if professor_match:
        professor_name = professor_match.group(1).strip()
        # Only apply filter if we have full name (at least 2 words)
        if len(professor_name.split()) >= 2:
            filters["professor_name"] = professor_name
            logger.info(f"🎯 Detected professor filter: {professor_name}")
        else:
            # Just first name - let semantic search find it without filter
            logger.info(f"⚠️ Only first name detected ({professor_name}), skipping filter - using semantic search")
    
    # Pattern: Course names (adm tech, ciência da computação, etc)
    course_patterns = {
        r'adm\s*tech|administra[çc][aã]o.*tecnologia': 'adm-tech',
        r'ci[êe]ncia.*computa[çc][aã]o': 'ciencia-da-computacao',
        r'engenharia.*software': 'engenharia-de-software',
        r'engenharia.*computa[çc][aã]o': 'engenharia-de-computacao',
        r'sistemas.*informa[çc][aã]o': 'sistemas-de-informacao',
    }
    
    for pattern, course_slug in course_patterns.items():
        if re.search(pattern, query_lower):
            filters["course"] = course_slug
            logger.info(f"Detected course filter: {course_slug}")
            break
    
    return filters if filters else None



def rag_inference_pipeline(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    adjacency_limit: int = DEFAULT_ADJACENT_LIMIT,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Complete RAG pipeline for knowledge retrieval. 
    
    Converts query to embedding, searches Qdrant with optional metadata filtering, 
    expands with graph adjacency, and formats context for LLM.
    
    Args:
        query: Natural language search query
        top_k: Maximum number of results to return
        adjacency_limit: Maximum adjacent nodes per result
        metadata_filter: Optional metadata filters (e.g., {"project": "Projeto 7"})
    """
    # Build cache key including filter
    cache_key_suffix = f"_{metadata_filter}" if metadata_filter else ""
    cache_key = f"{query}_{top_k}{cache_key_suffix}"
    
    # Adjust top_k when filtering by academic_year to avoid mixing years
    # When filtering by year, we want ONLY that year's content, not general content
    adjusted_top_k = top_k
    if metadata_filter and "academic_year" in metadata_filter:
        adjusted_top_k = min(5, top_k)  # Use smaller top_k for year-specific queries
        logger.info(f"Academic year filter detected, reducing top_k from {top_k} to {adjusted_top_k}")
    
    # Check cache first
    cached_result = cached_rag_query(cache_key, adjusted_top_k)
    if cached_result is not None:
        logger.info(f"RAG cache hit for query: {query[:50]}...")
        return cached_result

    # Execute RAG pipeline
    query_vector = query_embedding(query=query)
    retrieval = retrieval_from_qdrant(
        query_embedding=query_vector,
        top_k=adjusted_top_k,
        adjacency_limit=adjacency_limit,
        metadata_filter=metadata_filter,
    )
    
    # Se não encontrou nada, retorna payload vazio indicando ausência de dados
    if not retrieval:
        logger.warning(f"⚠️ Nenhum resultado encontrado para: {query[:50]}...")
        empty_payload = {
            "query": query,
            "retrieved_nodes": [],
            "total_nodes": 0,
            "no_results": True  # Flag para o agente saber que não há dados
        }
        cache_rag_result(cache_key, empty_payload, adjusted_top_k)
        return empty_payload
    
    payload = build_graph_rag_payload(
        query=query,
        query_embedding=query_vector,
        retrieved_nodes=retrieval,
    )

    # Cache the result
    cache_rag_result(cache_key, payload, adjusted_top_k)
    return payload


def retrieve_inteli_knowledge(
    query: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Retrieve knowledge about Inteli from vector database. 
    
    Main tool for answering questions about Inteli courses, scholarships, people, 
    facilities, and admission process using RAG pipeline.
    
    Automatically detects and applies metadata filters from query:
    - "Projeto X" -> filters by project
    - "Módulo X" -> maps to Projeto X and filters
    - "Xº ano" -> filters by academic year
    """
    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ValueError("retrieve_inteli_knowledge recebeu uma consulta vazia.")
    
    # Extract metadata filters from query
    metadata_filter = _extract_metadata_filters_from_query(normalized_query)
    if metadata_filter:
        logger.info(f"Applying metadata filters: {metadata_filter}")

    retrieval_payload = rag_inference_pipeline(
        query=normalized_query,
        top_k=DEFAULT_TOP_K,
        adjacency_limit=DEFAULT_ADJACENT_LIMIT,
        metadata_filter=metadata_filter,
    )

    state_entry = {
        "query": normalized_query,
        "top_k": DEFAULT_TOP_K,
        "adjacency_limit": DEFAULT_ADJACENT_LIMIT,
        "result_count": retrieval_payload.get("result_count", 0),
        "metadata_filter": metadata_filter,
    }
    tool_context.state.setdefault("knowledge_retrievals", []).append(state_entry)
    
    # Build message including filter info
    message_parts = [
        f"Retornados {state_entry['result_count']} nós com até "
        f"{DEFAULT_ADJACENT_LIMIT} vizinhos por nó"
    ]
    if metadata_filter:
        filter_desc = ", ".join(f"{k}={v}" for k, v in metadata_filter.items())
        message_parts.append(f"Filtros aplicados: {filter_desc}")

    return {
        "success": True,
        "query": normalized_query,
        "result_count": retrieval_payload.get("result_count", 0),
        "chunks": retrieval_payload.get("results", []),
        "context": retrieval_payload.get("context", ""),
        "query_embedding": retrieval_payload.get("query_embedding"),
        "metadata_filter": metadata_filter,
        "message": " | ".join(message_parts),
    }


__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_ADJACENT_LIMIT",
    "rag_inference_pipeline",
    "retrieve_inteli_knowledge",
]
