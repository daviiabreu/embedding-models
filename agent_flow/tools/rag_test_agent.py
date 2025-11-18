"""CLI utilitário para testar o Graph RAG usando o modelo do Google ADK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "agent_flow" / ".env")


DEFAULT_ADK_MODEL = os.getenv(
    "DEFAULT_MODEL", os.getenv("ADK_MODEL")
)
DEFAULT_ADK_APP = os.getenv("RAG_TEST_APP", "agents")


from collections.abc import Sequence as SequenceABC

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when python-dotenv not installed
    load_dotenv = None


AGENT_FLOW_DIR = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(AGENT_FLOW_DIR / ".env", override=False)



QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "inteli-documents-embeddings")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "300"))
DEFAULT_ADJACENT_LIMIT = int(os.getenv("RAG_ADJACENT_LIMIT", "10"))
ADJACENCY_FIELD = os.getenv("RAG_ADJACENCY_FIELD", "adjacent_ids")
SCORE_THRESHOLD = os.getenv("RAG_SCORE_THRESHOLD")
INCLUDE_EMBEDDINGS = os.getenv("RAG_INCLUDE_EMBEDDINGS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}


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
    

def _stringify_point_id(point_id: Any) -> str:
    """Normalize any Qdrant point identifier into a string key."""
    if isinstance(point_id, bytes):
        return point_id.decode("utf-8", errors="ignore")
    return str(point_id)


def _convert_to_qdrant_id(value: Any) -> Any:
    """Convert raw adjacency values to the type accepted by Qdrant."""
    if isinstance(value, (int,)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        if stripped.isdigit():
            try:
                return int(stripped)
            except ValueError:  # pragma: no cover - defensive
                return stripped
        return stripped
    return value


def _prepare_vector(vector: Any) -> Any:
    """Convert vectors returned by Qdrant into plain Python types."""
    if vector is None:
        return None
    if isinstance(vector, list):
        return vector
    if isinstance(vector, dict):
        prepared: Dict[str, Any] = {}
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
    payload: Dict[str, Any], metadata: Dict[str, Any]
) -> Any:
    """Return the adjacency field from payload or metadata when available."""
    for key in (ADJACENCY_FIELD, "adjacent_ids", "neighbors", "edges"):
        value = payload.get(key)
        if value is None:
            value = metadata.get(key)
        if value is not None:
            return value
    return None


def _normalize_adjacency_ids(
    adjacency_raw: Any, limit: int
) -> Tuple[List[str], List[Any]]:
    """Normalize adjacency values into ordered ids and Qdrant-compatible ids."""
    if adjacency_raw is None or limit <= 0:
        return [], []

    if isinstance(adjacency_raw, str):
        tokens = [token.strip() for token in adjacency_raw.replace(";", ",").split(",")]
    elif isinstance(adjacency_raw, SequenceABC):
        tokens = list(adjacency_raw)
    else:
        tokens = [adjacency_raw]

    normalized_strings: List[str] = []
    qdrant_ids: List[Any] = []
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
) -> Dict[str, Dict[str, Any]]:
    """Fetch adjacency payloads from Qdrant once and expose them as dicts."""
    if not adjacency_ids:
        return {}

    unique_ids: List[Any] = []
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

    adjacency_map: Dict[str, Dict[str, Any]] = {}
    for record in records:
        payload = record.payload or {}
        metadata = payload.get("metadata") or {}
        key = _stringify_point_id(record.id)
        adjacency_entry: Dict[str, Any] = {
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


def _extract_query_points(results: Any) -> List[Any]:
    """Normalize Qdrant query results into a list of scored points."""
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
    """Ensure a tuple-wrapped point is unwrapped so it exposes payload/score."""
    if isinstance(point, tuple) and point:
        candidate = point[0]
        if hasattr(candidate, "payload") or isinstance(candidate, dict):
            return candidate
    return point

def query_embedding(query: str) -> List[float]:
    """Encode the user query into an embedding vector."""
    if not query:
        raise ValueError("query_embedding_step recebeu uma query vazia.")
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model.encode(query).tolist()


def retrieval_from_qdrant(
    query_embedding: List[float],
    top_k: int = DEFAULT_TOP_K,
    adjacency_limit: int = DEFAULT_ADJACENT_LIMIT,
) -> List[Dict[str, Any]]:
    """Retrieve the top-k chunks and their first-degree neighbors from Qdrant."""
    if not query_embedding:
        raise ValueError("retrieval_from_qdrant_step recebeu embedding vazio.")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    threshold = _parse_score_threshold()

    query_result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=INCLUDE_EMBEDDINGS,
        score_threshold=threshold,
    )
    scored_points = _extract_query_points(query_result)
    retrieved_nodes: List[Dict[str, Any]] = []
    adjacency_lookup: Dict[str, List[str]] = {}
    adjacency_requests: List[Any] = []

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
        raw_id = getattr(point, "id", None)
        if raw_id is None and isinstance(point, dict):
            raw_id = point.get("id")
        node_id = _stringify_point_id(raw_id)

        raw_score = getattr(point, "score", None)
        if raw_score is None and isinstance(point, dict):
            raw_score = point.get("score")
        entry: Dict[str, Any] = {
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

    if not retrieved_nodes:
        raise RuntimeError(
            "Nenhum contexto relevante foi retornado pela busca no Qdrant."
        )

    adjacency_data = _retrieve_adjacency_payloads(client, adjacency_requests)
    for entry in retrieved_nodes:
        adjacency_ids = adjacency_lookup.get(entry["id"], [])
        entry["adjacent"] = [
            adjacency_data[adj_id]
            for adj_id in adjacency_ids
            if adj_id in adjacency_data
        ]

    return retrieved_nodes


def _format_context_block(node: Dict[str, Any], index: int) -> str:
    """Helper to format a human-readable block for each node."""
    metadata = node.get("metadata") or {}
    header_parts = [f"Trecho {index}"]
    section = metadata.get("section") or metadata.get("section_context")
    if section:
        header_parts.append(f"seção: {section}")
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
    query_embedding: List[float],
    retrieved_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create the structured payload returned by the multi-agent tool."""
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


def rag_inference_pipeline(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    adjacency_limit: int = DEFAULT_ADJACENT_LIMIT,
) -> Dict[str, Any]:
    """Pipeline que encadeia embed + retrieval e retorna um grafo estruturado."""
    query_vector = query_embedding(query=query)
    retrieval = retrieval_from_qdrant(
        query_embedding=query_vector,
        top_k=top_k,
        adjacency_limit=adjacency_limit,
    )
    payload = build_graph_rag_payload(
        query=query,
        query_embedding=query_vector,
        retrieved_nodes=retrieval,
    )
    return payload



def retrieve_inteli_knowledge(
    query: str,
) -> Dict[str, Any]:
    """RAG tool that embeds a prompt and returns graph-based neighbors."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ValueError("retrieve_inteli_knowledge recebeu uma consulta vazia.")


    retrieval_payload = rag_inference_pipeline(
        query=normalized_query,
        top_k=DEFAULT_TOP_K,
        adjacency_limit=DEFAULT_ADJACENT_LIMIT,
    )

    state_entry = {
        "query": normalized_query,
        "top_k": DEFAULT_TOP_K,
        "adjacency_limit": DEFAULT_ADJACENT_LIMIT,
        "result_count": retrieval_payload.get("result_count", 0),
    }

    return {
        "success": True,
        "query": normalized_query,
        "result_count": retrieval_payload.get("result_count", 0),
        "chunks": retrieval_payload.get("results", []),
        "context": retrieval_payload.get("context", ""),
        "query_embedding": retrieval_payload.get("query_embedding"),
        "message": (
            f"Retornados {state_entry['result_count']} nós com até "
            f"{DEFAULT_ADJACENT_LIMIT} vizinhos por nó"
        ),
    }




def _format_node(node: Dict[str, Any], index: int) -> str:
    """Gera uma string legível com contexto e adjacências."""
    metadata = node.get("metadata") or {}
    header_parts: List[str] = [f"Trecho {index}"]
    section = metadata.get("section") or metadata.get("section_context")
    if section:
        header_parts.append(f"seção: {section}")
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
        lines.append("Adjacentes:")
        for adj in adjacency:
            adj_meta = adj.get("metadata") or {}
            adj_section = adj_meta.get("section") or adj_meta.get("section_context")
            label = adj_section or adj.get("id")
            snippet = adj.get("content") or ""
            snippet = textwrap.shorten(snippet, width=180, placeholder="…")
            lines.append(f"  - {label}: {snippet}")

    return "\n".join(lines)


def _create_adk_agent(model: str) -> Agent:
    """Instancia um agente simples do ADK focado em responder usando o contexto."""
    instruction = (
        "Você é um especialista do Inteli que responde perguntas usando APENAS o contexto"
        " recuperado de uma base vetorial. Se o contexto não trouxer detalhes suficientes,"
        " diga de forma direta que não possui informação disponível."
    )
    return Agent(
        name="rag_test_agent",
        model=model,
        description="Agente simples para validar o Graph RAG",
        instruction=instruction,
    )


def _build_llm_prompt(query: str, context_text: str) -> str:
    return textwrap.dedent(
        f"""
        Use o contexto abaixo para responder ao usuário. Cite trechos relevantes e mantenha o tom técnico e conciso.

        <contexto>
        {context_text}
        </contexto>

        Pergunta: {query}

        Se não houver dados suficientes, responda que as informações não estão disponíveis nesta base.
        """
    ).strip()


async def _run_adk_completion(agent: Agent, prompt: str, app_name: str) -> str:
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    events = await runner.run_debug(
        user_messages=prompt,
        user_id="rag_test_user",
        session_id="rag_test_session",
        quiet=True,
    )

    response_parts: List[str] = []
    for event in events:
        content = getattr(event, "content", None)
        if not content:
            continue
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                response_parts.append(text)

    return "".join(response_parts).strip()


def _generate_adk_answer(
    query: str, payload: Dict[str, Any], model: str, app_name: str
) -> str:
    context_text = payload.get("context", "")
    if not context_text:
        return "Não foi possível montar o contexto recuperado para enviar ao modelo."

    agent = _create_adk_agent(model)
    prompt = _build_llm_prompt(query, context_text)
    try:
        return asyncio.run(_run_adk_completion(agent, prompt, app_name))
    except Exception as exc:  # pragma: no cover - integração externa
        return f"[Falha ao chamar o modelo do Google ADK: {exc}]"


def _strip_embeddings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove embedding vectors from nodes/adjacencies before printing."""
    for node in payload.get("results", []) or []:
        node.pop("embedding", None)
        for adj in node.get("adjacent") or []:
            adj.pop("embedding", None)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa o Graph RAG e consulta o modelo do Google ADK usando o contexto recuperado."
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="Pergunta ou texto usado como entrada do RAG.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Quantidade de nós similares retornados (padrão usa RAG_TOP_K).",
    )
    parser.add_argument(
        "--adjacency-limit",
        type=int,
        default=None,
        help="Número de vizinhos imediatos por nó (padrão usa RAG_ADJACENT_LIMIT).",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=600,
        help="Limite de caracteres da resposta heurística de fallback.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ADK_MODEL,
        help="Modelo usado pelo Google ADK (ex.: gemini-2.0-flash-exp).",
    )
    parser.add_argument(
        "--app-name",
        default=DEFAULT_ADK_APP,
        help="Nome do aplicativo usado pelo InMemoryRunner/ADK.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Mostra o payload completo em JSON (útil para debug).",
    )

    args = parser.parse_args()

    payload = retrieve_inteli_knowledge(
        query=args.query,
    )
    payload = _strip_embeddings(payload)

    adk_answer = _generate_adk_answer(args.query, payload, args.model, args.app_name)

    print("\n================ RAG QUERY ===============")
    print(f"Query: {args.query}")
    print(f"Resultados: {payload['result_count']}")

    for idx, node in enumerate(payload.get("results", []), start=1):
        print("\n----------------------------------------")
        print(_format_node(node, idx))

    if args.show_json:
        print("\n================ RAW JSON ===============")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n================ RESPOSTA (GOOGLE ADK) ===============")
    print(adk_answer)


if __name__ == "__main__":  # pragma: no cover - script CLI
    main()
