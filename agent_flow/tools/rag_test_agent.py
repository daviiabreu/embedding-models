"""CLI utilitário para testar o Graph RAG usando o modelo do Google ADK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

from .rag_tool import run_graph_rag_tool

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "agent_flow" / ".env")


DEFAULT_ADK_MODEL = os.getenv(
    "DEFAULT_MODEL", os.getenv("ADK_MODEL")
)
DEFAULT_ADK_APP = os.getenv("RAG_TEST_APP", "agents")


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


def _build_naive_answer(query: str, payload: Dict[str, Any], max_chars: int) -> str:
    """Gera uma resposta heurística apenas concatenando os contextos retornados."""
    _ = query  # Mantido para facilitar futuras personalizações
    snippets: List[str] = []
    for idx, node in enumerate(payload.get("results", []), start=1):
        snippet = node.get("content") or ""
        snippets.append(f"[Trecho {idx}] {snippet}")
        for adj in node.get("adjacent") or []:
            snippets.append(f"[Adjacente {idx}] {adj.get('content')}")

    if not snippets:
        return "Nenhum contexto disponível para gerar resposta."

    joined = " ".join(snippets)
    return textwrap.shorten(joined, width=max_chars, placeholder="…")


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

    payload = run_graph_rag_tool(
        query=args.query,
        top_k=args.top_k,
        adjacency_limit=args.adjacency_limit,
    )

    print("\n================ RAG QUERY ===============")
    print(f"Query: {args.query}")
    print(f"Resultados: {payload['result_count']}")

    for idx, node in enumerate(payload.get("results", []), start=1):
        print("\n----------------------------------------")
        print(_format_node(node, idx))

    print("\n================ RESPOSTA (GOOGLE ADK) ===============")
    print(_generate_adk_answer(args.query, payload, args.model, args.app_name))

    print("\n================ RESPOSTA HEURÍSTICA ===============")
    print(_build_naive_answer(args.query, payload, args.max_answer_chars))

    if args.show_json:
        print("\n================ RAW JSON ===============")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - script CLI
    main()
