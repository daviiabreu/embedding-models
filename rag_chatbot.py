#!/usr/bin/env python3
"""
Simple Retrieval-Augmented Generation (RAG) chatbot.

Uses Qdrant as the vector store and the Google AI SDK (Gemini) as the
generative model. Requires a pre-populated Qdrant collection containing
the embeddings produced for the admission notice document.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

SYSTEM_INSTRUCTION = (
    "Você é o assistente virtual do processo seletivo do Inteli. "
    "Responda sempre em português claro, usando apenas as informações "
    "apresentadas no contexto fornecido. Quando não houver dados suficientes, "
    "admita isso explicitamente."
)


def load_environment() -> None:
    """
    Load environment variables from `.env` if available.

    The script is typically executed from the project root, but we call
    `load_dotenv()` regardless to keep behaviour predictable.
    """
    load_dotenv()


@dataclass
class RetrievedChunk:
    """Payload returned by a Qdrant similarity search."""

    content: str
    score: float
    section: Optional[str]
    page_number: Optional[int]
    chunk_id: Optional[str]


class RagChatbot:
    """Encapsulates the RAG flow: retrieval + generative answer."""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        embedding_model_name: str,
        google_api_key: str,
        google_model_name: str,
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.embedder = SentenceTransformer(embedding_model_name)
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        genai.configure(api_key=google_api_key)
        self.generative_model = genai.GenerativeModel(
            google_model_name,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        query_vector = self.embedder.encode(question).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            with_payload=True,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )

        chunks: List[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            content = payload.get("content")
            if not content:
                # Skip chunks missing the textual content we want to show the LLM.
                continue

            metadata = payload.get("metadata") or {}
            chunks.append(
                RetrievedChunk(
                    content=content,
                    score=result.score,
                    section=metadata.get("section"),
                    page_number=metadata.get("page_number"),
                    chunk_id=payload.get("chunk_id"),
                )
            )

        return chunks

    def build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Cria um prompt string simples, evitando estruturação complexa de 'parts'."""
        context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            meta = []
            if chunk.section:
                meta.append(f"seção: {chunk.section}")
            if chunk.page_number is not None:
                meta.append(f"página: {chunk.page_number}")
            if chunk.chunk_id:
                meta.append(f"id: {chunk.chunk_id}")
            header = f"Trecho {idx}" + (f" ({', '.join(meta)})" if meta else "")
            context_blocks.append(f"{header}\n{chunk.content}")

        context_text = "\n\n".join(context_blocks) if context_blocks else "Nenhum contexto encontrado."

        return (
            "Use exclusivamente os trechos relevantes abaixo para responder à pergunta.\n\n"
            f"Contexto:\n{context_text}\n\n"
            f"Pergunta: {question}\n\n"
            "Se não houver dados suficientes, diga isso explicitamente e, quando possível, cite seção/página."
        )


    def answer(self, question: str) -> str:
        chunks = self.retrieve(question)
        if not chunks:
            raise RuntimeError(
                "Nenhum contexto relevante foi encontrado no Qdrant para essa pergunta."
            )

        user_prompt = self.build_prompt(question, chunks)

        # Chamada ao modelo (mantém a system_instruction configurada no construtor)
        try:
            response = self.generative_model.generate_content(user_prompt)
        except Exception as exc:  # API falhou antes de retornar objeto de resposta
            api_detail = ""
            status = getattr(exc, "response", None)
            if status is not None:
                status_text = getattr(status, "text", None)
                if status_text:
                    api_detail = f" | Resposta da API: {status_text[:200]}"
            raise RuntimeError(
                "Erro na chamada ao modelo Gemini. "
                "Confirme GOOGLE_API_KEY e conectividade; atualize o pacote google-generativeai."
                f" Detalhes: {exc}{api_detail}"
            ) from exc

        # Extração robusta do texto
        text, debug_notes = self._extract_text_from_response(response)
        if text:
            return text

        feedback = self._format_prompt_feedback(response)
        debug_str = "; ".join(debug_notes) if debug_notes else "sem detalhes adicionais"
        response_excerpt = repr(response)
        if len(response_excerpt) > 240:
            response_excerpt = response_excerpt[:240] + "...<truncado>"
        raise RuntimeError(
            "O modelo não retornou texto na resposta. "
            f"Debug: {debug_str}. {feedback} Objeto bruto: {response_excerpt}"
        )

    @staticmethod
    def _extract_text_from_response(response: object) -> Tuple[str, List[str]]:
        """Tenta extrair texto do objeto de resposta, anotando o caminho percorrido."""
        debug_notes: List[str] = []

        # 1) Tentar via propriedade text (mais comum)
        try:
            text = getattr(response, "text")
            if text:
                return text.strip(), debug_notes
            debug_notes.append("response.text vazio")
        except AttributeError:
            debug_notes.append("response não possui atributo 'text'")
        except Exception as exc:  # pylint: disable=broad-except
            debug_notes.append(f"falha ao ler response.text: {exc}")

        # 2) Inspeciona candidatos
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts_texts: List[str] = []
            for idx, cand in enumerate(candidates, start=1):
                content = getattr(cand, "content", None)
                if content is None:
                    cand_text = getattr(cand, "output_text", None)
                    if cand_text:
                        parts_texts.append(str(cand_text))
                        continue
                    debug_notes.append(f"candidate[{idx}] sem content/output_text")
                    continue

                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        text_part = getattr(part, "text", None)
                        if text_part:
                            parts_texts.append(str(text_part))
                else:
                    cont_text = getattr(content, "text", None)
                    if cont_text:
                        parts_texts.append(str(cont_text))

            joined = "\n".join(t.strip() for t in parts_texts if t)
            if joined:
                return joined.strip(), debug_notes
            debug_notes.append("candidates presentes, porém sem texto extraível")
        else:
            debug_notes.append("resposta não possui candidates")

        # 3) Fallback: tentar acessar como dict-like
        if isinstance(response, dict):
            choice_texts = []
            for key in ("text", "output_text", "generated_text"):
                value = response.get(key)
                if value:
                    choice_texts.append(str(value))
            if choice_texts:
                return "\n".join(choice_texts).strip(), debug_notes
            debug_notes.append("dict de resposta sem campos text/output_text")

        return "", debug_notes

    @staticmethod
    def _format_prompt_feedback(response: object) -> str:
        """Extrai feedback de segurança ou motivos de bloqueio se disponíveis."""
        feedback = getattr(response, "prompt_feedback", None)
        if not feedback:
            return ""

        # prompt_feedback pode ter atributos como block_reason, safety_ratings
        fragments = []
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            fragments.append(f"Motivo do bloqueio: {block_reason}")

        safety_ratings = getattr(feedback, "safety_ratings", None)
        if safety_ratings:
            ratings_desc = []
            for rating in safety_ratings:
                category = getattr(rating, "category", None)
                probability = getattr(rating, "probability", None)
                if category or probability:
                    ratings_desc.append(f"{category}: {probability}")
            if ratings_desc:
                fragments.append("Safety ratings: " + ", ".join(ratings_desc))

        structured = getattr(feedback, "feedback", None)
        if structured:
            fragments.append(f"Feedback adicional: {structured}")

        return ("Feedback do modelo: " + " | ".join(fragments) + ". ") if fragments else ""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chatbot RAG que combina Qdrant e Google AI SDK (Gemini)."
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Pergunta a ser respondida. Se omitido, o chatbot executa em modo interativo.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Número de trechos retornados pelo Qdrant (sobrepõe variável de ambiente RAG_TOP_K).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Pontuação mínima dos resultados (sobrepõe variável de ambiente RAG_SCORE_THRESHOLD).",
    )
    return parser.parse_args()


def build_chatbot(args: argparse.Namespace) -> RagChatbot:
    load_environment()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        sys.exit("Configure QDRANT_URL e QDRANT_API_KEY no arquivo .env antes de usar o chatbot.")
    if not google_api_key:
        sys.exit("Configure GOOGLE_API_KEY no arquivo .env antes de usar o chatbot.")

    collection = os.getenv("QDRANT_COLLECTION", "inteli_admission_chunks")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    google_model_name = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

    top_k_env = int(os.getenv("RAG_TOP_K", "4"))
    score_threshold_env = os.getenv("RAG_SCORE_THRESHOLD")
    score_threshold = float(score_threshold_env) if score_threshold_env else None

    top_k = args.top_k if args.top_k is not None else top_k_env
    threshold = args.score_threshold if args.score_threshold is not None else score_threshold

    return RagChatbot(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection,
        embedding_model_name=embedding_model_name,
        google_api_key=google_api_key,
        google_model_name=google_model_name,
        top_k=top_k,
        score_threshold=threshold,
    )


def interactive_loop(chatbot: RagChatbot) -> None:
    print("Chatbot RAG (digite 'sair' para encerrar)")
    while True:
        try:
            question = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break

        try:
            answer = chatbot.answer(question)
        except Exception as exc:
            print(f"Erro ao gerar resposta: {exc}")
            continue

        print(f"Chatbot: {answer}\n")


def main() -> None:
    args = parse_args()
    chatbot = build_chatbot(args)

    if args.question:
        try:
            answer = chatbot.answer(args.question)
        except Exception as exc:
            sys.exit(f"Erro ao gerar resposta: {exc}")
        print(answer)
    else:
        interactive_loop(chatbot)


if __name__ == "__main__":
    main()
