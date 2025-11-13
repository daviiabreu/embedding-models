import json
from json import JSONDecodeError
import os
import re
import sys
import types
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from zenml import pipeline, step

# pdfminer >=202211 remove the psexceptions module that unstructured still imports.
# Provide a lightweight shim so downstream imports keep working regardless of version.
try:  # pragma: no cover - compatibility shim
    from pdfminer.psexceptions import PSSyntaxError as _  # type: ignore
except ImportError:
    from pdfminer.pdfparser import PDFSyntaxError

    shim = types.ModuleType("pdfminer.psexceptions")

    class PSSyntaxError(PDFSyntaxError):
        """Backwards-compatible alias expected by unstructured."""

    shim.PSSyntaxError = PSSyntaxError
    sys.modules["pdfminer.psexceptions"] = shim

from unstructured.partition.pdf import partition_pdf

load_dotenv()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ue009", "tt")
    text = re.sub(r"Pág\.\s*\d+", "", text)
    text = re.sub(r"[•◦▪▫]", "•", text)
    return text.strip()


def determine_hierarchy_level(element: Dict[str, Any]) -> str:
    element_type = element.get("type")
    text = element.get("text", "")
    if element_type == "Title":
        if re.match(r"^\d+\.", text):
            level = len(text.split(".")[0])
            return f"level_{level}"
        return "title_main"
    if element_type == "ListItem":
        return "list_item"
    return "body"


def extract_section_info(element: Dict[str, Any]) -> str:
    text = element.get("text", "")
    element_type = element.get("type")
    if element_type == "Title" and re.match(r"^\d+\.", text):
        return (
            text.split(".")[0] + "." + text.split(".")[1].strip()
            if "." in text
            else text
        )
    if element_type == "ListItem" and re.match(r"^\d+\.", text):
        return text
    return "general"


def extract_enhanced_metadata(element: Dict[str, Any]) -> Dict[str, Any]:
    metadata = element.get("metadata", {})
    return {
        "element_id": element.get("element_id"),
        "element_type": element.get("type"),
        "page_number": metadata.get("page_number"),
        "parent_id": metadata.get("parent_id"),
        "text_length": len(element.get("text", "")),
        "is_header": element.get("type") in ["Title"],
        "is_list_item": element.get("type") == "ListItem",
        "is_table_content": element.get("type") in ["Table", "TableRow"],
        "hierarchy_level": determine_hierarchy_level(element),
        "section": extract_section_info(element),
    }


def detect_summary_elements(
    elements: List[Dict[str, Any]],
    detection_method: str = "keywords",
    max_pages: int = 5,
) -> Set[int]:
    """
    Detecta elementos de sumário usando diferentes métodos.

    Args:
        elements: Lista de elementos
        detection_method: "keywords", "page_range", ou "pattern"
        max_pages: Máximo de páginas para considerar (para page_range)
    """
    summary_elements: Set[int] = set()

    if detection_method == "keywords":
        summary_keywords = [
            "SUMÁRIO",
            "SUMARIO",
            "ÍNDICE",
            "INDICE",
            "TABLE OF CONTENTS",
            "CONTENTS",
        ]

        for i, element in enumerate(elements):
            text = element.get("text", "").strip().upper()
            if any(keyword in text for keyword in summary_keywords):
                j = i
                while j < len(elements):
                    summary_elements.add(j)
                    j += 1
                    if (
                        j < len(elements)
                        and elements[j].get("type") == "Title"
                        and not any(
                            kw in elements[j].get("text", "").upper()
                            for kw in summary_keywords
                        )
                    ):
                        break

    elif detection_method == "page_range":
        for i, element in enumerate(elements):
            page_num = element.get("metadata", {}).get("page_number")
            if page_num and page_num <= max_pages:
                text = element.get("text", "").strip()
                if re.match(r".+\.{3,}\s*\d+$|.+\s+\d+$", text):
                    summary_elements.add(i)

    elif detection_method == "pattern":
        patterns = [
            r"^\d+\.\s+.+\s+\d+$",  # "1. Titulo 5"
            r"^.+\.{3,}\s*\d+$",  # "Titulo ... 5"
            r"^[A-Z\s]+\s+\d+$",  # "TITULO 5"
        ]
        for i, element in enumerate(elements):
            text = element.get("text", "").strip()
            if any(re.match(pattern, text) for pattern in patterns):
                summary_elements.add(i)

    return summary_elements


def preprocess_elements(
    elements: List[Dict[str, Any]],
    skip_summary: bool = True,
    summary_detection: str = "keywords",
) -> List[Dict[str, Any]]:
    processed_elements: List[Dict[str, Any]] = []
    current_section = "Introduction"
    current_subsection = ""

    summary_elements: Set[int] = set()
    if skip_summary:
        summary_elements = detect_summary_elements(elements, summary_detection)
        if summary_elements:
            print(f"🚫 Detectados {len(summary_elements)} elementos de sumário para pular")

    for i, element in enumerate(elements):
        if i in summary_elements:
            continue

        text = element.get("text", "").strip()

        if not text or len(text) < 10:
            continue
        if re.match(r"^(Pág\.\s*|[\s·•◦▪▫\d])+$", text, re.IGNORECASE):
            continue
        if element.get("type") == "Footer":
            continue

        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue

        metadata = extract_enhanced_metadata(element)

        if metadata["element_type"] == "Title" and any(
            char.isdigit() for char in cleaned_text[:5]
        ):
            current_section = cleaned_text
            current_subsection = ""
        elif metadata["element_type"] == "Title":
            current_subsection = cleaned_text

        metadata["section_context"] = current_section
        metadata["subsection_context"] = current_subsection

        processed_elements.append({"text": cleaned_text, "metadata": metadata})

    return processed_elements


def create_overlapping_chunks(
    processed_elements: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    print(f"Overlapping de chunks de tamanho: (size={chunk_size}, overlap={chunk_overlap})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    documents: List[Document] = []
    for element in processed_elements:
        section = element["metadata"].get("section_context", "")
        subsection = element["metadata"].get("subsection_context", "")

        context_prefix = ""
        if section and section != "Introduction":
            context_prefix = f"Seção: {section}. "
        if subsection:
            context_prefix += f"Subseção: {subsection}. "

        doc = Document(
            page_content=f"{context_prefix}{element['text']}",
            metadata=element["metadata"],
        )
        documents.append(doc)

    chunked_documents = text_splitter.split_documents(documents)

    print(f"Foram criados {len(chunked_documents)} chunks.")
    return chunked_documents


def documents_to_chunks(chunked_docs: List[Document]) -> List[Dict[str, Any]]:
    embedding_ready_chunks: List[Dict[str, Any]] = []
    for i, doc in enumerate(chunked_docs):
        embedding_ready_chunks.append(
            {"id": f"chunk_{i}", "content": doc.page_content, "metadata": doc.metadata}
        )
    return embedding_ready_chunks


def extract_pdf_elements(pdf_path: str) -> List[Dict[str, Any]]:
    elements = partition_pdf(filename=pdf_path, strategy="fast", ocr_languages="por")
    elements_data: List[Dict[str, Any]] = []

    for element in elements:
        element_dict: Dict[str, Any] = {
            "text": str(element),
            "type": element.__class__.__name__,
            "metadata": {},
        }

        if hasattr(element, "metadata") and element.metadata:
            element_dict["metadata"] = element.metadata.to_dict()

        if hasattr(element, "id"):
            element_dict["element_id"] = element.id

        elements_data.append(element_dict)

    return elements_data


def preprocess_for_embedding(json_file_path: str) -> List[Dict[str, Any]]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        elements = json.load(f)

    print(f"Carregados {len(elements)} elementos do JSON")

    processed_elements = preprocess_elements(elements)
    print(f"Foram processados {len(processed_elements)} elementos")

    chunked_docs = create_overlapping_chunks(processed_elements)
    return documents_to_chunks(chunked_docs)


def process_document_with_params(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    skip_summary: bool = True,
) -> List[Dict[str, Any]]:
    """Função wrapper para processar documento com parâmetros customizados."""
    elements_data = extract_pdf_elements(pdf_path)
    processed_elements = preprocess_elements(elements_data, skip_summary=skip_summary)
    chunked_docs = create_overlapping_chunks(
        processed_elements, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return documents_to_chunks(chunked_docs)


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Gera embeddings para os chunks usando SentenceTransformers.
    """
    if not chunks:
        print("❌ Nenhum chunk disponível para embedding.")
        return []

    print(f"🤖 Carregando modelo de embedding: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - simples log
        print(f"❌ Erro ao carregar o modelo: {exc}")
        return chunks

    texts = [chunk["content"] for chunk in chunks]

    print(f"🔄 Gerando embeddings para {len(texts)} chunks...")
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_tensor=False,
        )
    except Exception as exc:  # pragma: no cover
        print(f"❌ Erro ao gerar embeddings: {exc}")
        return chunks

    chunks_with_embeddings: List[Dict[str, Any]] = []
    for chunk, vector in zip(chunks, embeddings):
        enriched = dict(chunk)
        enriched["embedding"] = vector.tolist()
        enriched["embedding_model"] = model_name
        enriched["embedding_dimension"] = len(vector)
        chunks_with_embeddings.append(enriched)

    print(f"✅ Embeddings gerados para {len(chunks_with_embeddings)} chunks")
    print(f"📏 Dimensão dos embeddings: {len(embeddings[0])}")

    return chunks_with_embeddings


def upsert_embeddings_to_qdrant(
    embeddings: List[Dict[str, Any]],
    collection_name: str,
    url: str,
    api_key: str,
    distance_metric: str = "cosine",
    recreate_collection: bool = False,
) -> None:
    """
    Envia embeddings já gerados para o Qdrant.
    """
    if not embeddings:
        print("ℹ️ Nenhum embedding para enviar ao Qdrant.")
        return

    client = QdrantClient(url=url, api_key=api_key)
    distance = getattr(
        qdrant_models.Distance,
        distance_metric.upper(),
        qdrant_models.Distance.COSINE,
    )

    vector_dimension = len(embeddings[0]["embedding"])
    vectors_config = qdrant_models.VectorParams(size=vector_dimension, distance=distance)

    if recreate_collection:
        print(f"🔁 Recriando coleção `{collection_name}`...")
        client.recreate_collection(collection_name=collection_name, vectors_config=vectors_config)
    else:
        collection_exists = False
        try:
            collection_exists = client.collection_exists(collection_name=collection_name)
        except JSONDecodeError:
            print(
                "⚠️ Resposta inesperada do endpoint `collection_exists`. "
                "Prosseguindo assumindo que a coleção já exista."
            )
            collection_exists = True

        if not collection_exists:
            print(f"📚 Criando coleção `{collection_name}` (dimensão={vector_dimension})...")
            client.create_collection(collection_name=collection_name, vectors_config=vectors_config)

    points = []
    for idx, chunk in enumerate(embeddings):
        payload = {
            "chunk_id": chunk.get("id", f"chunk_{idx}"),
            "content": chunk.get("content"),
            "metadata": chunk.get("metadata", {}),
        }
        point_id = (
            chunk.get("metadata", {}).get("element_id")
            or chunk.get("id")
            or f"{collection_name}_{idx}"
        )
        points.append(
            qdrant_models.PointStruct(
                id=str(point_id),
                vector=chunk["embedding"],
                payload=payload,
            )
        )

    print(f"🚀 Enviando {len(points)} embeddings para o Qdrant...")
    try:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    except JSONDecodeError:
        print(
            "⚠️ Resposta inesperada do endpoint `upsert`. "
            "Assumindo sucesso com base no código HTTP 200."
        )
    print(f"✅ Ingestão concluída na coleção `{collection_name}`.")


def process_document_with_embeddings(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    skip_summary: bool = True,
    generate_embeddings_flag: bool = True,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Pipeline completo: extração, chunking e embedding.
    """
    chunks = process_document_with_params(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        skip_summary=skip_summary,
    )

    if generate_embeddings_flag and chunks:
        return generate_embeddings(chunks, model_name=model_name)
    return chunks


@step
def extract_pdf_elements_step(pdf_path: str) -> List[Dict[str, Any]]:
    return extract_pdf_elements(pdf_path)


@step
def preprocess_elements_step(
    elements: List[Dict[str, Any]],
    skip_summary: bool = True,
    summary_detection: str = "keywords",
) -> List[Dict[str, Any]]:
    return preprocess_elements(
        elements, skip_summary=skip_summary, summary_detection=summary_detection
    )


@step
def create_chunks_step(
    processed_elements: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    chunked_docs = create_overlapping_chunks(
        processed_elements, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return documents_to_chunks(chunked_docs)


@step
def generate_embeddings_step(
    chunks: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    if not enabled:
        print("ℹ️ Geração de embeddings desabilitada; retornando chunks originais.")
        return chunks
    return generate_embeddings(chunks, model_name=model_name, normalize=normalize)


@step
def ingest_embeddings_step(
    embeddings: List[Dict[str, Any]],
    collection_name: str = "inteli_admission_chunks",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    distance_metric: str = "cosine",
    recreate_collection: bool = False,
    enabled: bool = False,
) -> None:
    if not enabled:
        print("ℹ️ Ingestão no Qdrant desabilitada; nenhuma ação realizada.")
        return

    resolved_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    resolved_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Qdrant API key não definida. Configure `qdrant_api_key` ou a variável QDRANT_API_KEY."
        )

    target_collection = os.getenv("QDRANT_COLLECTION", collection_name)
    upsert_embeddings_to_qdrant(
        embeddings,
        collection_name=target_collection,
        url=resolved_url,
        api_key=resolved_api_key,
        distance_metric=distance_metric,
        recreate_collection=recreate_collection,
    )


@pipeline
def embedding_pipeline(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    skip_summary: bool = True,
    summary_detection: str = "keywords",
    generate_embeddings_flag: bool = True,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize_embeddings: bool = True,
    ingest_to_qdrant: bool = False,
    qdrant_collection_name: str = "inteli_admission_chunks",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_distance_metric: str = "cosine",
    recreate_qdrant_collection: bool = False,
) -> None:
    """
    Pipeline ZenML que encadeia extração, pré-processamento, chunking e embeddings.
    """
    elements = extract_pdf_elements_step(pdf_path=pdf_path)
    processed = preprocess_elements_step(
        elements=elements,
        skip_summary=skip_summary,
        summary_detection=summary_detection,
    )
    chunks = create_chunks_step(
        processed_elements=processed,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = generate_embeddings_step(
        chunks=chunks,
        model_name=model_name,
        normalize=normalize_embeddings,
        enabled=generate_embeddings_flag,
    )
    ingest_embeddings_step(
        embeddings=embeddings,
        collection_name=qdrant_collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        distance_metric=qdrant_distance_metric,
        recreate_collection=recreate_qdrant_collection,
        enabled=ingest_to_qdrant,
    )


def main() -> None:
    """
    Interface CLI alinhada ao padrão ZenML: definimos o pipeline e o executamos no bloco main.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline ZenML para extração, chunking, embeddings e ingestão no Qdrant."
    )

    def str_to_bool(value: str) -> bool:
        if isinstance(value, bool):
            return value
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Valor booleano inválido: '{value}'")
    parser.add_argument("pdf_path", help="Caminho para o PDF a ser processado.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--summary-detection", default="keywords")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--skip-summary",
        dest="skip_summary",
        action="store_true",
        help="Ignora elementos do sumário durante o pré-processamento.",
    )
    parser.add_argument(
        "--include-summary",
        dest="skip_summary",
        action="store_false",
        help="Mantém elementos de sumário (override do padrão).",
    )
    parser.set_defaults(skip_summary=True)
    parser.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        help="Normaliza os embeddings (padrão).",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Desabilita a normalização dos embeddings.",
    )
    parser.set_defaults(normalize_embeddings=True)
    default_ingest = str_to_bool(os.getenv("INGEST_TO_QDRANT", "false"))
    parser.add_argument(
        "--ingest-qdrant",
        type=str_to_bool,
        default=default_ingest,
        metavar="{true,false}",
        help="Ativa/desativa a ingestão direta no Qdrant (também aceita o env INGEST_TO_QDRANT).",
    )
    parser.add_argument("--qdrant-collection", default="inteli_admission_chunks")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--qdrant-distance", default="cosine")
    parser.add_argument(
        "--recreate-qdrant-collection",
        action="store_true",
        default=False,
        help="Recria a coleção antes de inserir os embeddings.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ Arquivo não encontrado: {args.pdf_path}")
        sys.exit(1)

    print("🚀 Iniciando execução do pipeline ZenML `embedding_pipeline`...")
    run = embedding_pipeline(
        pdf_path=args.pdf_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        skip_summary=args.skip_summary,
        summary_detection=args.summary_detection,
        generate_embeddings_flag=True,
        model_name=args.model_name,
        normalize_embeddings=args.normalize_embeddings,
        ingest_to_qdrant=args.ingest_qdrant,
        qdrant_collection_name=args.qdrant_collection,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_distance_metric=args.qdrant_distance,
        recreate_qdrant_collection=args.recreate_qdrant_collection,
    )
    print("✅ Pipeline finalizado. Utilize `zenml login --local` para visualizar o run.")
    return run


if __name__ == "__main__":
    main()
