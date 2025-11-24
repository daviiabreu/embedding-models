import os
import re
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pdfminer >=202211 remove the psexceptions module that unstructured still imports.
# Provide a lightweight shim so downstream imports keep working regardless of version.
try:  # pragma: no cover - compatibility shim
    from pdfminer.psexceptions import PSSyntaxError as _  # type: ignore
except ImportError:
    import sys as _sys
    import types

    from pdfminer.pdfparser import PDFSyntaxError

    shim = types.ModuleType("pdfminer.psexceptions")

    class PSSyntaxError(PDFSyntaxError):
        """Backwards-compatible alias expected by unstructured."""

    shim.PSSyntaxError = PSSyntaxError
    _sys.modules["pdfminer.psexceptions"] = shim

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from zenml import pipeline, step

load_dotenv()
load_dotenv("agent_flow/.env", override=False)

EMBEDDING_MODEL = os.getenv("EMBEDDINGS_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


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
    detection_method: str,
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


def _extract_txt_elements(file_path: Path) -> List[Dict[str, Any]]:
    """Transforma um arquivo .txt em lista de elementos compatíveis com o pipeline."""
    content = file_path.read_text(encoding="utf-8")
    segments = [segment.strip() for segment in content.split("\n\n") if segment.strip()]
    if not segments:
        segments = [content.strip()] if content.strip() else []

    elements_data: List[Dict[str, Any]] = []
    for idx, segment in enumerate(segments, start=1):
        if not segment:
            continue
        elements_data.append(
            {
                "text": segment,
                "type": "TextSegment",
                "metadata": {
                    "source_file": file_path.name,
                    "segment_index": idx,
                },
                "element_id": f"txt_{file_path.stem}_{idx}",
            }
        )

    return elements_data


def _extract_pdf_elements(pdf_path: str) -> List[Dict[str, Any]]:
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


@step
def extract_file_elements(pdf_path: str) -> List[Dict[str, Any]]:
    file_path = Path(pdf_path)
    if file_path.suffix.lower() == ".txt":
        return _extract_txt_elements(file_path)
    elif file_path.suffix.lower() == ".pdf":
        return _extract_pdf_elements(file_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {file_path.suffix}")


@step
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
            print(f"Detectados {len(summary_elements)} elementos de sumário para pular")

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


@step
def create_chunks(
    processed_elements: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    print(
        f"Overlapping de chunks de tamanho: (size={chunk_size}, overlap={chunk_overlap})..."
    )

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

    chunk_dicts: List[Dict[str, Any]] = []
    for idx, doc in enumerate(chunked_documents):
        chunk_dicts.append(
            {
                "id": f"chunk_{idx}",
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    print(f"Foram criados {len(chunk_dicts)} chunks.")
    return chunk_dicts


@step
def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = EMBEDDING_MODEL,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Gera embeddings para os chunks usando SentenceTransformers.
    """
    print("\nOi")
    if not chunks:
        print(" Nenhum chunk disponível para embedding.")
        return []

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - simples log
        print(f"Erro ao carregar o modelo: {exc}")
        return chunks

    texts = [chunk["content"] for chunk in chunks]

    print(f"Gerando embeddings para {len(texts)} chunks...")
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_tensor=False,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Erro ao gerar embeddings: {exc}")
        return chunks

    chunks_with_embeddings: List[Dict[str, Any]] = []
    for chunk, vector in zip(chunks, embeddings):
        enriched = dict(chunk)
        enriched["embedding"] = vector.tolist()
        enriched["embedding_model"] = model_name
        enriched["embedding_dimension"] = len(vector)
        chunks_with_embeddings.append(enriched)

    print(f" Embeddings gerados para {len(chunks_with_embeddings)} chunks")

    return chunks_with_embeddings


@step
def ingest_embeddings(
    embeddings: List[Dict[str, Any]],
    collection_name: str,
    distance_metric: str = "cosine",
    recreate_collection: bool = False,
    batch_size: int = 64,
) -> None:
    """
    Envia embeddings já gerados para o Qdrant.
    """
    if not embeddings:
        print(" Nenhum embedding para enviar ao Qdrant.")
        return
    if not QDRANT_API_KEY:
        raise ValueError("Qdrant API key não definida no .env (QDRANT_API_KEY).")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
    distance = getattr(
        qdrant_models.Distance,
        distance_metric.upper(),
        qdrant_models.Distance.COSINE,
    )

    vector_dimension = len(embeddings[0]["embedding"])
    vectors_config = qdrant_models.VectorParams(
        size=vector_dimension, distance=distance
    )

    if recreate_collection:
        print(f" Recriando coleção `{collection_name}`...")
        client.recreate_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )
    else:
        collection_exists = False
        try:
            collection_exists = client.collection_exists(
                collection_name=collection_name
            )
        except JSONDecodeError:
            print("Resposta inesperada do endpoint `collection_exists`. ")
            collection_exists = True

        if not collection_exists:
            print(
                f" Criando coleção `{collection_name}` (dimensão={vector_dimension})..."
            )
            client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )

    def _batched(
        seq: List[qdrant_models.PointStruct], size: int
    ) -> List[qdrant_models.PointStruct]:
        for start in range(0, len(seq), size):
            yield seq[start : start + size]

    points: List[qdrant_models.PointStruct] = []
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

    print(
        f"Enviando {len(points)} embeddings para o Qdrant em lotes de {batch_size}..."
    )
    total_sent = 0
    for batch in _batched(points, batch_size):
        try:
            client.upsert(collection_name=collection_name, points=batch, wait=True)
            total_sent += len(batch)
        except JSONDecodeError:
            print("Resposta inesperada do endpoint `upsert`. Prosseguindo.")

    print(
        f" Ingestão concluída na coleção `{collection_name}` ({total_sent} embeddings enviados)."
    )


@pipeline
def embedding_pipeline(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    skip_summary: bool = True,
    normalize_embeddings: bool = True,
    qdrant_collection_name: str = "inteli_documents_chunks",
    recreate_qdrant_collection: bool = False,
) -> None:
    """
    Pipeline ZenML que encadeia extração, pré-processamento, chunking e embeddings.
    """
    elements = extract_file_elements(pdf_path=pdf_path)
    processed = preprocess_elements(
        elements=elements,
        skip_summary=skip_summary,
    )
    chunks = create_chunks(
        processed_elements=processed,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(" Gerando embeddings para os chunks...")
    embeddings = generate_embeddings(
        chunks=chunks,
        normalize=normalize_embeddings,
    )
    ingest_embeddings(
        embeddings=embeddings,
        collection_name=qdrant_collection_name,
        recreate_collection=recreate_qdrant_collection,
    )


def main() -> None:
    """
    Interface CLI alinhada ao padrão ZenML: definimos o pipeline e o executamos no bloco main.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline ZenML para extração, chunking, embeddings e ingestão no Qdrant."
    )

    parser.add_argument("pdf_path", help="Caminho para o PDF a ser processado.")
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Desabilita a normalização dos embeddings.",
    )
    parser.set_defaults(normalize_embeddings=True)
    parser.add_argument("--qdrant-collection", default="inteli_documents_chunks")
    parser.add_argument(
        "--recreate-qdrant-collection",
        action="store_true",
        default=False,
        help="Recria a coleção antes de inserir os embeddings.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f" Arquivo não encontrado: {args.pdf_path}")
        sys.exit(1)

    print(" Iniciando execução do pipeline `embedding_pipeline`...")
    run = embedding_pipeline(
        pdf_path=args.pdf_path,
        chunk_size=600,
        chunk_overlap=120,
        skip_summary=True,
        normalize_embeddings=args.normalize_embeddings,
        qdrant_collection_name=args.qdrant_collection,
        recreate_qdrant_collection=args.recreate_qdrant_collection,
    )
    print(" Pipeline finalizado. Utilize `zenml login --local` para visualizar o run.")
    return run


if __name__ == "__main__":
    main()
