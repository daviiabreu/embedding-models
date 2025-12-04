import json
import logging
import os
import re
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Set

import docx2txt
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# --- UNSTRUCTURED IMPORTS (Detecção Avançada) ---
try:
    from unstructured.chunking.title import chunk_by_title
    from unstructured.partition.pdf import partition_pdf

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("⚠️ Unstructured não instalado. Usando fallback PyPDFLoader.")

# Shim para compatibilidade pdfminer
try:
    from pdfminer.psexceptions import PSSyntaxError as _
except ImportError:
    import sys as _sys
    import types

    from pdfminer.pdfparser import PDFSyntaxError

    shim = types.ModuleType("pdfminer.psexceptions")

    class PSSyntaxError(PDFSyntaxError):
        """Backwards-compatible alias."""

    shim.PSSyntaxError = PSSyntaxError
    _sys.modules["pdfminer.psexceptions"] = shim

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer


# ZenML Bypass
def step(func):
    return func


def pipeline(func):
    return func


load_dotenv()
load_dotenv("agent_flow/.env", override=False)

# --- CONFIGURAÇÕES ---
DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_DIMENSION = 384

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "inteli_hybrid_final")

# --- LOGGING ESTRUTURADO ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ================= FUNÇÕES AUXILIARES =================


def clean_text(text: str) -> str:
    """Higienização rigorosa do texto."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"Pág\.\s*\d+", "", text)
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ue009", "tt")
    return text


def infer_category(filename: str, existing_category: str = None) -> str:
    """Define taxonomia do documento."""
    if existing_category:
        return existing_category
    fname = filename.lower()
    if "edital" in fname or "regras" in fname:
        return "regras_edital"
    elif "faq" in fname:
        return "faq"
    elif "livro" in fname or "institucional" in fname:
        return "institucional"
    elif "tapi" in fname or "robo" in fname:
        return "contexto_robo"
    return "geral"


def infer_element_type_from_text(text: str) -> str:
    """Fallback: infere tipo se Unstructured não estiver disponível."""
    text = text.strip()
    if not text:
        return "Unknown"

    is_short = len(text) < 100
    starts_numeric = re.match(r"^\d+(\.\d+)*\.?\s", text)
    is_upper_title = text.isupper() and len(text) > 4

    if is_short and (starts_numeric or is_upper_title):
        return "Title"

    if re.match(r"^[\•\-\*]\s", text) or re.match(r"^\d+\)\s", text):
        return "ListItem"

    return "NarrativeText"


def detect_summary_elements(
    elements: List[Dict[str, Any]],
    detection_method: str = "keywords",
    max_pages: int = 15,
) -> Set[int]:
    """Detecta e retorna índices de elementos de sumário."""
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
            element_type = element.get("type", "")

            # Detecta título de sumário
            if element_type in ["Title", "Header"] and any(
                kw == text for kw in summary_keywords
            ):
                logger.info(f"🔍 Sumário detectado (keyword) no elemento {i}")

                # Marca elementos subsequentes
                j = i
                while j < len(elements) and j < i + 300:
                    summary_elements.add(j)
                    j += 1

                    # Critério de parada: título que não parece item de sumário
                    if j > i + 10 and elements[j].get("type") in ["Title", "Header"]:
                        next_text = elements[j].get("text", "")
                        # Se não tiver padrão de sumário, para
                        if not re.search(r"\.{3,}\s*\d+$", next_text):
                            break

    elif detection_method == "pattern":
        patterns = [
            r"^\d+\.?\s+.+\s+\d+$",
            r"^.+\.{3,}\s*\d+$",
        ]
        for i, element in enumerate(elements):
            page_num = element.get("metadata", {}).get("page_number", 999)
            if page_num <= max_pages:
                text = element.get("text", "").strip()
                if any(re.match(pattern, text) for pattern in patterns):
                    summary_elements.add(i)

    return summary_elements


def determine_hierarchy_level(element: Dict[str, Any]) -> str:
    """Define nível hierárquico (level_1, level_2, etc)."""
    element_type = element.get("type", "")
    text = element.get("text", "")

    if element_type in ["Title", "Header"]:
        match = re.match(r"^(\d+(\.\d+)*)", text)
        if match:
            dots = match.group(1).count(".")
            if not match.group(1).endswith("."):
                dots += 1
            return f"level_{dots}"
        return "title_main"

    if element_type == "ListItem":
        return "list_item"

    return "body"


def extract_section_info(element: Dict[str, Any]) -> str:
    """Extrai nome limpo da seção."""
    text = element.get("text", "")
    element_type = element.get("type", "")

    if element_type in ["Title", "Header"]:
        clean = re.sub(r"^(\d+(\.\d+)*\.?)\s*", "", text)
        return clean if len(clean) > 2 else text
    return "general"


def process_table_element(element) -> str:
    """Extrai HTML de tabelas se disponível."""
    if hasattr(element, "category") and element.category == "Table":
        if hasattr(element.metadata, "text_as_html") and element.metadata.text_as_html:
            return f"[TABELA]\n{element.metadata.text_as_html}"
    return str(element)


# ================= EXTRAÇÃO INTELIGENTE =================


@step
def extract_pdf_with_unstructured(file_path: Path) -> List[Dict[str, Any]]:
    """
    Estratégia PREMIUM: Usa Unstructured para detecção de layout e tabelas.
    """
    logger.info(f"🚀 Processamento Hi-Res (Unstructured): {file_path.name}")

    try:
        # Particionamento com detecção de layout
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            infer_table_structure=True,
            languages=["por"],
        )

        # Chunking semântico nativo
        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=200,
            max_characters=1000,
            new_after_n_chars=800,
        )

        processed_data = []
        for i, chunk in enumerate(chunks):
            # Processa tabelas especialmente
            content = (
                process_table_element(chunk)
                if "Table" in str(chunk.category)
                else chunk.text
            )
            content = clean_text(content)

            if len(content) < 30:  # Filtro de ruído
                continue

            processed_data.append(
                {
                    "text": content,
                    "type": str(chunk.category),
                    "metadata": {
                        "page_number": getattr(chunk.metadata, "page_number", 1),
                        "source": file_path.name,
                        "is_table": "Table" in str(chunk.category),
                    },
                    "element_id": f"uns_{file_path.stem}_{i}",
                }
            )

        logger.info(f"📊 Unstructured: {len(processed_data)} chunks semânticos criados")
        return processed_data

    except Exception as e:
        logger.error(f"❌ Erro no Unstructured: {e}")
        raise e


@step
def extract_pdf_fallback(file_path: Path) -> List[Dict[str, Any]]:
    """
    Estratégia FALLBACK: PyPDFLoader com inferência manual.
    Usado quando Unstructured não está disponível ou falha.
    """
    logger.warning(f"⚠️ Usando fallback PyPDFLoader: {file_path.name}")

    elements_data = []

    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        global_idx = 0
        for page in pages:
            page_num = page.metadata.get("page", 0) + 1
            raw_text = page.page_content

            # Quebra em linhas para granularidade
            lines = raw_text.split("\n")

            for line in lines:
                clean_line = clean_text(line)
                if len(clean_line) < 10:
                    continue

                # Inferência manual de tipo
                el_type = infer_element_type_from_text(clean_line)

                elements_data.append(
                    {
                        "text": clean_line,
                        "type": el_type,
                        "metadata": {
                            "page_number": page_num,
                            "source": file_path.name,
                            "is_table": False,
                        },
                        "element_id": f"pdf_{global_idx}",
                    }
                )
                global_idx += 1

    except Exception as e1:
        logger.warning(f"⚠️ PyPDFLoader falhou, tentando PyPDF2: {e1}")

        try:
            reader = PdfReader(str(file_path))
            global_idx = 0

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split("\n")
                for line in lines:
                    clean_line = clean_text(line)
                    if len(clean_line) < 10:
                        continue

                    elements_data.append(
                        {
                            "text": clean_line,
                            "type": infer_element_type_from_text(clean_line),
                            "metadata": {
                                "page_number": i + 1,
                                "source": file_path.name,
                                "is_table": False,
                            },
                            "element_id": f"pdf2_{global_idx}",
                        }
                    )
                    global_idx += 1

        except Exception as e2:
            logger.error(f"❌ Todos os métodos de PDF falharam: {e2}")
            raise e2

    logger.info(f"📊 Fallback: {len(elements_data)} elementos extraídos")
    return elements_data


@step
def extract_file_elements(
    file_path_str: str, use_unstructured: bool = True
) -> List[Dict[str, Any]]:
    """
    Extração inteligente com seleção automática de estratégia.
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    logger.info(f"📂 Processando: {file_path.name}")

    # JSON (Web Scraping)
    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        elements_data = []

        if isinstance(data, list):
            for idx, item in enumerate(data):
                elements_data.append(
                    {
                        "text": f"{item.get('title', '')}\n{item.get('content', '')}",
                        "type": "WebContent",
                        "metadata": {
                            "source": item.get("url", file_path.name),
                            "category": "web_scraping",
                            "page_number": 1,
                        },
                        "element_id": f"web_{idx}",
                    }
                )

        return elements_data

    # DOCX/TXT
    elif file_path.suffix.lower() in [".docx", ".txt", ".md"]:
        text = ""
        if file_path.suffix == ".docx":
            text = docx2txt.process(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        elements_data = []
        lines = text.split("\n")

        for idx, line in enumerate(lines):
            clean_line = clean_text(line)
            if len(clean_line) < 10:
                continue

            elements_data.append(
                {
                    "text": clean_line,
                    "type": infer_element_type_from_text(clean_line),
                    "metadata": {
                        "page_number": 1,
                        "source": file_path.name,
                        "segment_index": idx,
                    },
                    "element_id": f"doc_{idx}",
                }
            )

        return elements_data

    # PDF - Estratégia Inteligente
    elif file_path.suffix.lower() == ".pdf":
        if use_unstructured and UNSTRUCTURED_AVAILABLE:
            try:
                return extract_pdf_with_unstructured(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Unstructured falhou, usando fallback: {e}")
                return extract_pdf_fallback(file_path)
        else:
            return extract_pdf_fallback(file_path)

    else:
        raise ValueError(f"Formato não suportado: {file_path.suffix}")


# ================= PROCESSAMENTO E CHUNKING =================


def preprocess_elements(
    elements: List[Dict[str, Any]], skip_summary: bool = True
) -> List[Dict[str, Any]]:
    """Limpeza, detecção de sumário e enriquecimento de contexto."""

    # Detecção de sumário
    summary_indices = set()
    if skip_summary:
        summary_indices = detect_summary_elements(elements, detection_method="keywords")
        if not summary_indices:
            summary_indices = detect_summary_elements(
                elements, detection_method="pattern"
            )

        if summary_indices:
            logger.info(f"🧹 Removendo {len(summary_indices)} elementos de sumário")

    # Tracking de contexto
    current_context = {"section": "Introdução", "last_header": ""}

    processed_elements = []

    for i, element in enumerate(elements):
        if i in summary_indices:
            continue

        text = element.get("text", "")
        if len(text) < 10:
            continue

        # Enriquece metadados
        meta = element.get("metadata", {}).copy()
        element_type = element.get("type", "")

        meta["element_type"] = element_type
        meta["category"] = infer_category(meta.get("source", ""), meta.get("category"))
        meta["hierarchy_level"] = determine_hierarchy_level(element)
        meta["is_header"] = element_type in ["Title", "Header"]

        # Atualiza contexto
        if meta["is_header"]:
            section_name = extract_section_info(element)
            if section_name != "general":
                current_context["section"] = section_name
            current_context["last_header"] = text

        # Injeta contexto
        meta["context_section"] = current_context["section"]
        meta["context_header"] = current_context["last_header"]

        processed_elements.append({"text": text, "metadata": meta})

    logger.info(f"🧹 Processados: {len(processed_elements)} elementos válidos")
    return processed_elements


@step
def create_smart_chunks(
    processed_elements: List[Dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """Chunking com agrupamento por contexto e prefixo semântico."""
    logger.info(f"✂️ Chunking inteligente (size={chunk_size}, overlap={chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    # Agrupa por arquivo + seção
    grouped_text = {}

    for el in processed_elements:
        key = f"{el['metadata']['source']}|{el['metadata']['context_section']}"

        if key not in grouped_text:
            grouped_text[key] = {"text": [], "meta_sample": el["metadata"]}

        grouped_text[key]["text"].append(el["text"])

    chunk_dicts = []

    for key, data in grouped_text.items():
        full_text = "\n".join(data["text"])
        base_meta = data["meta_sample"]

        # PREFIXO DE CONTEXTO (CRÍTICO PARA EMBEDDINGS)
        context_prefix = f"[Contexto: {base_meta['context_section']}]\n"
        if base_meta.get("context_header"):
            context_prefix += f"[Seção: {base_meta['context_header']}]\n"

        contextualized_text = context_prefix + full_text

        docs = text_splitter.split_documents(
            [Document(page_content=contextualized_text, metadata=base_meta)]
        )

        for i, doc in enumerate(docs):
            safe_name = re.sub(r"[^a-zA-Z0-9]", "_", base_meta["source"])
            safe_section = re.sub(
                r"[^a-zA-Z0-9]", "_", base_meta["context_section"][:20]
            )
            chunk_id = f"{safe_name}_{safe_section}_{i}"

            chunk_dicts.append(
                {"id": chunk_id, "content": doc.page_content, "metadata": doc.metadata}
            )

    logger.info(f"📦 Total de chunks: {len(chunk_dicts)}")
    return chunk_dicts


# ================= EMBEDDINGS E INGESTÃO =================


@step
def generate_hybrid_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gera vetores densos e esparsos."""
    if not chunks:
        return []

    texts = [c["content"] for c in chunks]

    logger.info(f"🧠 Gerando Dense Embeddings ({DENSE_MODEL_NAME})...")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    dense_embeddings = dense_model.encode(texts, batch_size=32, show_progress_bar=True)

    logger.info(f"🧠 Gerando Sparse Embeddings ({SPARSE_MODEL_NAME})...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    sparse_embeddings = list(sparse_model.embed(texts))

    results = []
    for i, chunk in enumerate(chunks):
        c = dict(chunk)
        c["dense_vector"] = dense_embeddings[i].tolist()
        c["sparse_vector"] = sparse_embeddings[i]
        results.append(c)

    return results


@step
def ingest_hybrid_embeddings(
    embeddings: List[Dict[str, Any]], recreate_collection: bool = False
) -> None:
    """Ingestão no Qdrant com tratamento robusto de erros."""
    if not embeddings:
        logger.warning("⚠️ Nenhum embedding para ingerir")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Verifica coleção
    exists = False
    try:
        exists = client.collection_exists(COLLECTION_NAME)
    except JSONDecodeError:
        logger.warning("⚠️ JSONDecodeError ao verificar coleção, assumindo existente")
        exists = True
    except Exception as e:
        logger.error(f"❌ Erro ao verificar coleção: {e}")

    if recreate_collection and exists:
        logger.info(f"♻️ Recriando coleção '{COLLECTION_NAME}'...")
        try:
            client.delete_collection(COLLECTION_NAME)
            exists = False
        except Exception as e:
            logger.error(f"❌ Erro ao deletar coleção: {e}")

    if not exists:
        logger.info(f"✨ Criando coleção híbrida '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": qdrant_models.VectorParams(
                    size=DENSE_DIMENSION, distance=qdrant_models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": qdrant_models.SparseVectorParams(
                    index=qdrant_models.SparseIndexParams(on_disk=False)
                )
            },
        )

    points = []
    for item in embeddings:
        u_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["id"]))

        # Conversão de Sparse Vector (CRÍTICO!)
        sparse_raw = item["sparse_vector"]
        if hasattr(sparse_raw, "indices") and hasattr(sparse_raw, "values"):
            sparse_vector = qdrant_models.SparseVector(
                indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist()
            )
        else:
            sparse_vector = sparse_raw

        points.append(
            qdrant_models.PointStruct(
                id=u_id,
                vector={"dense": item["dense_vector"], "sparse": sparse_vector},
                payload={
                    "chunk_id": item["id"],  # ID legível
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "category": item["metadata"].get("category", "geral"),
                    "source": item["metadata"].get("source", "unknown"),
                    "context": item["metadata"].get("context_header", ""),
                    "hierarchy": item["metadata"].get("hierarchy_level", "body"),
                },
            )
        )

    batch_size = 64
    for i in range(0, len(points), batch_size):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME, points=points[i : i + batch_size]
            )
            logger.info(
                f"⬆️ Batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1} enviado"
            )
        except JSONDecodeError:
            logger.warning(
                "⚠️ JSONDecodeError no upsert, mas dados provavelmente foram inseridos"
            )
        except Exception as e:
            logger.error(f"❌ Erro no batch {i // batch_size + 1}: {e}")

    logger.info("✅ Ingestão híbrida concluída!")


# ================= PIPELINE PRINCIPAL =================


def embedding_pipeline(
    pdf_path: str,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
    recreate_collection: bool = False,
    skip_summary: bool = True,
    use_unstructured: bool = True,
) -> None:
    """Pipeline completo com estratégia adaptativa."""

    logger.info("=" * 60)
    logger.info(f"🚀 INICIANDO PIPELINE: {Path(pdf_path).name}")
    logger.info("=" * 60)

    raw = extract_file_elements(
        file_path_str=pdf_path, use_unstructured=use_unstructured
    )

    proc = preprocess_elements(elements=raw, skip_summary=skip_summary)

    chunks = create_smart_chunks(
        processed_elements=proc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    embeds = generate_hybrid_embeddings(chunks=chunks)

    ingest_hybrid_embeddings(embeddings=embeds, recreate_collection=recreate_collection)

    logger.info("=" * 60)
    logger.info("✅ PIPELINE FINALIZADO COM SUCESSO")
    logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Híbrido: Unstructured + Controle Granular"
    )
    parser.add_argument("file_path", help="Caminho do arquivo")
    parser.add_argument("--reset", action="store_true", help="Recria a coleção")
    parser.add_argument(
        "--keep-summary", action="store_true", help="Não remove sumários"
    )
    parser.add_argument(
        "--no-unstructured",
        action="store_true",
        help="Força uso de fallback (não usa Unstructured)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=600, help="Tamanho dos chunks"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=150, help="Overlap entre chunks"
    )

    args = parser.parse_args()

    embedding_pipeline(
        pdf_path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recreate_collection=args.reset,
        skip_summary=not args.keep_summary,
        use_unstructured=not args.no_unstructured,
    )


if __name__ == "__main__":
    main()
