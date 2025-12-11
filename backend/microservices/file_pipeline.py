import json
import logging
import os
import re
import uuid
from json import JSONDecodeError
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Set

import docx2txt
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
import tiktoken
from collections import defaultdict


load_dotenv()

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

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Conta tokens usando tiktoken."""
    return len(_encoding.encode(text or ""))


def read_text_with_fallback(file_path: Path) -> str:
    """Lê texto tentando múltiplas codificações e ignorando erros se necessário."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    logger.warning(f"Decodificação parcial (errors=ignore) para {file_path}")
    return file_path.read_text(encoding="utf-8", errors="ignore")


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


def normalize_bullets(text: str) -> str:
    """Normaliza bullets e espaçamentos."""
    text = re.sub(r"[•·▪►]", "- ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def is_header_footer_line(text: str) -> bool:
    """Heurística simples para remover cabeçalhos/rodapés recorrentes."""
    t = text.strip().lower()
    if len(t) < 4:
        return False
    return bool(
        re.match(r"^(page\s*\d+|p[aá]g\s*\d+)$", t)
        or re.match(r"^\d+\s*/\s*\d+$", t)
        or re.match(r"^figura\s+\d+", t)
        or re.match(r"^tabela\s+\d+", t)
    )


def is_table_like(text: str) -> bool:
    """Marca linhas que parecem tabela (delimitadores, muitas colunas)."""
    if "|" in text or "\t" in text:
        return True
    separators = len(re.findall(r"[;:]", text))
    numbers = len(re.findall(r"\d", text))
    return separators >= 3 and numbers > 5


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
                logger.info(f"Sumário detectado (keyword) no elemento {i}")

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

# ================= EXTRAÇÃO INTELIGENTE =================


def extract_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    PyPDFLoader com inferência manual.
    """

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
        logger.warning(f"PyPDFLoader falhou, tentando PyPDF2: {e1}")

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
            logger.error(f"Todos os métodos de PDF falharam: {e2}")
            raise e2

    logger.info(f"Fallback: {len(elements_data)} elementos extraídos")
    return elements_data


def extract_file_elements(
    file_path_str: str, use_unstructured: bool = True
) -> List[Dict[str, Any]]:
    """
    Extração inteligente com seleção automática de estratégia.
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    logger.info(f"Processando: {file_path.name}")

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
            text = read_text_with_fallback(file_path)

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
        return extract_pdf(file_path)

    else:
        raise ValueError(f"Formato não suportado: {file_path.suffix}")


# ================= PROCESSAMENTO E CHUNKING =================


def preprocess_elements(
    elements: List[Dict[str, Any]], skip_summary: bool = True
) -> List[Dict[str, Any]]:
    """Limpeza, detecção de sumário e enriquecimento de contexto."""

    repeated_lines: Dict[str, int] = defaultdict(int)

    # Detecção de sumário
    summary_indices = set()
    if skip_summary:
        summary_indices = detect_summary_elements(elements, detection_method="keywords")
        if not summary_indices:
            summary_indices = detect_summary_elements(
                elements, detection_method="pattern"
            )

        if summary_indices:
            logger.info(f"Removendo {len(summary_indices)} elementos de sumário")

    # Tracking de contexto
    current_context = {"section": "Introdução", "last_header": ""}

    processed_elements = []

    for i, element in enumerate(elements):
        if i in summary_indices:
            continue

        raw_text = element.get("text", "")
        text = normalize_bullets(clean_text(raw_text))

        if is_header_footer_line(text):
            continue

        if len(text) < 10:
            continue

        # Deduplicação simples de linhas recorrentes no documento
        repeated_lines[text] += 1
        if repeated_lines[text] > 3:
            continue

        # Enriquece metadados
        meta = element.get("metadata", {}).copy()
        element_type = element.get("type", "")

        meta["element_type"] = element_type
        meta["category"] = infer_category(meta.get("source", ""), meta.get("category"))
        meta["hierarchy_level"] = determine_hierarchy_level(element)
        meta["is_header"] = element_type in ["Title", "Header"]
        meta["is_table"] = meta.get("is_table", False) or is_table_like(text)

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

    logger.info(f"Processados: {len(processed_elements)} elementos válidos")
    return processed_elements


def create_smart_chunks(
    processed_elements: List[Dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Chunking por seções/subseções respeitando limites de tokens e overlap.

    - Agrupa por seção/contexto existente nos metadados.
    - Divide cada grupo em pedaços de até `chunk_size` tokens.
    - Mantém overlap fixo de `chunk_overlap` tokens entre chunks consecutivos.
    """
    logger.info("Chunking por seções com limite de tokens e overlap")

    if not processed_elements:
        return []

    # Garantir overlap de 300 tokens por padrão, mesmo que argumentos não sejam ajustados
    chunk_size_tokens = max(1, chunk_size)
    overlap_tokens = max(0, chunk_overlap)
    overlap_tokens = min(overlap_tokens, chunk_size_tokens - 1) if chunk_size_tokens > 1 else 0

    # Agrupa mantendo a ordem de primeira ocorrência por seção
    section_groups: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()

    for element in processed_elements:
        meta = element.get("metadata", {}) or {}

        # Preferência por identificadores explícitos; fallback para contexto inferido
        section_id = meta.get("section_id") or meta.get("section_path")
        section_label = section_id or meta.get("context_section") or "geral"
        header_label = meta.get("context_header") or meta.get("section_header", "")

        group_key = (meta.get("source"), section_label, header_label)

        if group_key not in section_groups:
            section_groups[group_key] = {
                "texts": [],
                "pages": set(),
                "meta_sample": meta,
                "section_label": section_label,
            }

        section_groups[group_key]["texts"].append(element.get("text", ""))
        page_number = meta.get("page_number")
        if page_number is not None:
            section_groups[group_key]["pages"].add(page_number)

    chunk_dicts: List[Dict[str, Any]] = []
    content_signatures: Set[str] = set()

    for section_idx, (_, data) in enumerate(section_groups.items()):
        base_meta = data["meta_sample"].copy()
        base_meta["pages"] = sorted(data["pages"]) if data["pages"] else []
        base_meta["section_length_chars"] = sum(len(t) for t in data["texts"])
        base_meta["section_elements"] = len(data["texts"])

        context_prefix = f"[Contexto: {base_meta.get('context_section', 'geral')}]\n"
        if base_meta.get("context_header"):
            context_prefix += f"[Seção: {base_meta['context_header']}]\n"

        safe_name = re.sub(
            r"[^a-zA-Z0-9]", "_", base_meta.get("source", "chunk_source")
        )
        safe_section = re.sub(
            r"[^a-zA-Z0-9]", "_", str(data.get("section_label", "section"))[:20]
        )

        section_texts = data["texts"]
        if not section_texts:
            continue

        avg_tokens = sum(count_tokens(t) for t in section_texts) / max(
            1, len(section_texts)
        )
        section_chunk_size = chunk_size_tokens
        if avg_tokens < 15:
            section_chunk_size = min(chunk_size_tokens, 600)
        elif avg_tokens < 30:
            section_chunk_size = min(chunk_size_tokens, 750)

        section_overlap = min(overlap_tokens, max(1, section_chunk_size // 2))

        def build_overlap_lines(lines: List[str]) -> List[str]:
            acc_tokens = 0
            overlap_lines: List[str] = []
            for line in reversed(lines):
                overlap_lines.insert(0, line)
                acc_tokens += count_tokens(line)
                if acc_tokens >= overlap_tokens:
                    break
            return overlap_lines

        current_lines: List[str] = []
        current_tokens = 0
        chunk_counter = 0
        section_chunks: List[Dict[str, Any]] = []

        for line in section_texts:
            tokens = count_tokens(line)
            if tokens == 0:
                continue

            if current_tokens + tokens <= section_chunk_size or not current_lines:
                current_lines.append(line)
                current_tokens += tokens
            else:
                content = context_prefix + "\n".join(current_lines)
                chunk_id = f"{safe_name}_{safe_section}_{section_idx}_{chunk_counter}"
                meta = dict(base_meta)
                meta.update(
                    {
                        "chunk_index": chunk_counter,
                        "section_label": data["section_label"],
                        "section_header": base_meta.get("context_header", ""),
                        "chunk_tokens": count_tokens(content),
                        "overlap_tokens": section_overlap,
                    }
                )
                meta["preview"] = content[:200]
                section_chunks.append({"id": chunk_id, "content": content, "metadata": meta})
                chunk_counter += 1

                overlap_lines = build_overlap_lines(current_lines)
                current_lines = overlap_lines + [line]
                current_tokens = sum(count_tokens(l) for l in overlap_lines) + tokens

        if current_lines:
            content = context_prefix + "\n".join(current_lines)
            chunk_id = f"{safe_name}_{safe_section}_{section_idx}_{chunk_counter}"
            meta = dict(base_meta)
            meta.update(
                {
                    "chunk_index": chunk_counter,
                    "section_label": data["section_label"],
                    "section_header": base_meta.get("context_header", ""),
                    "chunk_tokens": count_tokens(content),
                    "overlap_tokens": section_overlap,
                }
            )
            meta["preview"] = content[:200]
            section_chunks.append({"id": chunk_id, "content": content, "metadata": meta})

        total_section_chunks = len(section_chunks)
        for chunk in section_chunks:
            chunk["metadata"]["total_section_chunks"] = total_section_chunks
            sig = str(hash(chunk["content"]))
            if sig in content_signatures:
                continue
            content_signatures.add(sig)
            chunk_dicts.append(chunk)

    logger.info(f"Total de chunks por seção: {len(chunk_dicts)}")
    return chunk_dicts


# ================= EMBEDDINGS E INGESTÃO =================


def generate_hybrid_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gera vetores densos e esparsos."""
    if not chunks:
        return []

    texts = [c["content"] for c in chunks]

    logger.info(f"Gerando Dense Embeddings ({DENSE_MODEL_NAME})...")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    dense_embeddings = dense_model.encode(texts, batch_size=32, show_progress_bar=True)

    logger.info(f"Gerando Sparse Embeddings ({SPARSE_MODEL_NAME})...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    sparse_embeddings = list(sparse_model.embed(texts))

    results = []
    for i, chunk in enumerate(chunks):
        c = dict(chunk)
        c["dense_vector"] = dense_embeddings[i].tolist()
        c["sparse_vector"] = sparse_embeddings[i]
        results.append(c)

    return results


def ingest_hybrid_embeddings(
    embeddings: List[Dict[str, Any]], recreate_collection: bool = False
) -> None:
    """Ingestão no Qdrant com tratamento robusto de erros."""
    if not embeddings:
        logger.warning("Nenhum embedding para ingerir")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Verifica coleção
    exists = False
    try:
        exists = client.collection_exists(COLLECTION_NAME)
    except JSONDecodeError:
        logger.warning("JSONDecodeError ao verificar coleção, assumindo existente")
        exists = True
    except Exception as e:
        logger.error(f"Erro ao verificar coleção: {e}")

    if recreate_collection and exists:
        logger.info(f"Recriando coleção '{COLLECTION_NAME}'...")
        try:
            client.delete_collection(COLLECTION_NAME)
            exists = False
        except Exception as e:
            logger.error(f"Erro ao deletar coleção: {e}")

    if not exists:
        logger.info(f"Criando coleção híbrida '{COLLECTION_NAME}'...")
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
                f"⬆Batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1} enviado"
            )
        except JSONDecodeError:
            logger.warning(
                "JSONDecodeError no upsert, mas dados provavelmente foram inseridos"
            )
        except Exception as e:
            logger.error(f"Erro no batch {i // batch_size + 1}: {e}")

    logger.info("Ingestão híbrida concluída!")


def log_chunking_stats(chunks: List[Dict[str, Any]]) -> None:
    """Loga estatísticas simples para inspeção offline."""
    if not chunks:
        logger.info("Nenhum chunk gerado.")
        return

    total_tokens = sum(c.get("metadata", {}).get("chunk_tokens", 0) for c in chunks)
    logger.info(
        f"Chunks gerados: {len(chunks)} | Tokens médios por chunk: {total_tokens / max(1, len(chunks)):.1f}"
    )


ALLOWED_EXTENSIONS = {".pdf", ".json", ".docx", ".txt", ".md"}


def collect_input_files(path: str) -> List[str]:
    """Retorna lista de arquivos suportados a partir de um caminho (arquivo ou pasta)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {p}")

    if p.is_file():
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Extensão não suportada: {p.suffix}")
        return [str(p)]

    files: List[str] = []
    for root, _, filenames in os.walk(p):
        for name in filenames:
            if Path(name).suffix.lower() in ALLOWED_EXTENSIONS:
                files.append(str(Path(root) / name))

    if not files:
        raise ValueError("Nenhum arquivo suportado encontrado na pasta.")

    files.sort()
    logger.info(f"Arquivos encontrados: {len(files)}")
    return files


def embedding_pipeline_paths(
    file_paths: List[str],
    chunk_size: int = 900,
    chunk_overlap: int = 300,
    recreate_collection: bool = False,
    skip_summary: bool = True,
) -> None:
    """Processa vários arquivos em sequência, recriando coleção apenas uma vez se solicitado."""
    if not file_paths:
        logger.warning("Lista de arquivos vazia para ingestão.")
        return

    for idx, file_path in enumerate(file_paths):
        logger.info(f"==> Ingerindo ({idx + 1}/{len(file_paths)}): {file_path}")
        recreate = recreate_collection and idx == 0
        try:
            embedding_pipeline(
                pdf_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                recreate_collection=recreate,
                skip_summary=skip_summary,
            )
        except Exception as e:
            logger.error(f"Falha ao processar {file_path}: {e}", exc_info=True)
            continue


# ================= PIPELINE PRINCIPAL =================


def embedding_pipeline(
    pdf_path: str,
    chunk_size: int = 900,
    chunk_overlap: int = 300,
    recreate_collection: bool = False,
    skip_summary: bool = True,
    use_unstructured: bool = True,
) -> None:
    """Pipeline completo com estratégia adaptativa."""

    logger.info("=" * 60)
    logger.info(f"INICIANDO PIPELINE: {Path(pdf_path).name}")
    logger.info("=" * 60)

    raw = extract_file_elements(
        file_path_str=pdf_path, use_unstructured=use_unstructured
    )

    proc = preprocess_elements(elements=raw, skip_summary=skip_summary)

    chunks = create_smart_chunks(
        processed_elements=proc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    log_chunking_stats(chunks)

    embeds = generate_hybrid_embeddings(chunks=chunks)

    ingest_hybrid_embeddings(embeddings=embeds, recreate_collection=recreate_collection)

    logger.info("=" * 60)
    logger.info("PIPELINE FINALIZADO COM SUCESSO")
    logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Híbrido: Unstructured + Controle Granular"
    )
    parser.add_argument("path", help="Caminho do arquivo ou pasta")
    parser.add_argument("--reset", action="store_true", help="Recria a coleção")
    parser.add_argument(
        "--keep-summary", action="store_true", help="Não remove sumários"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=900, help="Tamanho máximo do chunk em tokens"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=300, help="Overlap entre chunks (tokens)"
    )

    args = parser.parse_args()

    files = collect_input_files(args.path)
    embedding_pipeline_paths(
        files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recreate_collection=args.reset,
        skip_summary=not args.keep_summary,
    )


if __name__ == "__main__":
    main()
