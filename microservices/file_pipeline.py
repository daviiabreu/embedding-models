import os
import re
import sys
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
import docx2txt
from PyPDF2 import PdfReader

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Shim para compatibilidade do pdfminer/unstructured
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

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf

# ZenML Bypass (devido a erro de ambiente)
# from zenml import pipeline, step
def step(func): return func
def pipeline(func): return func

load_dotenv()
load_dotenv("agent_flow/.env", override=False)

# Configurações Fixas para Consistência
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# Esse modelo gera vetores de dimensão 384
EMBEDDING_DIMENSION = 384 

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "inteli_documents_chunks")

def clean_text(text: str) -> str:
    """Limpeza básica de texto."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ue009", "tt")
    text = re.sub(r"Pág\.\s*\d+", "", text)
    text = re.sub(r"[•◦▪▫]", "•", text)
    return text.strip()

def infer_category_from_filename(filename: str) -> str:
    """
    Infere a categoria do documento baseado no nome do arquivo.
    Isso ajuda o RAG a filtrar contextos.
    """
    fname = filename.lower()
    if "edital" in fname or "regras" in fname:
        return "regras_edital"
    elif "faq" in fname or "perguntas" in fname:
        return "faq"
    elif "livro" in fname or "institucional" in fname:
        return "institucional"
    elif "tapi" in fname or "robo" in fname:
        return "contexto_robo"
    return "geral"

def extract_enhanced_metadata(element: Dict[str, Any], filename: str) -> Dict[str, Any]:
    metadata = element.get("metadata", {})
    category = infer_category_from_filename(filename)
    
    return {
        "source_file": filename,
        "category": category, # METADADO CRUCIAL PARA O RAG
        "element_id": element.get("element_id"),
        "element_type": element.get("type"),
        "page_number": metadata.get("page_number"),
        "is_header": element.get("type") in ["Title"],
    }

@step
def extract_file_elements(file_path_str: str) -> List[Dict[str, Any]]:
    """
    Extrai elementos brutos do arquivo (PDF, TXT ou DOCX).
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    print(f"📂 Extraindo elementos de: {file_path.name}")
    
    elements_data: List[Dict[str, Any]] = []

    # Lógica para TXT
    if file_path.suffix.lower() == ".txt":
        content = file_path.read_text(encoding="utf-8")
        segments = [s.strip() for s in content.split("\n\n") if s.strip()]
        for idx, segment in enumerate(segments, 1):
            elements_data.append({
                "text": segment,
                "type": "TextSegment",
                "metadata": {"page_number": 1},
                "element_id": f"txt_{idx}"
            })

    # Lógica para DOCX (ADICIONADO AGORA)
    elif file_path.suffix.lower() == ".docx":
        text = docx2txt.process(file_path)
        # Quebra simples por parágrafos duplos para simular estrutura
        segments = [s.strip() for s in text.split("\n\n") if s.strip()]
        for idx, segment in enumerate(segments, 1):
            elements_data.append({
                "text": segment,
                "type": "DocxSegment",
                "metadata": {"page_number": 1}, # Docx não tem paginação fixa simples
                "element_id": f"docx_{idx}"
            })
            
    # Lógica para PDF (Substituído por PyPDF2 para maior robustez)
    elif file_path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(file_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    elements_data.append({
                        "text": text,
                        "type": "Page",
                        "element_id": f"page_{i+1}",
                        "metadata": {"page_number": i+1}
                    })
        except Exception as e:
            print(f"❌ Erro ao ler PDF com PyPDF2: {e}")
            # Fallback opcional ou raise
            raise e
    else:
        raise ValueError(f"Formato não suportado: {file_path.suffix}")

    # Injeta o nome do arquivo nos metadados
    for item in elements_data:
        if "metadata" not in item:
            item["metadata"] = {}
        item["metadata"]["filename"] = file_path.name

    print(f"📊 Elementos brutos extraídos: {len(elements_data)}")
    return elements_data

@step
def preprocess_elements(
    elements: List[Dict[str, Any]],
    skip_summary: bool = True
) -> List[Dict[str, Any]]:
    """
    Limpa e enriquece os elementos com metadados de categoria.
    """
    processed_elements: List[Dict[str, Any]] = []
    
    # Detecção simples de sumário (pula as primeiras páginas se tiver muitos pontinhos)
    # Simplificado para focar no conteúdo
    
    for element in elements:
        text = element.get("text", "").strip()
        filename = element["metadata"].get("filename", "unknown")

        # Filtros básicos de qualidade
        if not text or len(text) < 10: continue
        if re.match(r"^(Pág\.\s*|[\s·•◦▪▫\d])+$", text): continue # Apenas números ou marcadores
        
        cleaned_text = clean_text(text)
        
        # Extração de metadados aprimorada
        meta = extract_enhanced_metadata(element, filename)
        
        processed_elements.append({
            "text": cleaned_text,
            "metadata": meta
        })

    print(f"🧹 Pré-processamento concluído. {len(processed_elements)} elementos válidos.")
    return processed_elements

@step
def create_chunks(
    processed_elements: List[Dict[str, Any]],
    chunk_size: int = 600, # Ajustado para MiniLM (aprox 150 tokens, seguro)
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Cria chunks de texto.
    Nota: Se for FAQ, idealmente o chunk_size deveria ser menor, mas 
    o RecursiveCharacterTextSplitter lida bem com isso se os separadores forem bons.
    """
    print(f"✂️ Chunking (Size={chunk_size}, Overlap={chunk_overlap})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # Prioriza quebra de parágrafo
        add_start_index=True,
    )

    # Agrupa texto por arquivo para manter contexto
    docs_per_file = {}
    for el in processed_elements:
        fname = el["metadata"]["source_file"]
        if fname not in docs_per_file:
            docs_per_file[fname] = []
        
        # Cria documento LangChain
        doc = Document(page_content=el["text"], metadata=el["metadata"])
        docs_per_file[fname].append(doc)

    chunk_dicts: List[Dict[str, Any]] = []
    
    for fname, docs in docs_per_file.items():
        # Split
        split_docs = text_splitter.split_documents(docs)
        
        for idx, doc in enumerate(split_docs):
            chunk_dicts.append({
                "id": f"{fname}_{idx}",
                "content": doc.page_content,
                "metadata": doc.metadata
            })

    print(f"📦 Total de chunks criados: {len(chunk_dicts)}")
    return chunk_dicts

@step
def generate_embeddings(
    chunks: List[Dict[str, Any]],
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Gera embeddings usando sentence-transformers.
    """
    if not chunks:
        return []

    print(f"🧠 Carregando modelo: {EMBEDDING_MODEL_NAME}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        raise e

    texts = [c["content"] for c in chunks]
    
    print(f"⚡ Gerando embeddings para {len(texts)} textos...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=normalize
    )

    chunks_with_embeddings = []
    for i, chunk in enumerate(chunks):
        enriched = dict(chunk)
        enriched["embedding"] = embeddings[i].tolist()
        chunks_with_embeddings.append(enriched)

    return chunks_with_embeddings

@step
def ingest_embeddings(
    embeddings: List[Dict[str, Any]],
    recreate_collection: bool = False,
) -> None:
    """
    Ingestão no Qdrant.
    """
    if not embeddings:
        print("⚠️ Nenhum embedding para ingerir.")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Configuração da coleção
    vectors_config = qdrant_models.VectorParams(
        size=EMBEDDING_DIMENSION, # 384 para MiniLM-L12
        distance=qdrant_models.Distance.COSINE
    )

    exists = client.collection_exists(collection_name=COLLECTION_NAME)

    if recreate_collection or not exists:
        print(f"♻️ Recriando coleção '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config
        )
    
    points = []
    for item in embeddings:
        # Converter ID string para UUID determinístico
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["id"]))
        
        points.append(qdrant_models.PointStruct(
            id=point_id, # ID deve ser único (string ou int)
            vector=item["embedding"],
            payload={
                "original_id": item["id"], # Guardar ID original no payload
                "content": item["content"],
                "metadata": item["metadata"],
                "category": item["metadata"].get("category", "geral"), # Facilita filtro
                "source": item["metadata"].get("source_file", "unknown")
            }
        ))

    # Upload em batches
    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"⬆️ Upserted batch {i} - {i+len(batch)}")

    print("✅ Ingestão concluída com sucesso!")

@pipeline
def embedding_pipeline(
    pdf_path: str,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
    recreate_collection: bool = False,
) -> None:
    """Pipeline Principal"""
    raw_elements = extract_file_elements(file_path_str=pdf_path)
    processed = preprocess_elements(elements=raw_elements)
    chunks = create_chunks(processed_elements=processed, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = generate_embeddings(chunks=chunks)
    ingest_embeddings(embeddings=embeddings, recreate_collection=recreate_collection)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Caminho do arquivo PDF ou TXT")
    parser.add_argument("--reset", action="store_true", help="Recria a coleção do zero")
    args = parser.parse_args()

    embedding_pipeline(
        pdf_path=args.file_path,
        recreate_collection=args.reset
    )

if __name__ == "__main__":
    main()