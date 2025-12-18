import os
import re
import json
import uuid
import logging
from collections import OrderedDict, defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Set
from json import JSONDecodeError

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import docx2txt

# --- UNSTRUCTURED IMPORTS (Detecção Avançada) ---
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title
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

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

# ZenML Bypass
def step(func): return func
def pipeline(func): return func

load_dotenv()
load_dotenv("agent_flow/.env", override=False)

# --- CONFIGURAÇÕES ---
DENSE_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
SPARSE_MODEL_NAME = 'Qdrant/bm25'
DENSE_DIMENSION = 384 

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "inteli_hybrid_final")

# --- LOGGING ESTRUTURADO ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= FUNÇÕES AUXILIARES =================

def clean_text(text: str) -> str:
    """Higienização rigorosa do texto."""
    if not text: return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'Pág\.\s*\d+', '', text)
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ue009", "tt")
    return text

def infer_category(filename: str, existing_category: str = None) -> str:
    """Define taxonomia do documento."""
    if existing_category and existing_category != "web_scraping":
        return existing_category
    
    fname = filename.lower()
    if "edital" in fname or "regras" in fname: return "regras_edital"
    elif "faq" in fname: return "faq"
    elif "livro" in fname or "institucional" in fname: return "institucional"
    elif "tapi" in fname or "robo" in fname: return "contexto_robo"
    elif "curso" in fname or "graduacao" in fname: return "curso"
    elif "campus" in fname or "infraestrutura" in fname: return "infraestrutura"
    elif "processo-seletivo" in fname or "vestibular" in fname: return "admissao"
    elif "metodologia" in fname or "ensino" in fname: return "metodologia"
    elif "docente" in fname or "professor" in fname: return "equipe"
    elif "contato" in fname: return "contato"
    return "geral"

def infer_element_type_from_text(text: str) -> str:
    """Fallback: infere tipo se Unstructured não estiver disponível."""
    text = text.strip()
    if not text: return "Unknown"
    
    is_short = len(text) < 100
    starts_numeric = re.match(r"^\d+(\.\d+)*\.?\s", text)
    is_upper_title = text.isupper() and len(text) > 4
    
    if is_short and (starts_numeric or is_upper_title):
        return "Title"
    
    if re.match(r"^[•\-\*]\s", text) or re.match(r"^\d+\)\s", text):
        return "ListItem"
        
    return "NarrativeText"

def detect_summary_elements(
    elements: List[Dict[str, Any]],
    detection_method: str = "keywords",
    max_pages: int = 15
) -> Set[int]:
    """Detecta e retorna índices de elementos de sumário."""
    summary_elements: Set[int] = set()

    if detection_method == "keywords":
        summary_keywords = ["SUMÁRIO", "SUMARIO", "ÍNDICE", "INDICE", "TABLE OF CONTENTS", "CONTENTS"]
        
        for i, element in enumerate(elements):
            text = element.get("text", "").strip().upper()
            element_type = element.get("type", "")
            
            if element_type in ["Title", "Header"] and any(kw == text for kw in summary_keywords):
                logger.info(f"📑 Sumário detectado (keyword) no elemento {i}")
                
                j = i
                while j < len(elements) and j < i + 300:
                    summary_elements.add(j)
                    j += 1
                    
                    if j > i + 10 and elements[j].get("type") in ["Title", "Header"]:
                        next_text = elements[j].get("text", "")
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
    
    if element_type in ["Title", "Header", "section_title", "subsection_title", "heading"]:
        match = re.match(r"^(\d+(\.\d+)*)", text)
        if match:
            dots = match.group(1).count('.')
            if not match.group(1).endswith('.'): 
                dots += 1
            return f"level_{dots}"
        
        # Hierarquia por tipo (para JSON do html_cleaner)
        if element_type == "section_title":
            return "level_1"
        elif element_type == "subsection_title":
            return "level_2"
        elif element_type == "heading":
            level = element.get("level", 3)
            return f"level_{level}"
        
        return "title_main"
    
    if element_type == "ListItem" or element_type == "list_item":
        return "list_item"
    
    return "body"

def extract_section_info(element: Dict[str, Any]) -> str:
    """Extrai nome limpo da seção."""
    text = element.get("text", "")
    element_type = element.get("type", "")
    
    # Para JSON do html_cleaner
    if element.get("section"):
        return element["section"]
    
    if element_type in ["Title", "Header", "section_title", "subsection_title"]:
        clean = re.sub(r"^(\d+(\.\d+)*\.?)\s*", "", text)
        return clean if len(clean) > 2 else text
    
    return "general"

def process_table_element(element) -> str:
    """Extrai HTML de tabelas se disponível."""
    if hasattr(element, 'category') and element.category == "Table":
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
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            infer_table_structure=True,
            languages=["por"],
        )
        
        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=200,
            max_characters=1000,
            new_after_n_chars=800
        )
        
        processed_data = []
        for i, chunk in enumerate(chunks):
            content = process_table_element(chunk) if "Table" in str(chunk.category) else chunk.text
            content = clean_text(content)
            
            if len(content) < 30:
                continue
            
            processed_data.append({
                "text": content,
                "type": str(chunk.category),
                "metadata": {
                    "page_number": getattr(chunk.metadata, "page_number", 1),
                    "source": file_path.name,
                    "is_table": "Table" in str(chunk.category)
                },
                "element_id": f"uns_{file_path.stem}_{i}"
            })
            
        logger.info(f"📊 Unstructured: {len(processed_data)} chunks semânticos criados")
        return processed_data

    except Exception as e:
        logger.error(f"❌ Erro no Unstructured: {e}")
        raise e

@step
def extract_pdf_fallback(file_path: Path) -> List[Dict[str, Any]]:
    """
    Estratégia FALLBACK: PyPDFLoader com inferência manual.
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
            
            lines = raw_text.split('\n')
            
            for line in lines:
                clean_line = clean_text(line)
                if len(clean_line) < 10:
                    continue
                
                el_type = infer_element_type_from_text(clean_line)
                
                elements_data.append({
                    "text": clean_line,
                    "type": el_type,
                    "metadata": {
                        "page_number": page_num,
                        "source": file_path.name,
                        "is_table": False
                    },
                    "element_id": f"pdf_{global_idx}"
                })
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
                    
                lines = text.split('\n')
                for line in lines:
                    clean_line = clean_text(line)
                    if len(clean_line) < 10:
                        continue
                    
                    elements_data.append({
                        "text": clean_line,
                        "type": infer_element_type_from_text(clean_line),
                        "metadata": {
                            "page_number": i + 1,
                            "source": file_path.name,
                            "is_table": False
                        },
                        "element_id": f"pdf2_{global_idx}"
                    })
                    global_idx += 1
                    
        except Exception as e2:
            logger.error(f"❌ Todos os métodos de PDF falharam: {e2}")
            raise e2
    
    logger.info(f"📊 Fallback: {len(elements_data)} elementos extraídos")
    return elements_data

@step
def extract_file_elements(file_path_str: str, use_unstructured: bool = True) -> List[Dict[str, Any]]:
    """
    Extração inteligente com seleção automática de estratégia.
    Agora otimizado para JSON limpo do html_cleaner.py.
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    logger.info(f"📂 Processando: {file_path.name}")
    
    # JSON (Cleaned HTML do html_cleaner.py ou genérico)
    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        elements_data = []
        
        # Formato estruturado do html_cleaner.py
        if isinstance(data, dict) and 'content_blocks' in data:
            logger.info(f"🌐 JSON estruturado detectado (html_cleaner)")
            metadata_base = data.get('metadata', {})
            
            # EXTRAÇÃO DO CURSO DO NOME DO ARQUIVO
            # Ex: "adm-tech_cleaned.json" -> "adm-tech"
            # Ex: "ciencia-da-computacao_cleaned.json" -> "ciencia-da-computacao"
            course_from_filename = None
            filename_stem = file_path.stem  # Remove .json
            if '_cleaned' in filename_stem:
                course_from_filename = filename_stem.replace('_cleaned', '')
            elif '_' in filename_stem:
                # Caso não tenha _cleaned, pega até o primeiro underscore
                course_from_filename = filename_stem.split('_')[0]
            else:
                # Se não tem underscore, usa o nome inteiro
                course_from_filename = filename_stem
            
            logger.info(f"📚 Curso detectado do filename: '{course_from_filename}'")
            
            for idx, block in enumerate(data['content_blocks']):
                text = block.get('text', '')
                if len(text) < 10:
                    continue
                
                # Enriquece com contexto hierárquico
                context_parts = []
                if block.get('section'):
                    context_parts.append(f"[Seção: {block['section']}]")
                if block.get('subsection'):
                    context_parts.append(f"[Subseção: {block['subsection']}]")
                if block.get('list_context'):
                    context_parts.append(f"[Contexto da lista: {block['list_context']}]")
                
                # Prefixo de metadados
                if metadata_base.get('title'):
                    context_parts.insert(0, f"[Página: {metadata_base['title']}]")
                
                contextualized_text = "\n".join(context_parts) + "\n" + text if context_parts else text
                
                elements_data.append({
                    "text": contextualized_text,
                    "type": block.get('type', 'WebContent'),
                    "section": block.get('section'),
                    "subsection": block.get('subsection'),
                    "level": block.get('level', 0),
                    "metadata": {
                        "source": file_path.name,  # Usa nome do JSON processado
                        "source_html": data.get('source_file'),  # HTML original
                        "category": metadata_base.get('page_type', 'web_scraping'),
                        "page_title": metadata_base.get('title', ''),
                        "url": metadata_base.get('url', ''),
                        "description": metadata_base.get('description', ''),
                        "page_number": 1,
                        # NOVOS METADADOS CRÍTICOS
                        "project": block.get('project'),
                        "academic_year": block.get('academic_year'),
                        # METADADOS PARA BUSCA DE PROFESSORES E CURSOS
                        "professor_name": block.get('professor_name'),
                        "course": course_from_filename  # 🔧 FIX: Extrai do nome do arquivo!
                    },
                    "element_id": f"web_{idx}"
                })
            
            logger.info(f"✅ {len(elements_data)} blocos estruturados extraídos")
        
        # Formato de lista (JSON genérico)
        elif isinstance(data, list):
            logger.info(f"📋 JSON genérico (lista) detectado")
            for idx, item in enumerate(data):
                title = item.get('title', '')
                content = item.get('content', '')
                combined_text = f"{title}\n{content}" if title else content
                
                if len(combined_text) < 10:
                    continue
                
                elements_data.append({
                    "text": combined_text,
                    "type": "WebContent",
                    "metadata": {
                        "source": item.get("url", file_path.name),
                        "category": "web_scraping",
                        "page_number": 1
                    },
                    "element_id": f"web_{idx}"
                })
            
            logger.info(f"✅ {len(elements_data)} itens extraídos")
        
        else:
            logger.warning(f"⚠️ Formato JSON não reconhecido")
        
        return elements_data
    
    # DOCX/TXT/MD
    elif file_path.suffix.lower() in [".docx", ".txt", ".md"]:
        text = ""
        if file_path.suffix == ".docx":
            text = docx2txt.process(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")
        
        elements_data = []
        lines = text.split('\n')
        
        for idx, line in enumerate(lines):
            clean_line = clean_text(line)
            if len(clean_line) < 10:
                continue
                
            elements_data.append({
                "text": clean_line,
                "type": infer_element_type_from_text(clean_line),
                "metadata": {
                    "page_number": 1,
                    "source": file_path.name,
                    "segment_index": idx
                },
                "element_id": f"doc_{idx}"
            })
        
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

@step
def preprocess_elements(
    elements: List[Dict[str, Any]], 
    skip_summary: bool = True
) -> List[Dict[str, Any]]:
    """Limpeza, detecção de sumário e enriquecimento de contexto."""
    
    # Detecção de sumário (apenas para PDFs)
    summary_indices = set()
    if skip_summary and elements and 'pdf' in elements[0].get('element_id', ''):
        summary_indices = detect_summary_elements(elements, detection_method="keywords")
        if not summary_indices:
            summary_indices = detect_summary_elements(elements, detection_method="pattern")
        
        if summary_indices:
            logger.info(f"🧹 Removendo {len(summary_indices)} elementos de sumário")

    # Tracking de contexto
    current_context = {
        "section": "Introdução",
        "last_header": ""
    }

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
        meta["is_header"] = element_type in ["Title", "Header", "section_title", "subsection_title", "heading"]
        
        # Atualiza contexto (se não vier do JSON estruturado)
        if not element.get("section"):
            if meta["is_header"]:
                section_name =sim,  extract_section_info(element)
                if section_name != "general":
                    current_context["section"] = section_name
                current_context["last_header"] = text
            
            meta["context_section"] = current_context["section"]
            meta["context_header"] = current_context["last_header"]
        else:
            # Usa contexto do JSON estruturado
            meta["context_section"] = element.get("section", current_context["section"])
            meta["context_header"] = element.get("subsection", current_context["last_header"])
        
        processed_elements.append({"text": text, "metadata": meta})

    logger.info(f"🧹 Processados: {len(processed_elements)} elementos válidos")
    return processed_elements

@step
def create_smart_chunks(
    processed_elements: List[Dict[str, Any]], 
    chunk_size: int = 800,  # Aumentado para Gemini
    chunk_overlap: int = 200  # Overlap generoso
) -> List[Dict[str, Any]]:
    """
    Chunking com agrupamento por contexto e prefixo semântico.
    Otimizado para Gemini (chunks maiores).
    
    SPECIAL HANDLING: professor_info blocks are kept as individual chunks
    to preserve professor_name metadata.
    """
    logger.info(f"✂️ Chunking inteligente (size={chunk_size}, overlap={chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    
    chunk_dicts = []
    
    # STEP 1: Extract professor_info blocks for individual processing
    professor_elements = []
    other_elements = []
    
    for el in processed_elements:
        element_type = el.get('type', '')
        if element_type == 'professor_info' or el['metadata'].get('professor_name'):
            professor_elements.append(el)
        else:
            other_elements.append(el)
    
    logger.info(f"👨‍🏫 Professor blocks (individual chunks): {len(professor_elements)}")
    logger.info(f"📄 Other elements (grouped chunks): {len(other_elements)}")
    
    # STEP 2: Process professor_info blocks as individual chunks (NO GROUPING)
    for el in professor_elements:
        prof_name = el['metadata'].get('professor_name', 'unknown')
        safe_prof = re.sub(r'[^a-zA-Z0-9]', '_', prof_name)
        chunk_id = f"professor_{safe_prof}_{uuid.uuid4().hex[:8]}"
        
        # Build context prefix for professor
        context_prefix = ""
        if el['metadata'].get('page_title'):
            context_prefix += f"[Página: {el['metadata']['page_title']}]\n"
        if el['metadata'].get('context_section'):
            context_prefix += f"[Seção: {el['metadata']['context_section']}]\n"
        if el['metadata'].get('context_header'):
            context_prefix += f"[Subseção: {el['metadata']['context_header']}]\n"
        if el['metadata'].get('professor_name'):
            context_prefix += f"[Professor: {el['metadata']['professor_name']}]\n"
        
        contextualized_text = context_prefix + el['text']
        
        chunk_dicts.append({
            "id": chunk_id,
            "content": contextualized_text,
            "metadata": el['metadata']  # PRESERVES professor_name!
        })
    
    # STEP 3: Process other elements with grouping (original logic)
    grouped_text = {}
    
    for el in other_elements:
        source = el['metadata'].get('source', 'unknown')
        section = el['metadata'].get('context_section', 'general')
        project = el['metadata'].get('project', 'no_project')
        year = el['metadata'].get('academic_year', 'no_year')
        key = f"{source}|{section}|{project}|{year}"
        
        if key not in grouped_text:
            grouped_text[key] = {
                "text": [],
                "meta_sample": el["metadata"]
            }
        
        grouped_text[key]["text"].append(el["text"])
    
    for key, data in grouped_text.items():
        full_text = "\n".join(data["text"])
        base_meta = data["meta_sample"]
        
        # PREFIXO DE CONTEXTO (CRÍTICO PARA EMBEDDINGS)
        context_prefix = ""
        if base_meta.get('page_title'):
            context_prefix += f"[Página: {base_meta['page_title']}]\n"
        if base_meta.get('context_section'):
            context_prefix += f"[Seção: {base_meta['context_section']}]\n"
        if base_meta.get('context_header'):
            context_prefix += f"[Subseção: {base_meta['context_header']}]\n"
        if base_meta.get('category'):
            context_prefix += f"[Categoria: {base_meta['category']}]\n"
        # METADADOS CRÍTICOS PARA GRADE CURRICULAR
        if base_meta.get('project'):
            context_prefix += f"[Projeto: {base_meta['project']}]\n"
        if base_meta.get('academic_year'):
            context_prefix += f"[Ano: {base_meta['academic_year']}]\n"
        
        contextualized_text = context_prefix + full_text
        
        docs = text_splitter.split_documents([
            Document(page_content=contextualized_text, metadata=base_meta)
        ])
        
        for i, doc in enumerate(docs):
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', base_meta.get('source', 'unknown'))
            safe_section = re.sub(r'[^a-zA-Z0-9]', '_', base_meta.get('context_section', 'general')[:20])
            
            # Incluir project e year no chunk_id para garantir unicidade
            project_val = base_meta.get('project') or 'no_proj'
            year_val = base_meta.get('academic_year') or 'no_year'
            safe_project = re.sub(r'[^a-zA-Z0-9]', '_', str(project_val))
            safe_year = re.sub(r'[^a-zA-Z0-9]', '_', str(year_val))
            
            chunk_id = f"{safe_name}_{safe_section}_{safe_project}_{safe_year}_{i}"
            
            chunk_dicts.append({
                "id": chunk_id,
                "content": doc.page_content,
                "metadata": doc.metadata
            })

    logger.info(f"📦 Total de chunks: {len(chunk_dicts)}")
    return chunk_dicts

# ================= EMBEDDINGS E INGESTÃO =================

@step
def generate_hybrid_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gera vetores densos e esparsos com tratamento robusto de erros."""
    if not chunks: 
        return []
        
    texts = [c["content"] for c in chunks]
    
    logger.info(f"🧠 Gerando Dense Embeddings ({DENSE_MODEL_NAME})...")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    dense_embeddings = dense_model.encode(texts, batch_size=32, show_progress_bar=True)

    logger.info(f"🧠 Gerando Sparse Embeddings ({SPARSE_MODEL_NAME})...")
    sparse_embeddings = []
    
    try:
        # Tenta processar em batch único
        sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        sparse_embeddings = list(sparse_model.embed(texts))
        logger.info(f"✅ Sparse embeddings gerados: {len(sparse_embeddings)} vetores")
        
    except BrokenPipeError as e:
        logger.warning(f"⚠️ Broken pipe no sparse embedding, tentando batch menor...")
        
        try:
            # Recarrega o modelo e tenta com batches menores
            sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
            batch_size = 10
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = list(sparse_model.embed(batch))
                sparse_embeddings.extend(batch_embeddings)
                logger.info(f"   Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} processado")
            
            logger.info(f"✅ Sparse embeddings gerados com batches menores")
            
        except Exception as e2:
            logger.error(f"❌ Falha total no sparse embedding: {e2}")
            logger.warning(f"⚠️ Continuando apenas com dense embeddings...")
            # Cria sparse embeddings vazios como fallback
            from qdrant_client.http import models as qdrant_models
            for _ in texts:
                sparse_embeddings.append(qdrant_models.SparseVector(indices=[], values=[]))
    
    except Exception as e:
        logger.error(f"❌ Erro inesperado no sparse embedding: {e}")
        logger.warning(f"⚠️ Continuando apenas com dense embeddings...")
        from qdrant_client.http import models as qdrant_models
        for _ in texts:
            sparse_embeddings.append(qdrant_models.SparseVector(indices=[], values=[]))

    results = []
    for i, chunk in enumerate(chunks):
        c = dict(chunk)
        c["dense_vector"] = dense_embeddings[i].tolist()
        c["sparse_vector"] = sparse_embeddings[i]
        results.append(c)
    
    return results

@step
def ingest_hybrid_embeddings(
    embeddings: List[Dict[str, Any]], 
    recreate_collection: bool = False
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
                    size=DENSE_DIMENSION, 
                    distance=qdrant_models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": qdrant_models.SparseVectorParams(
                    index=qdrant_models.SparseIndexParams(on_disk=False)
                )
            }
        )
    
    points = []
    for item in embeddings:
        u_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["id"]))
        
        # Conversão de Sparse Vector
        sparse_raw = item["sparse_vector"]
        if hasattr(sparse_raw, "indices") and hasattr(sparse_raw, "values"):
            sparse_vector = qdrant_models.SparseVector(
                indices=sparse_raw.indices.tolist(),
                values=sparse_raw.values.tolist()
            )
        else:
            sparse_vector = sparse_raw

        points.append(qdrant_models.PointStruct(
            id=u_id,
            vector={
                "dense": item["dense_vector"], 
                "sparse": sparse_vector
            },
            payload={
                "chunk_id": item["id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "category": item["metadata"].get("category", "geral"),
                "source": item["metadata"].get("source", "unknown"),
                "page_title": item["metadata"].get("page_title", ""),
                "url": item["metadata"].get("url", ""),
                "context": item["metadata"].get("context_header", ""),
                "hierarchy": item["metadata"].get("hierarchy_level", "body"),
                # METADADOS CRÍTICOS PARA GRADE CURRICULAR
                "project": item["metadata"].get("project"),
                "academic_year": item["metadata"].get("academic_year"),
                # METADADOS PARA BUSCA DE PROFESSORES E CURSOS
                "professor_name": item["metadata"].get("professor_name"),
                "course": item["metadata"].get("course")
            }
        ))

    batch_size = 64
    for i in range(0, len(points), batch_size):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME, 
                points=points[i:i+batch_size]
            )
            logger.info(f"⬆️ Batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} enviado")
        except JSONDecodeError:
            logger.warning("⚠️ JSONDecodeError no upsert, mas dados provavelmente foram inseridos")
        except Exception as e:
            logger.error(f"❌ Erro no batch {i//batch_size + 1}: {e}")
    
    logger.info("✅ Ingestão híbrida concluída!")

# ================= PIPELINE PRINCIPAL =================

@pipeline
def embedding_pipeline(
    pdf_path: str, 
    chunk_size: int = 800,  # Otimizado para Gemini
    chunk_overlap: int = 200, 
    recreate_collection: bool = False,
    skip_summary: bool = True,
    use_unstructured: bool = True
) -> None:
    """Pipeline completo otimizado para webscraping e Gemini."""
    
    logger.info("=" * 60)
    logger.info(f"🚀 INICIANDO PIPELINE: {Path(pdf_path).name}")
    logger.info("=" * 60)
    
    raw = extract_file_elements(
        file_path_str=pdf_path,
        use_unstructured=use_unstructured
    )
    
    processed = preprocess_elements(raw, skip_summary=skip_summary)
    chunks = create_smart_chunks(processed, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = generate_hybrid_embeddings(chunks)
    ingest_hybrid_embeddings(embeddings, recreate_collection=recreate_collection)
    
    logger.info("=" * 60)
    logger.info(f"✅ PIPELINE CONCLUÍDO: {len(embeddings)} embeddings gerados")
    logger.info("=" * 60)

# ================= BATCH PROCESSING =================

def process_directory(
    directory_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    recreate_collection: bool = False,
    skip_summary: bool = True,
    use_unstructured: bool = True
) -> None:
    """Processa todos os arquivos de um diretório recursivamente."""
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        logger.error(f"❌ Diretório não encontrado: {directory_path}")
        return
    
    # Extensões suportadas
    supported_extensions = ['.pdf', '.json', '.docx', '.txt', '.md']
    files = []
    
    for ext in supported_extensions:
        files.extend(dir_path.rglob(f'*{ext}'))
    
    if not files:
        logger.warning(f"⚠️ Nenhum arquivo encontrado em {directory_path}")
        return
    
    logger.info(f"📁 Encontrados {len(files)} arquivos para processar")
    
    # Contadores para estatísticas
    total_files = len(files)
    processed_success = 0
    processed_error = 0
    
    # Processa primeiro arquivo com reset
    for i, file_path in enumerate(files):
        current = i + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 Processando arquivo {current}/{total_files}: {file_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            is_first = (i == 0)
            embedding_pipeline(
                pdf_path=str(file_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                recreate_collection=(recreate_collection and is_first),
                skip_summary=skip_summary,
                use_unstructured=use_unstructured
            )
            processed_success += 1
            logger.info(f"✅ Sucesso! Progresso: {processed_success}/{total_files} ({processed_success/total_files*100:.1f}%)")
        except Exception as e:
            processed_error += 1
            logger.error(f"❌ Erro ao processar {file_path.name}: {e}")
            logger.info(f"⚠️  Erros até agora: {processed_error}/{total_files}")
            continue
    
    # Resumo final
    logger.info("\n" + "="*60)
    logger.info("🎉 PROCESSAMENTO EM LOTE CONCLUÍDO!")
    logger.info("="*60)
    logger.info(f"✅ Processados com sucesso: {processed_success}/{total_files}")
    logger.info(f"❌ Erros: {processed_error}/{total_files}")
    logger.info(f"📊 Taxa de sucesso: {processed_success/total_files*100:.1f}%")

# ================= MAIN =================

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print(":x_vermelho: Uso: python file_pipeline.py <path(s)> [--reset] [--unstructured]")
        sys.exit(1)
    reset = "--reset" in args
    use_unstr = "--unstructured" in args or UNSTRUCTURED_AVAILABLE
    # remove flags
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print(":x_vermelho: Nenhum caminho válido informado")
        sys.exit(1)
    logger.info(f":pacote: Total de paths recebidos: {len(paths)}")
    for i, path in enumerate(paths):
        path_obj = Path(path)
        try:
            if path_obj.is_dir():
                logger.info(f":pasta_aberta: Modo batch: processando diretório {path}")
                process_directory(
                    directory_path=path,
                    recreate_collection=(reset and i == 0),
                    use_unstructured=use_unstr
                )
            else:
                logger.info(f":página_virada_para_cima: Processando arquivo {path}")
                embedding_pipeline(
                    pdf_path=path,
                    recreate_collection=(reset and i == 0),
                    use_unstructured=use_unstr
                )
        except Exception as e:
            logger.error(f":x_vermelho: Erro ao processar {path}: {e}")
