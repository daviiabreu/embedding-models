from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

import numpy as np
import json
import re
import sys

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\ufb01', 'fi')
    text = text.replace('\ue009', 'tt')
    text = re.sub(r'Pág\.\s*\d+', '', text)
    text = re.sub(r'[•◦▪▫]', '•', text)
    return text.strip()

def determine_hierarchy_level(element):
    element_type = element.get('type')
    text = element.get('text', '')
    if element_type == 'Title':
        if re.match(r'^\d+\.', text):
            level = len(text.split('.')[0])
            return f"level_{level}"
        return "title_main"
    elif element_type == 'ListItem':
        return "list_item"
    else:
        return "body"

def extract_section_info(element):
    text = element.get('text', '')
    element_type = element.get('type')
    if element_type == 'Title' and re.match(r'^\d+\.', text):
        return text.split('.')[0] + "." + text.split('.')[1].strip() if '.' in text else text
    if element_type == 'ListItem' and re.match(r'^\d+\.', text):
        return text
    return "general"

def extract_enhanced_metadata(element):
    metadata = element.get('metadata', {})
    return {
        'element_id': element.get('element_id'),
        'element_type': element.get('type'),
        'page_number': metadata.get('page_number'),
        'parent_id': metadata.get('parent_id'),
        'text_length': len(element.get('text', '')),
        'is_header': element.get('type') in ['Title'],
        'is_list_item': element.get('type') == 'ListItem',
        'is_table_content': element.get('type') in ['Table', 'TableRow'],
        'hierarchy_level': determine_hierarchy_level(element),
        'section': extract_section_info(element)
    }

def preprocess_elements(elements, skip_summary=True, summary_detection="keywords"):
    processed_elements = []
    current_section = "Introduction"
    current_subsection = ""

    # Detectar elementos de sumário se solicitado
    summary_elements = set()
    if skip_summary:
        summary_elements = detect_summary_elements(elements, summary_detection)
        if summary_elements:
            print(f"🚫 Detectados {len(summary_elements)} elementos de sumário para pular")

    for i, element in enumerate(elements):
        # Pular elementos de sumário
        if i in summary_elements:
            continue

        text = element.get('text', '').strip()

        # Filtros básicos
        if not text or len(text) < 10:
            continue
        if re.match(r'^(Pág\.\s*|[\s·•◦▪▫\d])+$', text, re.IGNORECASE):
            continue
        if element.get('type') == 'Footer':
            continue

        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue

        metadata = extract_enhanced_metadata(element)

        # Atualiza o contexto da seção/subseção
        if metadata['element_type'] == 'Title' and any(char.isdigit() for char in cleaned_text[:5]):
            current_section = cleaned_text
            current_subsection = ""
        elif metadata['element_type'] == 'Title':
            current_subsection = cleaned_text

        metadata['section_context'] = current_section
        metadata['subsection_context'] = current_subsection

        processed_element = {
            'text': cleaned_text,
            'metadata': metadata
        }

        processed_elements.append(processed_element)

    return processed_elements

def create_overlapping_chunks(processed_elements, chunk_size=500, chunk_overlap=100):
    print(f"Overlapping de chunks de tamanho: (size={chunk_size}, overlap={chunk_overlap})...")

    # 1. Instancia o Text Splitter.
    # Tenta dividir o texto em separadores lógicos ("\n\n", "\n", " ", "")
    # para manter a coesão semântica.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Adiciona o índice de início do chunk no metadado
    )

    # 2. Converte nossos elementos processados para o formato de Documento do.
    # O splitter do trabalha com esse formato.
    documents = []
    for element in processed_elements:
        section = element['metadata'].get('section_context', '')
        subsection = element['metadata'].get('subsection_context', '')

        context_prefix = ""
        if section and section != "Introduction":
            context_prefix = f"Seção: {section}. "
        if subsection:
            context_prefix += f"Subseção: {subsection}. "

        # O `page_content` é o texto que será dividido e vetorizado.
        # O `metadata` é preservado e associado a cada chunk resultante.
        doc = Document(
            page_content=f"{context_prefix}{element['text']}",
            metadata=element['metadata']
        )
        documents.append(doc)

    # 3. Executa o split. O cuida da divisão e sobreposição.
    chunked_documents = text_splitter.split_documents(documents)

    print(f"Foram criados {len(chunked_documents)} chunks.")
    return chunked_documents

def preprocess_for_embedding(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        elements = json.load(f)

    print(f"Carregados {len(elements)} elementos do JSON")

    processed_elements = preprocess_elements(elements)
    print(f"Foram processados {len(processed_elements)} elementos")

    chunked_docs = create_overlapping_chunks(processed_elements)

    embedding_ready_chunks = []
    for i, doc in enumerate(chunked_docs):
        embedding_ready_chunks.append({
            'id': f"chunk_{i}",
            'content': doc.page_content,
            'metadata': doc.metadata
        })

    return embedding_ready_chunks

def detect_summary_elements(elements, detection_method="keywords", max_pages=5):
    """
    Detecta elementos de sumário usando diferentes métodos

    Args:
        elements: Lista de elementos
        detection_method: "keywords", "page_range", ou "pattern"
        max_pages: Máximo de páginas para considerar (para page_range)
    """
    summary_elements = set()

    if detection_method == "keywords":
        summary_keywords = [
            'SUMÁRIO', 'SUMARIO', 'ÍNDICE', 'INDICE',
            'TABLE OF CONTENTS', 'CONTENTS'
        ]

        for i, element in enumerate(elements):
            text = element.get('text', '').strip().upper()
            if any(keyword in text for keyword in summary_keywords):
                j = i
                while j < len(elements):
                    summary_elements.add(j)
                    j += 1
                    if (j < len(elements) and
                        elements[j].get('type') == 'Title' and
                        not any(kw in elements[j].get('text', '').upper()
                               for kw in summary_keywords)):
                        break

    elif detection_method == "page_range":
        for i, element in enumerate(elements):
            page_num = element.get('metadata', {}).get('page_number')
            if page_num and page_num <= max_pages:
                text = element.get('text', '').strip()
                # Padrão típico de linha de sumário
                if re.match(r'.+\.{3,}\s*\d+$|.+\s+\d+$', text):
                    summary_elements.add(i)

    elif detection_method == "pattern":
        for i, element in enumerate(elements):
            text = element.get('text', '').strip()
            patterns = [
                r'^\d+\.\s+.+\s+\d+$',  # "1. Titulo 5"
                r'^.+\.{3,}\s*\d+$',    # "Titulo ... 5"
                r'^[A-Z\s]+\s+\d+$'     # "TITULO 5"
            ]
            if any(re.match(pattern, text) for pattern in patterns):
                summary_elements.add(i)

    return summary_elements

def process_document_with_params(pdf_path, chunk_size=500, chunk_overlap=100, skip_summary=True):
    """Função wrapper para processar documento com parâmetros customizados"""
    # Extrair elementos
    elements = partition_pdf(filename=pdf_path, strategy="fast", ocr_languages="por")

    # Converter elementos para formato de dicionário
    elements_data = []
    for element in elements:
        element_dict = {
            'text': str(element),
            'type': element.__class__.__name__,
            'metadata': {}
        }

        # Extrair metadados se disponíveis
        if hasattr(element, 'metadata') and element.metadata:
            element_dict['metadata'] = element.metadata.to_dict()

        # Adicionar element_id se disponível
        if hasattr(element, 'id'):
            element_dict['element_id'] = element.id

        elements_data.append(element_dict)

    # Preprocessar elementos com opção de pular sumário
    processed_elements = preprocess_elements(elements_data, skip_summary=skip_summary)

    # Criar chunks com parâmetros customizados
    chunked_docs = create_overlapping_chunks(
        processed_elements,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Preparar output final
    embedding_ready_chunks = []
    for i, doc in enumerate(chunked_docs):
        embedding_ready_chunks.append({
            'id': f"chunk_{i}",
            'content': doc.page_content,
            'metadata': doc.metadata
        })

    return embedding_ready_chunks

def generate_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2', normalize=True):
    """
    Gera embeddings para os chunks usando SentenceTransformers

    Args:
        chunks: Lista de chunks com formato {'id': str, 'content': str, 'metadata': dict}
        model_name: Nome do modelo de embedding a ser usado
        normalize: Se deve normalizar os embeddings

    Returns:
        Lista de chunks enriquecidos com embeddings
    """
    if not chunks:
        print('❌ Nenhum chunk disponível para embedding.')
        return []

    print(f'🤖 Carregando modelo de embedding: {model_name}')
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f'❌ Erro ao carregar o modelo: {e}')
        return chunks  # Retorna chunks sem embeddings se houver erro

    # Extrair textos dos chunks
    texts = [chunk['content'] for chunk in chunks]

    print(f'🔄 Gerando embeddings para {len(texts)} chunks...')
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_tensor=False  # Retorna numpy arrays
        )
    except Exception as e:
        print(f'❌ Erro ao gerar embeddings: {e}')
        return chunks  # Retorna chunks sem embeddings se houver erro

    # Enriquecer chunks com embeddings
    chunks_with_embeddings = []
    for chunk, vector in zip(chunks, embeddings):
        enriched = dict(chunk)  # Copia o chunk original
        enriched['embedding'] = vector.tolist()  # Converte numpy array para lista
        enriched['embedding_model'] = model_name  # Adiciona info do modelo usado
        enriched['embedding_dimension'] = len(vector)  # Adiciona dimensão do vetor
        chunks_with_embeddings.append(enriched)

    print(f'✅ Embeddings gerados para {len(chunks_with_embeddings)} chunks')
    print(f'📏 Dimensão dos embeddings: {len(embeddings[0])}')

    return chunks_with_embeddings

def process_document_with_embeddings(pdf_path, chunk_size=500, chunk_overlap=100,
                                   skip_summary=True, generate_embeddings_flag=True,
                                   model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Pipeline completo: extração, chunking e embedding

    Args:
        pdf_path: Caminho para o arquivo PDF
        chunk_size: Tamanho máximo do chunk
        chunk_overlap: Sobreposição entre chunks
        skip_summary: Se deve pular páginas de sumário
        generate_embeddings_flag: Se deve gerar embeddings
        model_name: Nome do modelo de embedding

    Returns:
        Lista de chunks com embeddings (se solicitado)
    """
    # Usar a função existente para processar o documento
    chunks = process_document_with_params(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        skip_summary=skip_summary
    )

    # Gerar embeddings se solicitado
    if generate_embeddings_flag and chunks:
        chunks = generate_embeddings(chunks, model_name=model_name)

    return chunks

def main():
    """
    Função de demonstração - mostra como usar as funções programaticamente
    Para interface completa, use: streamlit run app.py
    """

    print("📄 Embedding Pipeline - Modo Demonstração")
    print("💡 Para interface completa, execute: streamlit run app.py")

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]

        if not os.path.exists(pdf_path):
            print(f"❌ Arquivo não encontrado: {pdf_path}")
            return

        print(f"🔍 Processando: {pdf_path}")

        # Processar com parâmetros padrão
        chunks = process_document_with_params(
            pdf_path=pdf_path,
            chunk_size=500,
            chunk_overlap=100,
            skip_summary=True
        )

        if chunks:
            print(f"✅ {len(chunks)} chunks criados")
            print(f"📊 Total de caracteres: {sum(len(chunk['content']) for chunk in chunks):,}")
            print(f"📏 Tamanho médio: {sum(len(chunk['content']) for chunk in chunks) / len(chunks):.1f}")
        else:
            print("❌ Nenhum chunk foi criado")

    else:
        print("❌ Nenhum arquivo especificado")
        print("📖 Uso: python main.py arquivo.pdf")
        print("📖 Ou execute: streamlit run app.py")

if __name__ == "__main__":
    main()