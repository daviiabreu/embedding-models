from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import json
import re

file_path = "documents"
base_file_name = "Edital-Processo-Seletivo-Inteli_-Graduacao-2026_AJUSTADO"

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
    """Extract section information from element"""
    text = element.get('text', '')
    element_type = element.get('type')
    if element_type == 'Title' and re.match(r'^\d+\.', text):
        return text.split('.')[0] + "." + text.split('.')[1].strip() if '.' in text else text
    if element_type == 'ListItem' and re.match(r'^\d+\.', text):
        return text
    return "general"

def extract_enhanced_metadata(element):
    """Extract and enrich metadata from each element"""
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

def preprocess_elements(elements):
    processed_elements = []
    current_section = "Introduction"
    current_subsection = ""
    
    for element in elements:
        text = element.get('text', '').strip()
        
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
    print(f"Creating overlapping chunks with (size={chunk_size}, overlap={chunk_overlap})...")
    
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
        # O prefixo de contexto ainda é útil para ser incluído no texto.
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
    
    print(f"Created {len(chunked_documents)} chunks using Langchain.")
    return chunked_documents

def preprocess_for_embedding(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    
    print(f"Carregados {len(elements)} elementos do JSON")
    
    # Step 1: Limpeza e enriquecimento de metadados (nosso código customizado)
    processed_elements = preprocess_elements(elements)
    print(f"Preprocessed {len(processed_elements)} elements")
    
    # Step 2: Chunking com sobreposição usando
    # Substitui as chamadas para create_contextual_chunks e optimize_chunks
    chunked_docs = create_overlapping_chunks(processed_elements)

    # Step 3: Preparar o output final no formato desejado
    embedding_ready_chunks = []
    for i, doc in enumerate(chunked_docs):
        embedding_ready_chunks.append({
            'id': f"chunk_{i}",
            'content': doc.page_content, # O texto do chunk já contextualizado
            'metadata': doc.metadata      # Os metadados originais + os do splitter
        })
    
    return embedding_ready_chunks

def main():
    try:
        # Step 1: Extract elements (only if JSON doesn't exist)
        json_output_path = f"{file_path}/{base_file_name}-output.json"
        
        try:
            with open(json_output_path, 'r', encoding='utf-8') as f:
                pass
            print("O arquivo JSON já existe, pulando etapa de extração.")
        except FileNotFoundError:
            print("Extraindo elementos do PDF...")
            elements = partition_pdf(filename=f"{file_path}/{base_file_name}.pdf", 
                         strategy="fast", 
                         ocr_languages="por")
            elements_to_json(elements=elements, filename=json_output_path)
            print("Extração completa do PDF")
        
        # Step 2: Preprocess for embedding
        print("Inicializando pipeline...")
        embedding_chunks = preprocess_for_embedding(json_output_path)
        
        # Step 3: Save preprocessed chunks
        chunks_output_path = f"{file_path}/{base_file_name}-chunks.json"
        with open(chunks_output_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_chunks, f, ensure_ascii=False, indent=4) # Indent 4 para melhor leitura
        
        print(f"✅ {len(embedding_chunks)} chunks estão prontos para o embedding")
        print(f"✅ Chunks salvos em: {chunks_output_path}")
        
        # Step 5: Show statistics
        print(f"\n--- ESTATÍSTICAS ---")
        total_chars = sum(len(chunk['content']) for chunk in embedding_chunks)
        if embedding_chunks:
            avg_chunk_size = total_chars / len(embedding_chunks)
            print(f"Total de chunks: {len(embedding_chunks)}")
            print(f"Total de caracteres: {total_chars}")
            print(f"Tamanho médio de chunks: {avg_chunk_size:.1f} caracteres")
        else:
            print("Nenhum chunk foi criado.")
        
    except Exception as e:
        print(f"❌ Erro no processamento da pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()