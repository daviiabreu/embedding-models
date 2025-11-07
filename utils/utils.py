import json
import os
import tempfile
from datetime import datetime
import streamlit as st

# Importar funções do main.py
from main import (
    preprocess_elements, 
    create_overlapping_chunks,
    clean_text,
    extract_enhanced_metadata
)
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

class DocumentProcessor:
    """Classe para gerenciar o processamento de documentos"""
    
    def __init__(self):
        self.processed_dir = "processed_documents"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Cria diretórios necessários se não existirem"""
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs("uploaded_documents", exist_ok=True)
    
    def extract_pdf_elements(self, pdf_path, strategy="fast", ocr_language="por"):
        """Extrai elementos do PDF usando unstructured"""
        try:
            elements = partition_pdf(
                filename=pdf_path, 
                strategy=strategy, 
                ocr_languages=ocr_language
            )
            return elements
        except Exception as e:
            st.error(f"Erro na extração do PDF: {str(e)}")
            return None
    
    def save_elements_json(self, elements, base_name):
        """Salva elementos extraídos em JSON"""
        json_path = os.path.join(self.processed_dir, f"{base_name}_elements.json")
        
        try:
            elements_to_json(elements=elements, filename=json_path)
            return json_path
        except Exception as e:
            st.error(f"Erro ao salvar elementos: {str(e)}")
            return None
    
    def process_document_pipeline(self, pdf_path, config):
        """Pipeline completo de processamento"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Extração
            status_text.text("🔍 Extraindo elementos do PDF...")
            progress_bar.progress(20)
            
            elements = self.extract_pdf_elements(
                pdf_path, 
                config['strategy'], 
                config['ocr_language']
            )
            
            if not elements:
                return None, None
            
            # Step 2: Salvar JSON
            status_text.text("💾 Salvando elementos extraídos...")
            progress_bar.progress(40)
            
            json_path = self.save_elements_json(elements, base_name)
            if not json_path:
                return None, None
            
            # Step 3: Preprocessamento
            status_text.text("⚙️ Preprocessando elementos...")
            progress_bar.progress(60)
            
            # Converter elementos para formato dict se necessário
            elements_data = []
            for element in elements:
                if hasattr(element, 'to_dict'):
                    elements_data.append(element.to_dict())
                else:
                    elements_data.append({
                        'text': str(element),
                        'type': element.__class__.__name__,
                        'metadata': getattr(element, 'metadata', {})
                    })
            
            processed_elements = preprocess_elements(elements_data)
            
            # Step 4: Chunking
            status_text.text("📄 Criando chunks...")
            progress_bar.progress(80)
            
            chunked_docs = create_overlapping_chunks(
                processed_elements,
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
            
            # Step 5: Preparar output final
            status_text.text("✅ Finalizando...")
            progress_bar.progress(90)
            
            embedding_ready_chunks = []
            for i, doc in enumerate(chunked_docs):
                embedding_ready_chunks.append({
                    'id': f"chunk_{i}",
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            # Step 6: Salvar chunks
            chunks_path = self.save_chunks(embedding_ready_chunks, base_name)
            
            progress_bar.progress(100)
            status_text.text("🎉 Processamento concluído!")
            
            return embedding_ready_chunks, chunks_path
            
        except Exception as e:
            st.error(f"❌ Erro no pipeline: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    def save_chunks(self, chunks, base_name):
        """Salva chunks processados em JSON"""
        chunks_path = os.path.join(self.processed_dir, f"{base_name}_chunks.json")
        
        try:
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            return chunks_path
        except Exception as e:
            st.error(f"Erro ao salvar chunks: {str(e)}")
            return None
    
    def get_processed_files(self):
        """Retorna lista de arquivos processados"""
        if not os.path.exists(self.processed_dir):
            return []
        
        files = []
        for filename in os.listdir(self.processed_dir):
            if filename.endswith('_chunks.json'):
                filepath = os.path.join(self.processed_dir, filename)
                stat = os.stat(filepath)
                files.append({
                    'name': filename,
                    'path': filepath,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def load_chunks(self, chunks_path):
        """Carrega chunks de um arquivo JSON"""
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Erro ao carregar chunks: {str(e)}")
            return None

def create_download_button(chunks, filename):
    """Cria botão de download para os chunks"""
    if chunks:
        chunks_json = json.dumps(chunks, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="💾 Baixar Chunks (JSON)",
            data=chunks_json,
            file_name=f"{filename}_chunks.json",
            mime="application/json",
            help="Baixar chunks processados em formato JSON"
        )