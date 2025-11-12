import streamlit as st
import os
from datetime import datetime

def create_file_uploader():
    """Componente para upload de arquivos com validação"""
    
    st.markdown("### 📤 Upload de Documentos")
    
    uploaded_files = st.file_uploader(
        "Arraste e solte arquivos PDF aqui ou clique para selecionar",
        type=['pdf'],
        accept_multiple_files=True,
        help="Suporta múltiplos arquivos PDF (máx. 200MB cada)",
        key="pdf_uploader"
    )
    
    if uploaded_files:
        # Validar tamanho dos arquivos
        valid_files = []
        for file in uploaded_files:
            if file.size > 200 * 1024 * 1024:  # 200MB
                st.error(f"❌ {file.name} é muito grande (máx. 200MB)")
            else:
                valid_files.append(file)
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} arquivo(s) válido(s) selecionado(s)")
            
            # Mostrar detalhes dos arquivos
            with st.expander("📋 Detalhes dos arquivos"):
                for file in valid_files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(file.name)
                    with col2:
                        st.text(f"{file.size / 1024 / 1024:.1f} MB")
                    with col3:
                        st.text(file.type)
        
        return valid_files
    
    return None

def save_uploaded_file(uploaded_file, upload_dir="uploaded_documents"):
    """Salva arquivo enviado pelo usuário com timestamp"""
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
    filename = f"{timestamp}_{safe_filename}"
    file_path = os.path.join(upload_dir, filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path, filename

def create_processing_interface():
    """Interface para configuração e execução do processamento"""
    
    st.markdown("### ⚙️ Configurações de Processamento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.slider(
            "Tamanho do Chunk", 
            min_value=200, 
            max_value=1000, 
            value=500, 
            step=50,
            help="Tamanho máximo de cada chunk em caracteres"
        )
    
    with col2:
        chunk_overlap = st.slider(
            "Sobreposição", 
            min_value=0, 
            max_value=min(200, chunk_size // 2), 
            value=min(100, chunk_size // 5), 
            step=25,
            help="Número de caracteres de sobreposição entre chunks"
        )
    
    # Configurações avançadas
    with st.expander("🔧 Configurações Avançadas"):
        ocr_language = st.selectbox(
            "Idioma OCR",
            ["por", "eng", "spa", "fra"],
            index=0,
            help="Idioma para reconhecimento de texto OCR"
        )
        
        strategy = st.selectbox(
            "Estratégia de Extração",
            ["fast", "hi_res", "ocr_only"],
            index=0,
            help="fast: mais rápido, hi_res: melhor qualidade, ocr_only: apenas OCR"
        )
        
        min_text_length = st.number_input(
            "Comprimento mínimo de texto",
            min_value=5,
            max_value=100,
            value=10,
            help="Elementos com menos caracteres serão ignorados"
        )
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'ocr_language': ocr_language,
        'strategy': strategy,
        'min_text_length': min_text_length
    }

def display_processing_results(chunks, original_filename):
    """Exibe resultados do processamento"""
    
    if not chunks:
        st.error("❌ Nenhum chunk foi gerado")
        return
    
    st.success("✅ Processamento concluído!")
    
    # Estatísticas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Chunks", len(chunks))
    
    with col2:
        total_chars = sum(len(chunk['content']) for chunk in chunks)
        st.metric("Total de Caracteres", f"{total_chars:,}")
    
    with col3:
        avg_size = total_chars / len(chunks) if chunks else 0
        st.metric("Tamanho Médio", f"{avg_size:.0f}")
    
    with col4:
        sections = set(chunk['metadata'].get('section_context', 'N/A') for chunk in chunks)
        st.metric("Seções Identificadas", len(sections))
    
    # Distribuição de tamanhos
    with st.expander("📊 Análise Detalhada"):
        import matplotlib.pyplot as plt
        import numpy as np
        
        sizes = [len(chunk['content']) for chunk in chunks]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histograma de tamanhos
        ax1.hist(sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Tamanho do Chunk (caracteres)')
        ax1.set_ylabel('Frequência')
        ax1.set_title('Distribuição de Tamanhos')
        
        # Box plot
        ax2.boxplot(sizes)
        ax2.set_ylabel('Tamanho (caracteres)')
        ax2.set_title('Estatísticas de Tamanho')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Estatísticas numéricas
        st.write("**Estatísticas:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mínimo", min(sizes))
        with col2:
            st.metric("Máximo", max(sizes))
        with col3:
            st.metric("Mediana", int(np.median(sizes)))
        with col4:
            st.metric("Desvio Padrão", f"{np.std(sizes):.0f}")
    
    # Preview dos chunks
    with st.expander("👀 Preview dos Chunks"):
        num_preview = st.slider("Número de chunks para preview", 1, min(10, len(chunks)), 3)
        
        for i in range(num_preview):
            chunk = chunks[i]
            st.write(f"**Chunk {i+1}:**")
            
            # Metadados em colunas
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.write(f"*Página:* {chunk['metadata'].get('page_number', 'N/A')}")
            with meta_col2:
                st.write(f"*Tipo:* {chunk['metadata'].get('element_type', 'N/A')}")
            with meta_col3:
                st.write(f"*Seção:* {chunk['metadata'].get('section_context', 'N/A')}")
            
            # Conteúdo
            content = chunk['content']
            if len(content) > 300:
                st.write(content[:300] + "...")
                if st.button(f"Ver completo", key=f"expand_{i}"):
                    st.write(content)
            else:
                st.write(content)
            
            st.write("---")
    
    return True