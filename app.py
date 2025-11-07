import streamlit as st
import os
import json
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import numpy as np

# Importar apenas as funções que existem no main.py
from main import (
    preprocess_elements,
    create_overlapping_chunks
)
from unstructured.partition.pdf import partition_pdf

# Configuração da página
st.set_page_config(
    page_title="Document Chunking Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_uploaded_file(uploaded_file, upload_dir="uploaded_documents"):
    """Salva arquivo enviado pelo usuário"""
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
    filename = f"{timestamp}_{safe_filename}"
    file_path = os.path.join(upload_dir, filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path, filename

def process_pdf_pipeline(pdf_path, chunk_size=500, chunk_overlap=100, skip_summary=True):
    """Pipeline completo de processamento do PDF"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔍 Processando documento...")
        progress_bar.progress(50)

        # USAR A FUNÇÃO DO MAIN.PY que já está corrigida
        from main import process_document_with_params

        embedding_chunks = process_document_with_params(
            pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_summary=skip_summary  # ADICIONAR ESTE PARÂMETRO
        )

        progress_bar.progress(80)

        # Salvar chunks
        if not os.path.exists("processed_documents"):
            os.makedirs("processed_documents")

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        chunks_path = f"processed_documents/{base_name}_chunks.json"

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_chunks, f, ensure_ascii=False, indent=2)

        progress_bar.progress(100)
        status_text.text("✅ Processamento concluído!")

        return embedding_chunks, chunks_path

    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        st.error(traceback.format_exc())
        return None, None


def main():
    st.title("📄 Document Chunking Pipeline")
    st.markdown("Faça upload de documentos PDF e processe-os em chunks para embeddings")

    # Sidebar com configurações
    st.sidebar.header("⚙️ Configurações")

    chunk_size = st.sidebar.slider(
        "Tamanho do Chunk",
        min_value=200,
        max_value=1000,
        value=500,
        step=50,
        help="Tamanho máximo de cada chunk em caracteres"
    )

    chunk_overlap = st.sidebar.slider(
        "Sobreposição",
        min_value=0,
        max_value=200,
        value=100,
        step=25,
        help="Número de caracteres de sobreposição entre chunks"
    )

    # NOVA OPÇÃO: Pular sumário
    skip_summary = st.sidebar.checkbox(
        "Pular páginas de sumário",
        value=True,
        help="Remove automaticamente páginas de sumário do processamento"
    )

    # Área principal
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload de Documentos")

        uploaded_files = st.file_uploader(
            "Arraste e solte arquivos PDF aqui ou clique para selecionar",
            type=['pdf'],
            accept_multiple_files=True,
            help="Suporta múltiplos arquivos PDF"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} arquivo(s) selecionado(s)")

            # Botão para processar
            if st.button("🚀 Iniciar Processamento", type="primary"):
                for uploaded_file in uploaded_files:
                    st.subheader(f"Processando: {uploaded_file.name}")

                    # Salvar arquivo
                    file_path, filename = save_uploaded_file(uploaded_file)
                    st.info(f"📁 Arquivo salvo: {filename}")

                    # Processar arquivo
                    chunks, chunks_path = process_pdf_pipeline(
                        file_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        skip_summary=skip_summary  # ADICIONAR ESTE PARÂMETRO
                    )

                    if chunks:
                        # Mostrar estatísticas
                        st.success("✅ Processamento concluído!")

                        col_stats1, col_stats2, col_stats3 = st.columns(3)

                        with col_stats1:
                            st.metric("Total de Chunks", len(chunks))

                        with col_stats2:
                            total_chars = sum(len(chunk['content']) for chunk in chunks)
                            st.metric("Total de Caracteres", f"{total_chars:,}")

                        with col_stats3:
                            avg_size = total_chars / len(chunks) if chunks else 0
                            st.metric("Tamanho Médio", f"{avg_size:.0f}")

                        # Preview dos chunks
                        with st.expander("👀 Preview dos Chunks"):
                            for i, chunk in enumerate(chunks[:3]):  # Mostrar apenas os primeiros 3
                                st.write(f"**Chunk {i+1}:**")
                                st.write(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                                st.write("---")

                            if len(chunks) > 3:
                                st.info(f"+ {len(chunks) - 3} chunks adicionais...")

                        # Botão de download
                        if chunks_path and os.path.exists(chunks_path):
                            with open(chunks_path, 'r', encoding='utf-8') as f:
                                chunks_json = f.read()

                            st.download_button(
                                label="💾 Baixar Chunks (JSON)",
                                data=chunks_json,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_chunks.json",
                                mime="application/json"
                            )

    with col2:
        st.header("📊 Informações")

        st.info("""
        **Como usar:**
        1. Configure os parâmetros na barra lateral
        2. Faça upload dos arquivos PDF
        3. Clique em "Iniciar Processamento"
        4. Baixe os chunks processados
        """)

        st.warning("""
        **Parâmetros:**
        - **Chunk Size**: Tamanho ideal entre 300-600 caracteres
        - **Overlap**: 10-20% do chunk size é recomendado
        """)

        # Histórico de arquivos processados
        if os.path.exists("processed_documents"):
            files = [f for f in os.listdir("processed_documents") if f.endswith("_chunks.json")]
            if files:
                st.subheader("📁 Arquivos Processados")
                for file in files[-5:]:  # Últimos 5 arquivos
                    st.text(file)

if __name__ == "__main__":
    main()
