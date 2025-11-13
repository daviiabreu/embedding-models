import streamlit as st
import os
import json
from datetime import datetime
import traceback
import numpy as np

# Importar as funções do main.py
from streamlit.main import (
    preprocess_elements,
    create_overlapping_chunks,
    process_document_with_embeddings  # Nova função
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

def process_pdf_pipeline(pdf_path, chunk_size=500, chunk_overlap=100, skip_summary=True,
                        generate_embeddings_flag=True, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Pipeline completo de processamento do PDF com embeddings"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔍 Processando documento...")
        progress_bar.progress(30)

        # Usar a nova função que inclui embeddings
        chunks = process_document_with_embeddings(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_summary=skip_summary,
            generate_embeddings_flag=generate_embeddings_flag,
            model_name=model_name
        )

        progress_bar.progress(80)

        # Salvar chunks
        if not os.path.exists("processed_documents"):
            os.makedirs("processed_documents")

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        suffix = "_with_embeddings" if generate_embeddings_flag else "_chunks"
        chunks_path = f"processed_documents/{base_name}{suffix}.json"

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        progress_bar.progress(100)
        status_text.text("✅ Processamento concluído!")

        return chunks, chunks_path

    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        st.error(traceback.format_exc())
        return None, None

def display_embedding_stats(chunks):
    """Exibe estatísticas dos embeddings"""
    if not chunks or 'embedding' not in chunks[0]:
        return

    st.subheader("🧠 Estatísticas dos Embeddings")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Chunks com Embeddings", len(chunks))

    with col2:
        embedding_dim = chunks[0].get('embedding_dimension', len(chunks[0]['embedding']))
        st.metric("Dimensão dos Vetores", embedding_dim)

    with col3:
        model_used = chunks[0].get('embedding_model', 'N/A')
        st.metric("Modelo Usado", model_used.split('/')[-1])

    with col4:
        # Calcular similaridade média entre chunks
        embeddings = np.array([chunk['embedding'] for chunk in chunks[:10]])  # Apenas primeiros 10 para performance
        if len(embeddings) > 1:
            similarities = np.dot(embeddings, embeddings.T)
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            st.metric("Similaridade Média", f"{avg_similarity:.3f}")

def main():
    st.title("📄 Document Chunking & Embedding Pipeline")
    st.markdown("Faça upload de documentos PDF e processe-os em chunks com embeddings")

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

    skip_summary = st.sidebar.checkbox(
        "Pular páginas de sumário",
        value=True,
        help="Remove automaticamente páginas de sumário do processamento"
    )

    # NOVA SEÇÃO: Configurações de Embedding
    st.sidebar.header("🧠 Configurações de Embedding")

    generate_embeddings_flag = st.sidebar.checkbox(
        "Gerar Embeddings",
        value=True,
        help="Gera vetores de embedding para cada chunk"
    )

    if generate_embeddings_flag:
        model_options = {
            "MiniLM-L6 (rápido, 384d)": "sentence-transformers/all-MiniLM-L6-v2",
            "MiniLM-L12 (melhor, 384d)": "sentence-transformers/all-MiniLM-L12-v2",
            "MPNet (excelente, 768d)": "sentence-transformers/all-mpnet-base-v2",
            "Multilingual (português, 768d)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        }

        selected_model = st.sidebar.selectbox(
            "Modelo de Embedding",
            options=list(model_options.keys()),
            index=0,
            help="Escolha o modelo para gerar embeddings"
        )

        model_name = model_options[selected_model]

        st.sidebar.info(f"""
        **Modelo selecionado:**
        {selected_model}

        **Características:**
        - Velocidade varia por modelo
        - Dimensões maiores = melhor qualidade
        - Multilingual para textos em português
        """)

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
                        skip_summary=skip_summary,
                        generate_embeddings_flag=generate_embeddings_flag,
                        model_name=model_name if generate_embeddings_flag else None
                    )

                    if chunks:
                        # Mostrar estatísticas básicas
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

                        # Mostrar estatísticas de embedding se gerados
                        if generate_embeddings_flag and 'embedding' in chunks[0]:
                            display_embedding_stats(chunks)

                        # Preview dos chunks
                        with st.expander("👀 Preview dos Chunks"):
                            for i, chunk in enumerate(chunks[:3]):
                                st.write(f"**Chunk {i+1}:**")

                                # Metadados
                                col_meta1, col_meta2, col_meta3 = st.columns(3)
                                with col_meta1:
                                    st.write(f"*Página:* {chunk['metadata'].get('page_number', 'N/A')}")
                                with col_meta2:
                                    st.write(f"*Tipo:* {chunk['metadata'].get('element_type', 'N/A')}")
                                with col_meta3:
                                    if 'embedding' in chunk:
                                        st.write(f"*Embedding:* ✅ ({chunk.get('embedding_dimension', 'N/A')}d)")
                                    else:
                                        st.write("*Embedding:* ❌")

                                # Conteúdo
                                content = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                                st.write(content)
                                st.write("---")

                            if len(chunks) > 3:
                                st.info(f"+ {len(chunks) - 3} chunks adicionais...")

                        # Botão de download
                        if chunks_path and os.path.exists(chunks_path):
                            with open(chunks_path, 'r', encoding='utf-8') as f:
                                chunks_json = f.read()

                            suffix = "_with_embeddings" if generate_embeddings_flag else "_chunks"

                            st.download_button(
                                label=f"💾 Baixar {'Chunks + Embeddings' if generate_embeddings_flag else 'Chunks'} (JSON)",
                                data=chunks_json,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}{suffix}.json",
                                mime="application/json"
                            )

    with col2:
        st.header("📊 Informações")

        st.info("""
        **Como usar:**
        1. Configure os parâmetros na barra lateral
        2. Escolha se quer gerar embeddings
        3. Faça upload dos arquivos PDF
        4. Clique em "Iniciar Processamento"
        5. Baixe os chunks processados
        """)

        st.warning("""
        **Parâmetros:**
        - **Chunk Size**: 300-600 caracteres ideal
        - **Overlap**: 10-20% do chunk size
        - **Embeddings**: Aumenta tempo de processamento
        """)

        if generate_embeddings_flag:
            st.success("""
            **Embeddings habilitados!**
            - Chunks terão vetores numéricos
            - Ideais para busca semântica
            - Compatíveis com bancos vetoriais
            """)

if __name__ == "__main__":
    main()
