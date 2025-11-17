import streamlit as st
import os
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime
import logging

# Configurar path para importações
project_root = Path(__file__).parent
sys.path.append(str(project_root / "pipeline"))
sys.path.append(str(project_root / "stt"))
sys.path.append(str(project_root / "tts"))

from stt_service import transcribe_audio
from llm_service import get_llm_response
from tts_service import text_to_speech

# Configurar logging para capturar na interface
logging.basicConfig(level=logging.INFO)

# Configuração da página
st.set_page_config(
    page_title="🤖 AI Audio Chat",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS para estilizar o chat
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}

.chat-message.user {
    background-color: #2b313e;
    align-items: flex-end;
}

.chat-message.bot {
    background-color: #475063;
    align-items: flex-start;
}

.chat-message .avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 10px;
}

.chat-message .message {
    background-color: #ffffff;
    color: #000000;
    padding: 10px 15px;
    border-radius: 20px;
    max-width: 70%;
    word-wrap: break-word;
}

.pipeline-step {
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
    border-left: 4px solid #00ff00;
    background-color: #f0f8f0;
    color: #000000;
}

.pipeline-step.processing {
    border-left-color: #ffaa00;
    background-color: #fff8f0;
    color: #000000;
}

.pipeline-step.error {
    border-left-color: #ff0000;
    background-color: #fff0f0;
    color: #000000;
}

.status-container {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa o estado da sessão"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = {}

def display_chat_message(message, is_user=True):
    """Exibe uma mensagem no chat"""
    message_class = "user" if is_user else "bot"
    avatar = "👤" if is_user else "🤖"

    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 30px; margin-right: 10px;">{avatar}</span>
            <strong>{"Você" if is_user else "Laika"}</strong>
        </div>
        <div class="message">
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

def update_pipeline_status(step, status, message=""):
    """Atualiza o status da pipeline"""
    st.session_state.pipeline_status[step] = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

def display_pipeline_status():
    """Exibe o status atual da pipeline"""
    if not st.session_state.pipeline_status:
        return

    st.markdown("### 🔄 Status da Pipeline")

    steps = [
        ("upload", "📁 Upload do Áudio"),
        ("transcription", "🎤 Transcrição (STT)"),
        ("llm", "🤖 Processamento LLM"),
        ("tts", "🔊 Síntese de Voz (TTS)"),
        ("complete", "✅ Concluído")
    ]

    for step_key, step_name in steps:
        if step_key in st.session_state.pipeline_status:
            status_info = st.session_state.pipeline_status[step_key]
            status = status_info["status"]
            message = status_info["message"]
            timestamp = status_info["timestamp"]

            if status == "processing":
                icon = "🔄"
                class_name = "processing"
            elif status == "completed":
                icon = "✅"
                class_name = ""
            elif status == "error":
                icon = "❌"
                class_name = "error"
            else:
                icon = "⏳"
                class_name = ""

            st.markdown(f"""
            <div class="pipeline-step {class_name}">
                <strong>{icon} {step_name}</strong> - {timestamp}
                {f"<br><em>{message}</em>" if message else ""}
            </div>
            """, unsafe_allow_html=True)

def process_audio_pipeline(audio_file, audio_filename):
    """Processa o áudio através da pipeline completa"""

    # Criar diretórios se não existirem
    input_dir = Path("input_audio")
    output_dir = Path("output_audio")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    try:
        # ETAPA 1: Salvar áudio
        update_pipeline_status("upload", "processing", "Salvando arquivo de áudio...")

        # Salvar arquivo temporário
        audio_path = input_dir / audio_filename
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        update_pipeline_status("upload", "completed", f"Áudio salvo: {audio_filename}")
        time.sleep(0.5)  # Pausa para visualização

        # ETAPA 2: Transcrição
        update_pipeline_status("transcription", "processing", "Convertendo áudio para texto...")

        transcription = transcribe_audio(str(audio_path))

        if not transcription:
            update_pipeline_status("transcription", "error", "Falha na transcrição")
            return None, None, None

        update_pipeline_status("transcription", "completed", f"Texto: {transcription[:100]}...")
        time.sleep(0.5)

        # ETAPA 3: LLM
        update_pipeline_status("llm", "processing", "Gerando resposta inteligente...")

        llm_response = get_llm_response(transcription)

        if not llm_response:
            update_pipeline_status("llm", "error", "Falha na geração de resposta")
            return transcription, None, None

        update_pipeline_status("llm", "completed", f"Resposta: {llm_response[:100]}...")
        time.sleep(0.5)

        # ETAPA 4: TTS
        update_pipeline_status("tts", "processing", "Convertendo resposta para áudio...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"response_{timestamp}.mp3"
        output_path = output_dir / output_filename

        audio_success = text_to_speech(llm_response, str(output_path))

        if not audio_success:
            update_pipeline_status("tts", "error", "Falha na síntese de áudio")
            return transcription, llm_response, None

        update_pipeline_status("tts", "completed", f"Áudio gerado: {output_filename}")
        time.sleep(0.5)

        # ETAPA 5: Concluído
        update_pipeline_status("complete", "completed", "Pipeline executada com sucesso!")

        return transcription, llm_response, str(output_path)

    except Exception as e:
        st.error(f"Erro na pipeline: {e}")
        update_pipeline_status("complete", "error", f"Erro: {str(e)}")
        return None, None, None

def main():
    """Função principal da aplicação"""

    initialize_session_state()

    # Header
    st.title("Demo Laika")
    st.markdown("### Converse com a Laika usando áudio!")

    # Layout em colunas
    col1, col2 = st.columns([2, 1])

    with col1:
        # Chat History
        st.markdown("### 💬 Conversa")

        # Container para mensagens
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                display_chat_message(message["content"], message["role"] == "user")

        # Upload de áudio
        st.markdown("---")
        st.markdown("### 📁 Enviar Áudio")

        uploaded_file = st.file_uploader(
            "Escolha um arquivo de áudio",
            type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
            help="Formatos suportados: MP3, WAV, M4A, OGG, FLAC"
        )

        # Botão de processamento
        if uploaded_file is not None:
            # Mostrar preview do arquivo
            st.audio(uploaded_file)

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("🚀 Processar Áudio", type="primary", use_container_width=True):
                    # Limpar status anterior
                    st.session_state.pipeline_status = {}

                    # Placeholder para status em tempo real
                    status_placeholder = st.empty()

                    with st.spinner("Processando..."):
                        # Processar áudio
                        transcription, llm_response, audio_path = process_audio_pipeline(
                            uploaded_file, uploaded_file.name
                        )

                        # Adicionar mensagens ao chat
                        if transcription:
                            st.session_state.messages.append({
                                "role": "user",
                                "content": transcription
                            })

                        if llm_response:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": llm_response,
                                "audio_path": audio_path  # Adicionar caminho do áudio
                            })

                        # Rerun para atualizar o chat
                        st.rerun()

            with col_btn2:
                if st.button("🗑️ Limpar Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.pipeline_status = {}
                    st.rerun()

    with col2:
        # Status da Pipeline
        display_pipeline_status()

        # Reprodutor de áudio da última resposta
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "assistant" and "audio_path" in last_message:
                audio_path = last_message.get("audio_path")

                if audio_path and Path(audio_path).exists():
                    st.markdown("### 🎧 Resposta da Laika")

                    # Player de áudio
                    with open(audio_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')

                    # Download
                    st.download_button(
                        label="⬇️ Baixar Áudio",
                        data=audio_bytes,
                        file_name=Path(audio_path).name,
                        mime="audio/mp3",
                        use_container_width=True
                    )
                else:
                    st.warning("🔊 Áudio não disponível")

if __name__ == "__main__":
    main()
