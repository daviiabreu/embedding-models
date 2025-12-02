import streamlit as st
import os
import sys
import time
from dotenv import load_dotenv

# Configuração de Caminhos
# Adiciona o diretório raiz ao path para importar agent_flow
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = .../embedding-models/streamlit
# root_dir = .../embedding-models
root_dir = os.path.dirname(current_dir) 
sys.path.append(root_dir)
# Adiciona também a pasta agent_flow para que os imports internos de 'agents' funcionem
sys.path.append(os.path.join(root_dir, "agent_flow"))

# Carrega variáveis de ambiente
load_dotenv(os.path.join(root_dir, ".env"))
load_dotenv(os.path.join(root_dir, "agent_flow", ".env"), override=False)

# Importação do Agente (Lazy import para evitar erros se não for usado)
try:
    # Tenta importar diretamente de agents (já que agent_flow está no path)
    from agents.orchestrator_agent import OrchestratorAgent
except ImportError as e:
    print(f"Erro de importação do Agente: {e}")
    OrchestratorAgent = None

def render_chat_page():
    """Renderiza a interface do Cão Robô (Agente)"""
    st.title("🤖 Inteli Robot Dog")
    st.markdown("""
    Converse com o Cão Robô do Inteli! Ele pode:
    - 📚 **Responder dúvidas** sobre o Inteli (RAG)
    - 👟 **Guiar um tour** pelo campus
    - 💬 **Bater um papo** descontraído
    """)

    # Inicializa o Orquestrador se necessário
    if "orchestrator" not in st.session_state:
        if OrchestratorAgent is None:
            st.error("❌ Não foi possível carregar o Agente Orquestrador. Verifique se a pasta 'agent_flow' está acessível.")
            return

        with st.spinner("Inicializando agentes..."):
            try:
                st.session_state.orchestrator = OrchestratorAgent()
                # Mensagem inicial
                if "agent_messages" not in st.session_state:
                    st.session_state.agent_messages = []
                    initial_msg = "Olá! Sou o Cão Robô do Inteli. Posso te ajudar com informações ou te levar para um tour. O que deseja?"
                    st.session_state.agent_messages.append({"role": "assistant", "content": initial_msg})
            except Exception as e:
                st.error(f"Falha ao iniciar o orquestrador: {e}")
                return

    # Inicializa histórico se não existir
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Exibir Histórico
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input do Usuário
    if prompt := st.chat_input("Digite sua mensagem para o Robô..."):
        # Adiciona mensagem do usuário
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processamento do Bot
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                start_time = time.time()
                response = st.session_state.orchestrator.process_message(prompt)
                end_time = time.time()
                
                message_placeholder.markdown(response)
                
                # Adiciona resposta ao histórico
                st.session_state.agent_messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Desculpe, tive um erro: {e}"
                message_placeholder.markdown(error_msg)
                st.session_state.agent_messages.append({"role": "assistant", "content": error_msg})
