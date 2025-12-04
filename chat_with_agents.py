import os
import sys
import time

from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()
load_dotenv("agent_flow/.env", override=False)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_flow.agents.orchestrator_agent import OrchestratorAgent
except ImportError as e:
    print(f"\n❌ ERRO CRÍTICO DE IMPORTAÇÃO: {e}")
    sys.exit(1)

# Cores ANSI para o terminal
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header():
    print("\n" + "█" * 70)
    print(
        f"{BOLD}🤖 INTERFACE DE CHAT - SISTEMA MULTI-AGENTES INTELI (DEBUG MODE){RESET}"
    )
    print("█" * 70)
    print("\n💡 O sistema agora possui:")
    print("   1. Agente de Conhecimento (RAG Híbrido) -> Responde dúvidas")
    print("   2. Agente de Tour (Roteiro) -> Guia o passeio")
    print("   3. Orquestrador -> Decide quem deve responder você")
    print("-" * 70 + "\n")


def print_rag_debug(orchestrator, user_input):
    """
    Função auxiliar para mostrar o que o RAG encontrou ANTES da resposta.
    Isso ajuda a entender por que ele acertou ou errou.
    """
    # Hack para acessar o método interno do orquestrador apenas para debug visual
    # Em produção, o orquestrador faria isso internamente.
    try:
        context_raw = orchestrator._query_knowledge_base(user_input)
        if not context_raw:
            print(
                f"{YELLOW}⚠️  RAG Debug: Nenhum contexto relevante encontrado no Qdrant.{RESET}"
            )
            return

        print(f"{YELLOW}🔍 RAG DEBUG - FONTES RECUPERADAS:{RESET}")

        # O contexto vem como string formatada, vamos tentar parsear visualmente
        sources = context_raw.split("--- FONTE: ")
        for i, source in enumerate(sources[1:], 1):  # Pula o primeiro split vazio
            lines = source.split("\n")
            header = lines[0]  # Ex: "Edital.pdf | Seção: 1. Cursos ---"
            snippet = lines[1][:100] + "..." if len(lines) > 1 else ""

            print(f"   [{i}] {header}")
            print(f'       📝 "{snippet}"')
        print("-" * 70)

    except Exception as e:
        print(f"{RED}Erro no debug do RAG: {e}{RESET}")


def print_bot_response(response: str, execution_time: float, intent: str):
    """Formata a resposta final."""
    print(f"\n{BLUE}🧠 Intenção Detectada: {intent}{RESET}")
    print(f"{BLUE}🤖 [DOG/INTELI]:{RESET}")
    print(f"{GREEN}{response}{RESET}")
    print(f"\n⏱️ Time: {execution_time:.2f}s")
    print("=" * 70)


def main():
    print_header()

    try:
        print("⚙️  Inicializando o Sistema de Agentes...")
        orchestrator = OrchestratorAgent()
        print("✅ Sistema Pronto!\n")
    except Exception as e:
        print(f"❌ Falha ao iniciar: {e}")
        return

    while True:
        try:
            user_input = input(f"{BOLD}👤 VOCÊ: {RESET}").strip()

            if user_input.lower() in ["sair", "exit", "quit"]:
                print("\n👋 Até logo!")
                break

            if not user_input:
                continue

            start_time = time.time()

            # 1. Identifica intenção primeiro (para saber se mostra debug do RAG)
            intent = orchestrator._decide_intent(user_input)

            # 2. Se for pergunta de conhecimento, mostra o debug do RAG antes
            if intent == "KNOWLEDGE":
                print_rag_debug(orchestrator, user_input)

            # 3. Processa a resposta final
            # (Nota: O orquestrador vai chamar o RAG de novo internamente,
            # mas o cache do Qdrant torna isso rápido)
            response = orchestrator.process_message(user_input)

            end_time = time.time()
            print_bot_response(response, end_time - start_time, intent)

        except KeyboardInterrupt:
            print("\n\n👋 Interrompido.")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")


if __name__ == "__main__":
    main()
