import logging
import os

import google.generativeai as genai
from dotenv import load_dotenv
from google.adk.agents import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import agents from same package
try:
    from backend.agent_flow.agents.context_agent import create_context_agent
    from backend.agent_flow.agents.knowledge_agent import create_knowledge_agent
    from backend.agent_flow.agents.personality_agent import create_personality_agent
    from backend.agent_flow.agents.safety_agent import create_safety_agent
except ImportError:
    from .context_agent import create_context_agent
    from .knowledge_agent import create_knowledge_agent
    from .personality_agent import create_personality_agent
    from .safety_agent import create_safety_agent

# Carrega ambiente
load_dotenv("backend/agent_flow/.env")
load_dotenv(".env")


def create_orchestrator_agent(
    model: str = None,
    safety_agent: Agent = None,
    context_agent: Agent = None,
    personality_agent: Agent = None,
    knowledge_agent: Agent = None,
) -> Agent:
    """
    Cria um Orchestrator Agent ADK-nativo que usa sub-agents como ferramentas.

    Características:
    1. É um ADK Agent real (não classe Python)
    2. Usa ToolContext real (não MockToolContext)
    3. Delega para Safety Agent via ADK (não chama tools diretamente)
    4. Permite orquestração inteligente via LLM

    Args:
        model: Modelo LLM a usar
        safety_agent: Safety Agent (criado se None)
        context_agent: Context Agent (criado se None)
        personality_agent: Personality Agent (criado se None)
        knowledge_agent: Knowledge Agent (criado se None)

    Returns:
        Agent: Orchestrator Agent configurado
    """
    if model is None:
        model = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")

    # Criar sub-agents se não fornecidos
    if safety_agent is None:
        logger.info("[Setup] Creating Safety Agent...")
        safety_agent = create_safety_agent(model=model)

    if context_agent is None:
        logger.info("[Setup] Creating Context Agent...")
        context_agent = create_context_agent(model=model)

    if personality_agent is None:
        logger.info("[Setup] Creating Personality Agent...")
        personality_agent = create_personality_agent(model=model)

    if knowledge_agent is None:
        logger.info("[Setup] Creating Knowledge Agent...")
        knowledge_agent = create_knowledge_agent(model=model)

    # System instruction para o Orchestrator
    instruction = """You are LIA, Inteli's friendly robot dog tour guide.

Process flow:
1. Check input safety (use safety_agent)
2. Retrieve context if needed (use context_agent)
3. For Inteli questions, get info (use knowledge_agent)
4. Check output safety (use safety_agent)
5. Respond in friendly tone with occasional [latido]

CRITICAL: Always validate safety before and after processing.

Agent usage:
- safety_agent: Input/output validation (PII, jailbreak, policy)
- context_agent: Conversation history and user profile
- personality_agent: Communication style adaptation
- knowledge_agent: Inteli information (courses, admissions, campus)

If agent fails, respond: "Desculpe [latido], tive um probleminha. Pode perguntar de novo?"
"""

    # Criar o Orchestrator Agent com sub-agents como tools
    orchestrator = Agent(
        name="orchestrator_agent",
        model=model,
        description="Main coordinator for the Inteli robot dog tour guide system. Validates safety, manages context, personalizes responses, and routes to specialized agents.",
        instruction=instruction,
        tools=[
            safety_agent,  # ✅ Agent como tool
            context_agent,  # ✅ Agent como tool
            personality_agent,  # ✅ Agent como tool
            knowledge_agent,  # ✅ Agent como tool
        ],
    )

    logger.info("[Setup] Orchestrator Agent created successfully")
    logger.info(f"[Setup] Model: {model}")
    logger.info("[Setup] Sub-agents: safety, context, personality, knowledge")

    return orchestrator


class OrchestratorAgent:
    """
    Wrapper para o Orchestrator Agent que mantém interface compatível.

    Esta classe encapsula o ADK Agent e fornece métodos convenientes
    para processar mensagens.
    """

    def __init__(
        self,
        model: str = None,
        safety_agent: Agent = None,
        context_agent: Agent = None,
        personality_agent: Agent = None,
        knowledge_agent: Agent = None,
    ):
        """
        Inicializa o Orchestrator Agent.

        Args:
            model: Modelo LLM
            safety_agent: Safety Agent opcional
            context_agent: Context Agent opcional
            personality_agent: Personality Agent opcional
            knowledge_agent: Knowledge Agent opcional
        """
        self.model = model or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")

        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no .env!")
        genai.configure(api_key=api_key)

        # Criar o Orchestrator Agent ADK-nativo
        self.agent = create_orchestrator_agent(
            model=self.model,
            safety_agent=safety_agent,
            context_agent=context_agent,
            personality_agent=personality_agent,
            knowledge_agent=knowledge_agent,
        )

        # Conversation history (backup local)
        self.conversation_history: list[dict[str, str]] = []

        logger.info("[Orchestrator] Initialized successfully")

    def process_message(self, user_message: str) -> str:
        """
        Processa mensagem do usuário usando o Orchestrator Agent.

        Esta versão usa o ADK Agent real que:
        - Valida safety automaticamente via safety_agent
        - Gerencia contexto via context_agent
        - Personaliza via personality_agent
        - Roteia para knowledge_agent quando necessário
        - Usa ToolContext real (não mock)

        Args:
            user_message: Mensagem do usuário

        Returns:
            str: Resposta do sistema
        """
        logger.info(f"[Orchestrator] Input: {user_message[:60]}...")

        try:
            # Usar o Orchestrator Agent para processar
            # O Agent automaticamente:
            # 1. Valida safety (input) via safety_agent
            # 2. Busca contexto via context_agent
            # 3. Analisa personalidade via personality_agent
            # 4. Roteia para knowledge_agent quando necessário
            # 5. Valida safety (output) via safety_agent
            # 6. Armazena contexto via context_agent

            response = self.agent.run(user_message)

            # Backup local
            self._add_to_history("user", user_message)
            self._add_to_history("assistant", response)

            return response

        except ValueError as e:
            # Input validation or parsing errors
            logger.warning(f"[Orchestrator] Validation error: {e}")
            return "Desculpe [latido], não consegui processar sua mensagem. Pode tentar reformular?"
        except KeyError as e:
            # Missing required data
            logger.error(f"[Orchestrator] Missing data: {e}", exc_info=True)
            return "Desculpe [latido], tive um problema interno. Pode tentar de novo?"
        except Exception as e:
            # Unexpected errors - log with full trace
            logger.critical(
                f"[Orchestrator] Unexpected error: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return (
                "Desculpe [latido], tive um probleminha técnico. Pode tentar novamente?"
            )

    def _add_to_history(self, role: str, content: str):
        """Adiciona mensagem ao histórico local."""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Retorna histórico de conversas."""
        return self.conversation_history

    def clear_history(self):
        """Limpa histórico."""
        self.conversation_history = []
