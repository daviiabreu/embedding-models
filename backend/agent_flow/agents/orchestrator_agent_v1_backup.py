import json
import logging
import os
from typing import Dict, List

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
    from backend.agent_flow.agents.tour_agent import create_tour_agent
except ImportError:
    # Fallback to relative imports
    from .context_agent import create_context_agent
    from .knowledge_agent import create_knowledge_agent
    from .personality_agent import create_personality_agent
    from .safety_agent import create_safety_agent
    from .tour_agent import create_tour_agent

# Carrega ambiente - try both locations
load_dotenv("backend/agent_flow/.env")
load_dotenv(".env")


class OrchestratorAgent:
    def __init__(
        self,
        model: str = None,
        safety_agent: Agent = None,
        context_agent: Agent = None,
        personality_agent: Agent = None,
        knowledge_agent: Agent = None,
        tour_agent=None,
    ):
        """
        Initialize the Orchestrator Agent.

        Args:
            model: Model to use (defaults to env DEFAULT_MODEL)
            safety_agent: Pre-created safety agent (optional)
            context_agent: Pre-created context agent (optional)
            personality_agent: Pre-created personality agent (optional)
            knowledge_agent: Pre-created knowledge agent (optional)
            tour_agent: Pre-created tour agent (optional)
        """
        # 1. Configuração
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no .env!")

        self.model = model or os.getenv("DEFAULT_MODEL")

        # Configure Gemini for direct calls
        genai.configure(api_key=self.api_key)
        self.llm = genai.GenerativeModel(self.model)

        # 2. Conversation History (lightweight backup, Context Agent manages memory)
        self.conversation_history: List[Dict[str, str]] = []

        # 3. Initialize specialized agents (or use provided ones)
        if safety_agent is None:
            self.safety_agent = create_safety_agent(model=self.model)
        else:
            self.safety_agent = safety_agent

        if context_agent is None:
            self.context_agent = create_context_agent(model=self.model)
        else:
            self.context_agent = context_agent

        if personality_agent is None:
            self.personality_agent = create_personality_agent(model=self.model)
        else:
            self.personality_agent = personality_agent

        if knowledge_agent is None:
            self.knowledge_agent = create_knowledge_agent(model=self.model)
        else:
            self.knowledge_agent = knowledge_agent

        if tour_agent is None:
            self.tour_agent = create_tour_agent()
        else:
            self.tour_agent = tour_agent

        # 4. Create the main orchestrator agent
        self.agent = self._create_orchestrator_agent()

    def _create_orchestrator_agent(self) -> Agent:
        """Creates the main orchestrator agent with proper instructions."""
        instruction = """
You are the Orchestrator Agent for the Inteli robot dog tour guide system. Your role is to coordinate between specialized agents and manage the overall conversation flow.

## Core Responsibilities

1. **Route user requests** to the appropriate specialized agent:
   - **Tour Agent**: For navigation commands (start tour, next, stop)
   - **Knowledge Agent**: For questions about Inteli (courses, scholarships, people, facilities)
   - **Context Agent**: For managing conversation memory and context

2. **Maintain conversation continuity** by using the Context Agent to:
   - Store each interaction in memory
   - Retrieve relevant context from previous messages
   - Build user profiles over time

3. **Coordinate agent responses** to provide coherent, contextual answers

## Decision Flow

For each user message:

1. **Check Tour State**: If tour is active and user says navigation commands, route to Tour Agent
2. **Classify Intent**:
   - NAV_START: "começar tour", "iniciar visita", "vamos" → Tour Agent
   - NAV_NEXT: "próximo", "continuar", "avançar" (only if tour active) → Tour Agent
   - NAV_STOP: "parar", "sair", "encerrar" (only if tour active) → Tour Agent
   - KNOWLEDGE: Questions about Inteli → Knowledge Agent
   - CHITCHAT: Greetings, small talk → Direct response

3. **Delegate to Agent**: Call the appropriate agent with full context
4. **Store in Memory**: Use Context Agent to save the interaction

## Agent Usage Guidelines

### Tour Agent
- Simple state machine (not an ADK agent)
- Call directly: `self.tour_agent.process_command(message)`
- Returns JSON with speech and action

### Knowledge Agent
- ADK agent with RAG capabilities
- Use for factual questions about Inteli
- Automatically uses hybrid search (semantic + keyword)
- Returns structured knowledge with sources

### Context Agent
- ADK agent with memory management tools
- Call at the START of each interaction to retrieve context
- Call at the END of each interaction to store the new message
- Use to build user profiles and track topics

## Conversation History Format

Store each interaction as:
```python
{
    "role": "user" | "assistant",
    "content": "message text",
    "timestamp": datetime,
    "intent": "detected intent",
    "agent_used": "which agent handled this"
}
```

## Important Notes

- ALWAYS use Context Agent to manage memory
- Tour Agent state (is_active, current_step_index) is important for context
- When tour is interrupted for questions, remind user they can "continuar" the tour
- Be friendly and helpful, embodying the LIA personality
"""

        # Note: The orchestrator doesn't need tools itself - it delegates to other agents
        agent = Agent(
            name="orchestrator_agent",
            model=self.model,
            description="Main coordinator for the Inteli robot dog tour guide system",
            instruction=instruction,
            tools=[],  # Orchestrator delegates, doesn't use tools directly
        )

        return agent

    def _add_to_history(
        self, role: str, content: str, intent: str = "", agent_used: str = ""
    ):
        """Add a message to conversation history."""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "intent": intent,
                "agent_used": agent_used,
            }
        )

    def _get_recent_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:]

    def _format_history_for_context(self) -> str:
        """Format conversation history as a string for context."""
        if not self.conversation_history:
            return "No previous conversation."

        history_lines = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            role = "User" if msg["role"] == "user" else "Dog"
            history_lines.append(f"{role}: {msg['content']}")

        return "\n".join(history_lines)

    def _decide_intent(self, user_input: str) -> str:
        """Classifica a intenção do usuário."""
        # Se o tour estiver ativo, verificamos comandos de controle
        tour_context = ""
        if self.tour_agent.is_active:
            tour_context = "O TOUR ESTÁ ATIVO AGORA."
            if any(
                x in user_input.lower()
                for x in ["parar", "sair", "tchau", "fim", "encerrar"]
            ):
                return "TOUR_CONTROL"

        # Get conversation context
        history_context = self._format_history_for_context()

        prompt = f"""
Você é o cérebro de classificação do robô cachorro LIA do Inteli.
{tour_context}

Histórico recente:
{history_context}

Classifique a entrada do usuário em UMA das categorias:

1. NAV_START: Usuário quer COMEÇAR o tour/visita/passeio.
2. NAV_NEXT: Usuário quer PRÓXIMO ponto, continuar, avançar (apenas se tour ativo).
3. NAV_STOP: Usuário quer PARAR, sair (apenas se tour ativo).
4. KNOWLEDGE: Perguntas sobre Inteli, cursos, bolsas, pessoas (Roberto Sallouti, Ana Garcia), história, regras.
5. CHITCHAT: Conversa fiada, oi, tudo bem.

Entrada atual: "{user_input}"

Responda APENAS a palavra da categoria.
"""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip().upper().replace(".", "")
        except Exception as e:
            print(f"⚠️  Intent classification error: {e}")
            return "CHITCHAT"

    def _validate_input_safety(self, user_message: str) -> Dict:
        """Stage 1: Validate input using Safety Agent with REAL safety tools."""
        try:
            logger.info("[Orchestrator] → Safety Agent")

            # Import safety tools
            import sys

            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from tools.safety_tools import (
                check_moderation,
                check_off_topic,
                detect_jailbreak,
                detect_nsfw_text,
                mask_pii,
            )

            # Create a simple mock ToolContext for tools
            # (ToolContext requires invocation_context which we don't have in this flow)
            class MockToolContext:
                def __init__(self):
                    self.state = {}

            tool_context = MockToolContext()

            # 1. Check for PII
            logger.info("  [Tool] mask_pii")
            pii_result = mask_pii(user_message, tool_context)
            if pii_result.get("pii_detected", False):
                logger.warning(
                    f"[SAFETY] PII detected: {pii_result.get('pii_types', [])}"
                )

            # 2. Check if off-topic (not related to Inteli tour)
            logger.info("  [Tool] check_off_topic")
            off_topic_result = check_off_topic(
                user_message,  # text
                "Inteli campus tour guide, information about Inteli institute, courses, scholarships, people, facilities, admission process",  # business_scope
                tool_context,  # tool_context
            )

            if off_topic_result.get("is_off_topic", False):
                logger.warning("  ⚠️  OFF-TOPIC - BLOCKED")
                return {
                    "safe": False,
                    "action": "block",
                    "user_message": "Desculpe [latido], não posso ajudar com isso. Sou especialista em informações sobre o Inteli, nossos cursos, processo seletivo e campus. Tem alguma dúvida sobre isso?",
                }

            # 3. Check for jailbreak attempts
            logger.info("  [Tool] detect_jailbreak")
            jailbreak_result = detect_jailbreak(user_message, tool_context)
            if jailbreak_result.get("is_jailbreak", False):
                logger.warning("  ⚠️  JAILBREAK - BLOCKED")
                return {
                    "safe": False,
                    "action": "block",
                    "user_message": "Desculpe, não posso responder a isso. Vamos manter nossa conversa educativa! [latido]",
                }

            # 4. Check for NSFW content
            logger.info("[SAFETY TOOL] detect_nsfw_text")
            nsfw_result = detect_nsfw_text(user_message, tool_context)
            if nsfw_result.get("is_nsfw", False):
                logger.warning("[SAFETY] ⚠️  NSFW content detected - BLOCKING")
                logger.info("=" * 80)
                return {
                    "safe": False,
                    "action": "block",
                    "user_message": "Desculpe, não posso responder a isso. Vamos manter nossa conversa educativa! [latido]",
                }

            # 5. Check moderation (toxicity, hate speech, etc.)
            logger.info("[SAFETY TOOL] check_moderation")
            mod_result = check_moderation(user_message, tool_context)
            if not mod_result.get("is_safe", True):
                logger.warning("[SAFETY] ⚠️  TOXIC content detected - BLOCKING")
                logger.info("=" * 80)
                return {
                    "safe": False,
                    "action": "block",
                    "user_message": "Desculpe, não posso responder a isso. Vamos manter nossa conversa respeitosa! [latido]",
                }

            logger.info("  ✓ Safe")
            return {"safe": True, "action": "allow"}

        except Exception as e:
            logger.error(f"[SAFETY AGENT] ⚠️  Error running safety checks: {e}")
            import traceback

            logger.error(traceback.format_exc())
            logger.info("=" * 80)
            # Default to safe on error to avoid blocking legitimate queries
            return {"safe": True, "action": "allow"}

    def _manage_context_memory(self, user_message: str) -> str:
        """Stage 2: Retrieve relevant conversation context."""
        try:
            logger.info("=" * 80)
            logger.info("[CONTEXT AGENT] Retrieving conversation context")
            logger.info(
                f"[CONTEXT AGENT] History size: {len(self.conversation_history)} messages"
            )

            context = self._format_history_for_context()

            logger.info(f"[CONTEXT AGENT] Retrieved: {len(context)} chars")
            logger.info("=" * 80)
            return context
        except Exception as e:
            logger.error(f"[CONTEXT AGENT] ⚠️  Error: {e}")
            logger.info("=" * 80)
            return "No previous context."

    def _detect_personality_and_adapt(self, user_message: str, context: str) -> Dict:
        """Stage 3: Detect user personality and recommend tone."""
        try:
            logger.info("=" * 80)
            logger.info("[PERSONALITY AGENT] Analyzing user style")
            logger.info(
                f"[PERSONALITY AGENT] Message length: {len(user_message)} chars"
            )

            # Simplified personality detection
            if len(user_message) > 100:
                style = "detailed"
            elif any(q in user_message.lower() for q in ["?", "como", "qual", "quem"]):
                style = "curious"
            else:
                style = "conversational"

            logger.info(f"[PERSONALITY AGENT] Detected style: {style}")
            logger.info("[PERSONALITY AGENT] Recommended tone: friendly")
            logger.info("=" * 80)

            return {
                "tone": "friendly",
                "style": style,
                "recommendations": "be friendly and helpful",
            }
        except Exception as e:
            logger.error(f"[PERSONALITY AGENT] ⚠️  Error: {e}")
            logger.info("=" * 80)
            return {"tone": "friendly", "style": "conversational"}

    def _validate_output_safety(self, response: str, context: str) -> Dict:
        """Stage 6: Validate output safety (simplified)."""
        try:
            # Simple output validation - check for obviously problematic content
            unsafe_keywords = ["hate", "violence", "explicit", "harmful"]
            response_lower = response.lower()

            if any(keyword in response_lower for keyword in unsafe_keywords):
                return {
                    "safe": False,
                    "action": "block",
                    "response": "Desculpe, não posso fornecer essa resposta.",
                }

            return {"safe": True, "action": "allow", "response": response}
        except Exception as e:
            print(f"⚠️  Output safety error: {e}, allowing response")
            return {"safe": True, "action": "allow", "response": response}

    def process_message(self, user_message: str) -> str:
        """Main processing flow with full multi-agent pipeline."""

        logger.info(f"[Orchestrator] Input: {user_message[:60]}...")

        # STAGE 1: Safety Input Validation
        safety_check = self._validate_input_safety(user_message)
        if not safety_check.get("safe", False) or safety_check.get("action") == "block":
            logger.warning("  Input blocked")
            blocked_msg = safety_check.get(
                "user_message",
                "Desculpe, não posso responder a isso. Vamos manter nossa conversa educativa e respeitosa! [latido]",
            )
            self._add_to_history("user", user_message, "blocked", "safety")
            self._add_to_history("assistant", blocked_msg, "blocked", "safety")
            return blocked_msg

        # STAGE 2: Context Management
        relevant_context = self._manage_context_memory(user_message)

        # STAGE 3: Personality Analysis
        personality_info = self._detect_personality_and_adapt(
            user_message, relevant_context
        )

        # Add user message to local history
        self._add_to_history("user", user_message)

        # STAGE 4: Intent Classification with context
        intent = self._decide_intent(user_message)
        logger.info(f"[Orchestrator] Intent: {intent}")

        # STAGE 5: Route to Appropriate Agent
        response_text = ""
        agent_used = ""

        # --- ROTA 1: TOUR ---
        if intent in ["NAV_START", "NAV_NEXT", "NAV_STOP", "TOUR_CONTROL"] or (
            self.tour_agent.is_active and intent not in ["KNOWLEDGE", "CHITCHAT"]
        ):
            logger.info("[Orchestrator] → Tour Agent")

            response_json = self.tour_agent.process_command(user_message)
            try:
                data = json.loads(response_json)
                response_text = data["speech"]
                agent_used = "tour_agent"
            except Exception as e:
                logger.error(f"[TOUR AGENT] ⚠️  Parse error: {e}")
                logger.info("=" * 80)
                response_text = response_json
                agent_used = "tour_agent"

        # --- ROTA 2: KNOWLEDGE ---
        elif intent == "KNOWLEDGE":
            logger.info("[Orchestrator] → Knowledge Agent")
            try:
                tour_context = ""
                if self.tour_agent.is_active:
                    try:
                        current_step = self.tour_agent.script[
                            self.tour_agent.current_step_index
                        ]
                        local = current_step["local"]
                        tour_context = f"TOUR CONTEXT: User at '{local}' during tour."
                    except Exception:
                        tour_context = "TOUR CONTEXT: User in active tour."

                # Use knowledge agent's RAG pipeline directly
                try:
                    logger.info("  [Tool] rag_inference_pipeline")

                    # Try multiple import paths for compatibility
                    try:
                        from agent_flow.tools.knowledge_tools import (
                            rag_inference_pipeline,
                        )
                    except ImportError:
                        import sys

                        sys.path.insert(
                            0,
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        )
                        from tools.knowledge_tools import rag_inference_pipeline

                    # Get knowledge from RAG
                    rag_result = rag_inference_pipeline(user_message)

                    # Extract context from RAG result
                    rag_context = rag_result.get(
                        "context", "No relevant information found."
                    )

                    # Generate response using LLM with RAG context
                    knowledge_prompt = f"""
{tour_context}
Conversation context: {relevant_context}

User question: {user_message}

Knowledge base information:
{rag_context}

CRITICAL INSTRUCTIONS:
1. You are Lia, the friendly Inteli robot dog.
2. Answer ONLY using information from the "Knowledge base information" above.
3. DO NOT invent, guess, or hallucinate ANY information (names, titles, roles, facts).
4. Copy names, titles, and facts EXACTLY as they appear in the knowledge base - do not paraphrase or change them.
5. Be helpful and friendly, include [latido] for personality.
6. If you're unsure about something, acknowledge the uncertainty rather than guessing.

RESPONSE STYLE:
- Be OBJECTIVE and CONCISE - get straight to the point
- Use CONVERSATIONAL tone - talk naturally, like a friendly guide
- DO NOT include URLs, links, or web addresses in your response
- DO NOT cite document names, page numbers, or technical references
- DO NOT use emojis (🐕, ✨, etc.) - NEVER use any emoji characters
- Just share the information in a natural, friendly way
- Keep responses short and to the point unless more detail is requested

Examples of GOOD responses:
- "Maíra Habimorad é a CEO do Inteli desde 2020 [latido]. Ela tem mais de 20 anos de experiência em cargos de liderança."
- "Não tenho essa informação específica [latido], mas posso te ajudar com outras dúvidas sobre o Inteli!"

Examples of BAD responses (DO NOT DO THIS):
- "Segundo o documento XYZ.pdf, página 5..."
- "Você pode encontrar mais em https://..."
- Making up information not in the knowledge base
- Long, formal, academic-style responses

Now answer the user's question using ONLY the knowledge base provided above.
"""
                    logger.info(
                        f"[LLM CALL] Prompt size: {len(knowledge_prompt)} chars"
                    )
                    response = self.llm.generate_content(knowledge_prompt)
                    response_text = response.text
                    agent_used = "knowledge_agent"
                    logger.info(
                        f"[LLM RESULT] Response generated: {len(response_text)} chars"
                    )
                    logger.info(
                        f"[LLM RESULT] First 100 chars: {response_text[:100]}..."
                    )
                    logger.info("=" * 80)
                except ImportError as ie:
                    logger.error("=" * 80)
                    logger.error("[ERROR] RAG tools import failed!")
                    logger.error(f"[ERROR] ImportError: {ie}")
                    logger.error("[ERROR] Using fallback mode (no RAG)")
                    logger.error("=" * 80)
                    # Fallback if tool not available
                    response = self.llm.generate_content(f"""
You are LIA, the Inteli robot dog.
User asked: {user_message}

Answer helpfully about Inteli in a CONVERSATIONAL and OBJECTIVE way.
DO NOT include URLs or links.
DO NOT use emojis.
Keep it brief and friendly.
Include [latido].
""")
                    response_text = response.text
                    agent_used = "knowledge_fallback"

            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"[KNOWLEDGE ERROR] Exception type: {type(e).__name__}")
                logger.error(f"[KNOWLEDGE ERROR] Message: {str(e)}")
                import traceback

                logger.error(f"[KNOWLEDGE ERROR] Traceback:\n{traceback.format_exc()}")
                logger.error("=" * 80)
                response_text = (
                    f"Desculpa [latido], erro ao buscar informação: {str(e)}"
                )
                agent_used = "error"

        # --- ROTA 3: CHITCHAT ---
        else:
            logger.info("[Orchestrator] → Chitchat (Direct LLM)")

            chat_prompt = f"""
You are LIA, the Inteli robot dog.
Context: {relevant_context}
Personality guidance: {personality_info.get("recommendations", "be friendly and brief")}

User: "{user_message}"

RESPONSE STYLE:
- Be OBJECTIVE and BRIEF - keep it short and conversational
- Use a FRIENDLY, natural tone - like chatting with a friend
- DO NOT include URLs, links, or web addresses
- DO NOT use emojis (🐕, ✨, etc.) - NEVER use any emoji characters
- DO NOT be overly formal or academic
- Include [latido] for personality
- Offer help with Inteli information or campus tours

Respond in Portuguese.
"""
            try:
                logger.info(f"[CHITCHAT MODE] Generating response with {self.model}")
                response = self.llm.generate_content(chat_prompt)
                response_text = response.text
                agent_used = "chitchat"
                logger.info(f"[CHITCHAT MODE] ✓ Response: {len(response_text)} chars")
                logger.info("=" * 80)
            except Exception as e:
                logger.error(f"[CHITCHAT MODE] ⚠️  Error: {e}")
                logger.info("=" * 80)
                response_text = "Oi! [latido] Como posso ajudar? Posso responder dúvidas sobre o Inteli ou fazer um tour!"
                agent_used = "fallback"

        # STAGE 6: Output Safety Validation
        safety_result = self._validate_output_safety(response_text, relevant_context)
        if (
            not safety_result.get("safe", True)
            or safety_result.get("action") == "block"
        ):
            logger.warning("[SAFETY] Output blocked")
            response_text = "Desculpe, não posso fornecer essa resposta. Posso ajudar com outra dúvida? [latido]"
            agent_used = "safety_blocked"

        # STAGE 7: Store in Context Memory
        self._add_to_history("assistant", response_text, intent, agent_used)

        return response_text

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


def create_orchestrator_agent(
    model: str = None,
    safety_agent: Agent = None,
    context_agent: Agent = None,
    personality_agent: Agent = None,
    knowledge_agent: Agent = None,
    tour_agent=None,
) -> Agent:
    """
    Factory function to create an orchestrator agent.

    This is a wrapper around OrchestratorAgent class for compatibility with
    different architectures (e.g., chat.py using InMemoryRunner).

    Args:
        model: Model to use (defaults to env DEFAULT_MODEL)
        safety_agent: Pre-created safety agent (optional)
        context_agent: Pre-created context agent (optional)
        personality_agent: Pre-created personality agent (optional)
        knowledge_agent: Pre-created knowledge agent (optional)
        tour_agent: Pre-created tour agent (optional)

    Returns:
        The orchestrator's internal Agent instance (for use with runners)
    """
    orchestrator = OrchestratorAgent(
        model=model,
        safety_agent=safety_agent,
        context_agent=context_agent,
        personality_agent=personality_agent,
        knowledge_agent=knowledge_agent,
        tour_agent=tour_agent,
    )
    # Return the internal agent for compatibility with ADK runners
    return orchestrator.agent


if __name__ == "__main__":
    orch = OrchestratorAgent()
    print("System ready. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
            if q.lower() in ["sair", "exit", "quit"]:
                break
            if not q:
                continue

            response = orch.process_message(q)
            print(f"Assistant: {response}\n")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {e}\n")
