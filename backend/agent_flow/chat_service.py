from dotenv import load_dotenv

# Import with fallback for different execution contexts
try:
    from backend.agent_flow.agents.orchestrator_agent import OrchestratorAgent
except ImportError:
    from .agents.orchestrator_agent import OrchestratorAgent

try:
    from backend.agent_flow.utils import ValidationError, validate_user_input
except ImportError:
    from .utils import ValidationError, validate_user_input

# Load .env from current directory or parent locations
load_dotenv(".env", override=False)  # backend/agent_flow/.env
load_dotenv("../../.env", override=False)  # project root .env if it exists


class ChatService:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()

    def give_response(self, prompt: str):
        # Validate input before processing
        try:
            validate_user_input(prompt)
        except ValidationError as e:
            return f"Desculpe [latido], sua mensagem não pôde ser processada: {str(e)}"

        response = self.orchestrator.process_message(prompt.lower())

        return response
