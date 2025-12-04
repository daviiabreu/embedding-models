from dotenv import load_dotenv

# Import with fallback for different execution contexts
try:
    from backend.agent_flow.agents.orchestrator_agent import OrchestratorAgent
except ImportError:
    from .agents.orchestrator_agent import OrchestratorAgent

# Load .env from current directory or parent locations
load_dotenv(".env", override=False)  # backend/agent_flow/.env
load_dotenv("../../.env", override=False)  # project root .env if it exists


class ChatService:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()

    def give_response(self, prompt: str):
        response = self.orchestrator.process_message(prompt.lower())

        return response
