from dotenv import load_dotenv

from backend.agent_flow.agents.orchestrator_agent import OrchestratorAgent

load_dotenv("../.env", override=False)


class ChatService:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()

    def give_response(self, prompt: str):
        response = self.orchestrator.process_message(prompt.lower())

        return response
