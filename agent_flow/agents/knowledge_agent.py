import os
import sys

from google.adk.agents import Agent

# Add parent directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.knowledge_tools import (
    answer_question,
    get_specific_info,
    search_inteli_knowledge,
)


def create_knowledge_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Knowledge Agent.

    This agent handles RAG-based information retrieval about Inteli.

    Args:
        model: The LLM model to use

    Returns:
        Configured Knowledge Agent
    """
    # TODO: Add detailed instruction prompt
    instruction = """
Knowledge agent instruction placeholder.
"""

    agent = Agent(
        name="knowledge_agent",
        model=model,
        description="RAG-powered knowledge retrieval specialist for Inteli information",
        instruction=instruction,
        tools=[
            search_inteli_knowledge,
            get_specific_info,
            answer_question,
        ],
    )

    return agent
