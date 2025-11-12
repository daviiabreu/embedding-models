"""Context Agent for knowledge retrieval."""

import os
import sys

from google.adk.agents import Agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.document_tools import get_user_preferences, search_knowledge_base


def create_context_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Context Agent.

    This agent retrieves relevant information from the knowledge base
    to help answer user queries.

    Args:
        model: The LLM model to use

    Returns:
        Configured Context Agent
    """
    instruction = """
You manage the robot dog's knowledge and context.

Your job is to:
1. Search the knowledge base when the user asks questions
2. Retrieve user preferences to personalize responses
3. Provide relevant context to help answer queries

Use these tools:
- search_knowledge_base: Search for information about robot dog features, care, commands, etc.
- get_user_preferences: Get user's preferences and interaction history

When you find relevant information:
- Summarize it clearly
- Include specific details
- Cite the source (e.g., "From the knowledge base...")

If no relevant information is found:
- Say "I don't have specific information about that"
- Suggest related topics that might help
"""

    agent = Agent(
        name="context_agent",
        model=model,
        description="Retrieves knowledge and context",
        instruction=instruction,
        tools=[search_knowledge_base, get_user_preferences],
    )

    return agent
