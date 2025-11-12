"""Context Agent for managing knowledge retrieval and conversation memory."""

import os
import sys

from google.adk.agents import Agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.context_tools import (
    build_context_profile,
    detect_context_gaps,
    manage_context,
    manage_conversation_memory,
    retrieve_relevant_context,
    track_topics_discussed,
)


def create_context_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Context Agent.

    This agent manages knowledge retrieval, conversation memory,
    and context preparation for the robot dog tour guide.

    Args:
        model: The LLM model to use

    Returns:
        Configured Context Agent
    """
    # TODO: Add detailed instruction prompt
    instruction = """
Context agent instruction placeholder.
"""

    agent = Agent(
        name="context_agent",
        model=model,
        description="Manages knowledge retrieval, conversation memory, and context preparation",
        instruction=instruction,
        tools=[
            retrieve_relevant_context,
            manage_conversation_memory,
            track_topics_discussed,
            detect_context_gaps,
            build_context_profile,
            manage_context,
        ],
    )

    return agent
