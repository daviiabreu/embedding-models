import os
import sys

from google.adk.agents import Agent

# Add parent directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.personality_tools import (
    adapt_tone,
    apply_personality_adaptations,
    build_personality_profile,
    detect_communication_style,
    detect_emotional_state,
    detect_engagement_level,
    detect_personality_type,
)


def create_personality_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Personality Agent.

    This agent detects user personality and adapts responses accordingly.

    Args:
        model: The LLM model to use

    Returns:
        Configured Personality Agent
    """
    # TODO: Add detailed instruction prompt
    instruction = """
Personality agent instruction placeholder.
"""

    agent = Agent(
        name="personality_agent",
        model=model,
        description="Detects user personality and adapts responses for personalization",
        instruction=instruction,
        tools=[
            detect_personality_type,
            detect_communication_style,
            detect_emotional_state,
            detect_engagement_level,
            build_personality_profile,
            adapt_tone,
            apply_personality_adaptations,
        ],
    )

    return agent
