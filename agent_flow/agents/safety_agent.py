"""Safety Agent for content validation."""

import os
import sys

from google.adk.agents import Agent

# Add parent directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.safety_tools import (
    check_content_safety,
    check_output_safety,
    detect_jailbreak,
    detect_nsfw_text,
)


def create_safety_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Safety Agent.

    This agent validates all user inputs and outputs for safety.

    Args:
        model: The LLM model to use

    Returns:
        Configured Safety Agent
    """
    # TODO: Add detailed instruction prompt
    instruction = """
Safety agent instruction placeholder.
"""

    agent = Agent(
        name="safety_agent",
        model=model,
        description="Validates user inputs and outputs for safety",
        instruction=instruction,
        tools=[
            check_content_safety,
            check_output_safety,
            detect_jailbreak,
            detect_nsfw_text,
        ],
    )

    return agent
