"""Safety Agent for content validation."""

from google.adk.agents import Agent
import sys
import os

# Add parent directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.safety_tools import check_content_safety


def create_safety_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Safety Agent.

    This agent validates all user inputs for safety before processing.

    Args:
        model: The LLM model to use

    Returns:
        Configured Safety Agent
    """
    instruction = """
You are the safety guardian for the robot dog system.

Your job is to:
1. Check if user messages are safe and appropriate
2. Block harmful, dangerous, or inappropriate content
3. Ensure all interactions are family-friendly

Use the check_content_safety tool to validate inputs.

If content is unsafe:
- Return: "UNSAFE: [reason]"

If content is safe:
- Return: "SAFE"

Be cautious but not overly restrictive. Normal friendly conversation is fine.
"""

    agent = Agent(
        name="safety_agent",
        model=model,
        description="Validates user inputs for safety",
        instruction=instruction,
        tools=[check_content_safety]
    )

    return agent
