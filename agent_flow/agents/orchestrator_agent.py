"""Orchestrator Agent - Orchestrates all other agents."""

import os
import sys

from google.adk.agents import Agent

# Add parent directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Orchestrator typically doesn't need tools - it delegates to other agents
# But we can add some orchestration utilities if needed


def create_orchestrator_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Orchestrator Agent.

    This agent orchestrates all other agents (safety, context, personality, knowledge, tour).

    Args:
        model: The LLM model to use

    Returns:
        Configured Orchestrator Agent
    """
    # TODO: Add detailed instruction prompt
    instruction = """
Orchestrator agent instruction placeholder.
"""

    agent = Agent(
        name="orchestrator_agent",
        model=model,
        description="Orchestrates all agents and manages conversation flow",
        instruction=instruction,
        tools=[],  # Orchestrator delegates, doesn't use tools directly
    )

    return agent
