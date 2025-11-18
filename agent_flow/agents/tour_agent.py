import os
import sys

from google.adk.agents import Agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: Add tour tools


def create_tour_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    instruction = """
Instruction
"""

    agent = Agent(
        name="tour_agent",
        model=model,
        description="Handles physical tour guidance, navigation, and location information",
        instruction=instruction,
        tools=[],  # TODO: Add tools
    )

    return agent
