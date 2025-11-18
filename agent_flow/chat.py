#!/usr/bin/env python3
"""
Interactive CLI Chat Interface for the Inteli Robot Dog Tour Guide System

This script provides a conversational interface to the multi-agent orchestrator system.
The orchestrator coordinates Safety, Context, Personality, Knowledge, and Tour agents
to deliver an intelligent, safe, and personalized experience.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner

# Add agent_flow to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# pylint: disable=wrong-import-position
from agents.context_agent import create_context_agent  # noqa: E402
from agents.knowledge_agent import create_knowledge_agent  # noqa: E402
from agents.orchestrator_agent import create_orchestrator_agent  # noqa: E402
from agents.personality_agent import create_personality_agent  # noqa: E402
from agents.safety_agent import create_safety_agent  # noqa: E402

# Color codes for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


class ChatSession:
    """Manages a chat session with the orchestrator agent."""

    def __init__(self, model: Optional[str] = None, user_id: Optional[str] = None):
        """
        Initialize the chat session.

        Args:
            model: The model to use for the orchestrator (defaults to env DEFAULT_MODEL)
            user_id: User identifier for the session (defaults to 'cli_user')
        """
        self.model = model or os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        self.user_id = user_id or "cli_user"
        self.session_id = f"session_{self.user_id}"
        self.app_name = "inteli_robot_dog_tour_guide"

        # Initialize all agents
        print(f"{DIM}Initializing agent system...{RESET}")
        self._initialize_agents()
        print(f"{GREEN}✓ System ready!{RESET}\n")

    def _initialize_agents(self):
        """Initialize all specialized agents and the orchestrator."""
        # Create specialized agents
        print(f"{DIM}  → Creating Safety Agent...{RESET}")
        self.safety_agent = create_safety_agent()

        print(f"{DIM}  → Creating Context Agent...{RESET}")
        self.context_agent = create_context_agent()

        print(f"{DIM}  → Creating Personality Agent...{RESET}")
        self.personality_agent = create_personality_agent()

        print(f"{DIM}  → Creating Knowledge Agent...{RESET}")
        self.knowledge_agent = create_knowledge_agent()

        # Note: Tour agent is not implemented yet
        # self.tour_agent = create_tour_agent()

        # Create orchestrator with all agents as sub-agents
        print(f"{DIM}  → Creating Orchestrator Agent...{RESET}")
        self.orchestrator = create_orchestrator_agent(
            model=self.model,
            safety_agent=self.safety_agent,
            context_agent=self.context_agent,
            personality_agent=self.personality_agent,
            knowledge_agent=self.knowledge_agent,
            # tour_agent=self.tour_agent,  # Uncomment when implemented
        )

        # Create runner for the orchestrator
        # The orchestrator will coordinate all sub-agents
        self.runner = InMemoryRunner(
            agent=self.orchestrator,
            app_name=self.app_name,
        )

        # Store conversation history
        self.conversation_history = []

    def print_welcome(self):
        """Print welcome message."""
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(
            f"{BOLD}{CYAN}🐕 Inteli Robot Dog Tour Guide - Interactive Chat{RESET}".center(
                90
            )
        )
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")
        print(
            f"{YELLOW}Welcome! I'm your friendly robot dog tour guide at Inteli!{RESET}"
        )
        print(
            f"{DIM}I can help you learn about our facilities, programs, and navigate campus.{RESET}\n"
        )
        print(f"{DIM}Commands:{RESET}")
        print(f"{DIM}  • Type your message and press Enter to chat{RESET}")
        print(f"{DIM}  • Type 'exit', 'quit', or 'bye' to end the conversation{RESET}")
        print(f"{DIM}  • Type 'clear' to clear the screen{RESET}")
        print(f"{DIM}  • Type 'history' to see conversation history{RESET}\n")
        print(f"{BOLD}{BLUE}{'─' * 80}{RESET}\n")

    async def send_message(self, message: str) -> str:
        """
        Send a message to the orchestrator and get a response.

        Args:
            message: User's message

        Returns:
            The agent's response
        """
        try:
            # Run the orchestrator with the user message
            events = await self.runner.run_debug(
                user_messages=message,
                user_id=self.user_id,
                session_id=self.session_id,
                quiet=True,  # Suppress debug output
            )

            # Extract the response from events
            response_parts = []
            for event in events:
                # Get text content from the event
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_parts.append(part.text)

            response = "".join(response_parts).strip()

            if not response:
                response = "I'm sorry, I didn't quite catch that. Could you rephrase?"

            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            print(f"{RED}Error: {str(e)}{RESET}", file=sys.stderr)
            return error_msg

    def print_history(self):
        """Print conversation history."""
        if not self.conversation_history:
            print(f"{DIM}No conversation history yet.{RESET}\n")
            return

        print(f"\n{BOLD}{BLUE}{'─' * 80}{RESET}")
        print(f"{BOLD}{BLUE}Conversation History{RESET}")
        print(f"{BOLD}{BLUE}{'─' * 80}{RESET}\n")

        for turn in self.conversation_history:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                print(f"{BOLD}{CYAN}You:{RESET} {content}")
            else:
                print(f"{BOLD}{GREEN}🐕 Robot Dog:{RESET} {content}")
            print()

        print(f"{BOLD}{BLUE}{'─' * 80}{RESET}\n")

    async def run(self):
        """Run the interactive chat loop."""
        self.print_welcome()

        while True:
            try:
                # Get user input
                user_input = input(f"{BOLD}{CYAN}You:{RESET} ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print(
                        f"\n{GREEN}🐕 Goodbye! Come back soon to visit Inteli!{RESET}\n"
                    )
                    break

                if user_input.lower() == "clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    self.print_welcome()
                    continue

                if user_input.lower() == "history":
                    self.print_history()
                    continue

                # Send message to orchestrator
                print(f"{DIM}Thinking...{RESET}", end="\r")
                response = await self.send_message(user_input)

                # Print response
                print(f"{BOLD}{GREEN}🐕 Robot Dog:{RESET} {response}\n")

            except KeyboardInterrupt:
                print(
                    f"\n\n{YELLOW}Chat interrupted. Type 'exit' to quit gracefully.{RESET}\n"
                )
                continue
            except EOFError:
                print(f"\n{GREEN}🐕 Goodbye! Come back soon to visit Inteli!{RESET}\n")
                break
            except Exception as e:
                print(f"\n{RED}Error: {str(e)}{RESET}\n")
                import traceback

                traceback.print_exc()


async def main():
    """Main entry point for the chat CLI."""
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            f"{RED}Error: GOOGLE_API_KEY environment variable not set.{RESET}",
            file=sys.stderr,
        )
        print(
            f"{YELLOW}Please set your Google API key in the .env file.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create and run chat session
    session = ChatSession()
    await session.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Chat terminated.{RESET}")
        sys.exit(0)
