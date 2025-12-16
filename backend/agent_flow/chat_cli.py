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

from dotenv import load_dotenv

# Add agent_flow to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# pylint: disable=wrong-import-position
from agents.orchestrator_agent_v3 import OrchestratorAgent  # noqa: E402

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

    def __init__(self, model: str | None = None, user_id: str | None = None):
        """
        Initialize the chat session.

        Args:
            model: The model to use for the orchestrator (defaults to env DEFAULT_MODEL)
            user_id: User identifier for the session (defaults to 'cli_user')
        """
        self.model = model or os.getenv("DEFAULT_MODEL")
        self.user_id = user_id or "cli_user"
        self.session_id = f"session_{self.user_id}"
        self.app_name = "inteli_robot_dog_tour_guide"

        # Initialize all agents
        print(f"{DIM}Initializing agent system...{RESET}")
        self._initialize_agents()
        print(f"{GREEN}✓ System ready!{RESET}\n")

    def _initialize_agents(self):
        """Initialize the orchestrator agent (which manages all sub-agents internally)."""
        # Create orchestrator - it will initialize all sub-agents internally
        print(f"{DIM}  → Creating Orchestrator Agent...{RESET}")
        self.orchestrator = OrchestratorAgent(
            model=self.model,
            user_id=self.user_id,
            session_id=self.session_id,
        )

        # Store conversation history
        self.conversation_history = []

    def print_welcome(self):
        """Print welcome message."""
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(f"{BOLD}{CYAN}Interactive Chat{RESET}".center(90))
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")
        print()

    async def send_message(self, message: str) -> str:
        """
        Send a message to the orchestrator and get a response.

        Args:
            message: User's message

        Returns:
            The agent's response
        """
        try:
            # Call the orchestrator's process_message method directly
            response = self.orchestrator.process_message(message)

            if not response:
                response = "I'm sorry, I didn't quite catch that. Could you rephrase?"

            # Note: conversation history is managed by the orchestrator itself
            # We keep a copy here for the 'history' command
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            print(f"{RED}Error: {str(e)}{RESET}", file=sys.stderr)
            import traceback

            traceback.print_exc()
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
                print(f"{BOLD}{GREEN}🐕 LIA:{RESET} {content}")
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

                if user_input.lower() in {"exit", "quit", "bye"}:
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
                print(f"{BOLD}{GREEN}🐕 LIA:{RESET} {response}\n")

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
