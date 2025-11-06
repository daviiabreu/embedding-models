"""
Inteli Robot Dog Tour Guide Application

This is the main application that runs the robot dog tour guide using Google ADK.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from robot_dog_adk.agents.coordinator_agent import create_enhanced_coordinator


class InteliRobotDogApp:
    """Main application class for the Inteli Robot Dog tour guide."""

    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize the robot dog application.

        Args:
            model: LLM model to use (default: gemini-2.0-flash-exp)
        """
        self.model = model
        self.app_name = "inteli_robot_dog_tour"

        # Create session service
        self.session_service = InMemorySessionService()

        # Create the main coordinator agent
        print("🐕 Initializing Inteli Robot Dog Tour Guide...")
        self.agent = create_enhanced_coordinator(model)
        print("✅ Agent created successfully!")

        # Create runner
        self.runner = InMemoryRunner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        print("✅ Runner initialized!")

    async def start_tour(self, visitor_name: str = "visitante") -> None:
        """
        Start a new tour session.

        Args:
            visitor_name: Name of the visitor (for personalization)
        """
        user_id = f"visitor_{visitor_name}"
        session_id = f"session_{visitor_name}"

        # Create session
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )

        print(f"\n{'='*60}")
        print(f"🐕 INTELI ROBOT DOG TOUR GUIDE")
        print(f"{'='*60}\n")

        # Initial greeting
        greeting = types.Content(
            role="user",
            parts=[types.Part(text="Olá! Estou aqui para o tour do Inteli!")]
        )

        print(f"👤 {visitor_name}: Olá! Estou aqui para o tour do Inteli!\n")
        print("🐕 Robot Dog: ", end="", flush=True)

        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=greeting
        ):
            if event.is_final_response():
                response_text = event.content.parts[0].text
                print(response_text)
                print()

        return user_id, session_id

    async def send_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        visitor_name: str = "Visitante"
    ) -> str:
        """
        Send a message to the robot dog.

        Args:
            user_id: User identifier
            session_id: Session identifier
            message: Message to send
            visitor_name: Name for display

        Returns:
            Robot dog's response
        """
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )

        print(f"\n👤 {visitor_name}: {message}\n")
        print("🐕 Robot Dog: ", end="", flush=True)

        response_text = ""
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message
        ):
            if event.is_final_response():
                response_text = event.content.parts[0].text
                print(response_text)
                print()

        return response_text

    async def interactive_mode(self):
        """Run the robot dog in interactive CLI mode."""
        print("\n🐕 Welcome to Inteli Robot Dog Tour Guide!")
        print("Type 'exit' to quit, 'help' for commands\n")

        visitor_name = input("What's your name? ").strip() or "Visitante"

        user_id, session_id = await self.start_tour(visitor_name)

        while True:
            try:
                user_input = input(f"\n👤 {visitor_name}: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'exit':
                    print("\n🐕 Tchau, tchau! [latido] Foi ótimo conhecer você!\n")
                    break

                if user_input.lower() == 'help':
                    self._print_help()
                    continue

                await self.send_message(user_id, session_id, user_input, visitor_name)

            except KeyboardInterrupt:
                print("\n\n🐕 Até logo! [latido]\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def _print_help(self):
        """Print help information."""
        print("\n📋 COMMANDS:")
        print("  exit     - End the tour")
        print("  help     - Show this help message")
        print("\n💬 TRY ASKING:")
        print("  - Como funciona o processo seletivo?")
        print("  - Quais são os cursos?")
        print("  - Me fale sobre as bolsas")
        print("  - Vamos para a próxima parte do tour")
        print("  - Quantos clubes tem?")
        print()


async def demo_conversation():
    """
    Run a demo conversation showing the robot dog's capabilities.
    """
    app = InteliRobotDogApp()

    print("\n" + "="*60)
    print("🎬 DEMO: Inteli Robot Dog Tour Guide")
    print("="*60 + "\n")

    # Start tour
    user_id, session_id = await app.start_tour("João")

    # Demo conversation flow
    demo_messages = [
        "Meu nome é João e vim conhecer o Inteli!",
        "Como funciona o processo seletivo?",
        "E as bolsas? Tem auxílio financeiro?",
        "Vamos para a próxima parte do tour?",
        "Quais são os cursos disponíveis?",
        "Muito legal! Tem clubes estudantis?",
        "Obrigado pelo tour!"
    ]

    for message in demo_messages:
        await app.send_message(user_id, session_id, message, "João")
        # Small delay for readability
        import asyncio
        await asyncio.sleep(1)

    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("="*60 + "\n")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Inteli Robot Dog Tour Guide")
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Run mode: interactive or demo"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash-exp",
        help="LLM model to use"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        await demo_conversation()
    else:
        app = InteliRobotDogApp(model=args.model)
        await app.interactive_mode()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
