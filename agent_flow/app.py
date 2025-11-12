#!/usr/bin/env python3
"""
Inteli Robot Dog Tour Guide Application

This is the main application that runs the robot dog tour guide.
Supports multiple execution modes: full (ADK), simple (direct API), and demo.

Usage:
    python -m agent_flow.app                    # Interactive mode (full)
    python -m agent_flow.app --mode demo        # Run demo
    python -m agent_flow.app --mode simple      # Simplified mode
    python -m agent_flow.app --debug            # Enable debug logging
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Tuple

# Google ADK Imports - AGORA COM A SOLUÇÃO CORRETA!
from google.adk.runners import InMemoryRunner
from google.genai import types
from google import genai

from agent_flow.agents.coordinator_agent import create_enhanced_coordinator


# Load environment variables from .env in project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def ensure_model_prefix(model: str) -> str:
    """
    Ensure model name has the 'models/' prefix required by the API.
    
    Args:
        model: Model name (with or without prefix)
        
    Returns:
        Model name with 'models/' prefix
    """
    if not model.startswith("models/"):
        return f"models/{model}"
    return model


class InteliRobotDogApp:
    """Main application class for the Inteli Robot Dog tour guide."""

    def __init__(self, model: str = "gemini-2.0-flash-exp", mode: str = "full", debug: bool = False):
        """
        Initialize the robot dog application.

        Args:
            model: LLM model to use (default: gemini-2.0-flash-exp)
            mode: Execution mode (full, simple, demo)
            debug: Enable debug logging
        """
        # Ensure model name has correct prefix
        self.model = ensure_model_prefix(model)
        self.mode = mode
        self.debug = debug

        # Create the main coordinator agent
        print("🐕 Initializing Inteli Robot Dog Tour Guide...")
        
        # Para ADK, usar modelo SEM o prefixo "models/"
        # Para API direta, usar modelo COM o prefixo "models/"
        if mode == "full":
            # Remover prefixo "models/" para ADK
            adk_model = model.replace("models/", "") if model.startswith("models/") else model
            if self.debug:
                print(f"[DEBUG] ADK model name: {adk_model}")
            self.agent = create_enhanced_coordinator(adk_model)
        else:
            self.agent = create_enhanced_coordinator(self.model)
        
        print("✅ Agent created successfully!")

        # Initialize runner based on mode
        if mode == "full":
            # FULL MODE: Use ADK InMemoryRunner with run_debug()
            # run_debug() automatically handles session creation!
            self.runner = InMemoryRunner(
                agent=self.agent,
                app_name="inteli_robot_dog"
            )
            print("✅ ADK Multi-agent system initialized!")
        else:
            # SIMPLE/DEMO MODE: Use direct API client
            self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
            print("✅ Direct API client initialized!")

    async def chat_simple(self, history: list) -> Tuple[str, list]:
        """
        Send a message using direct API (simple mode).
        
        Args:
            history: Conversation history
            
        Returns:
            Tuple of (response_text, updated_history)
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=self.agent.instruction,
                    temperature=0.7,
                )
            )
            
            assistant_response = response.text
            
            # Add personality elements if missing
            if '[latido]' not in assistant_response and 'latido' not in assistant_response.lower():
                assistant_response = f"[latido] {assistant_response}"
            
            # Add assistant response to history
            history.append({
                "role": "model",
                "parts": [{"text": assistant_response}]
            })
            
            return assistant_response, history
            
        except Exception as e:
            raise Exception(f"API call failed: {e}")

    async def interactive_mode_full(self):
        """Run the robot dog in interactive mode with full ADK multi-agent coordination."""
        print("\n🐕 Welcome to Inteli Robot Dog Tour Guide!")
        print("Type 'exit' to quit, 'help' for commands\n")

        visitor_name = input("What's your name? ").strip() or "Visitante"

        print(f"\n{'='*60}")
        print(f"🐕 INTELI ROBOT DOG TOUR GUIDE (Multi-Agent ADK)")
        print(f"{'='*60}\n")

        # Session ID único para este visitante
        session_id = f"tour_{visitor_name.lower().replace(' ', '_')}"
        user_id = visitor_name

        while True:
            try:
                user_input = input(f"\n👤 {visitor_name}: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'sair']:
                    print("\n🐕 Tchau, tchau! [latido] Foi ótimo conhecer você!\n")
                    break

                if user_input.lower() == 'help':
                    self._print_help()
                    continue

                print("🐕 Robot Dog: ", end="", flush=True)

                # SOLUÇÃO CORRETA: Usar run_debug() que cria sessões automaticamente!
                events = await self.runner.run_debug(
                    user_messages=user_input,
                    user_id=user_id,
                    session_id=session_id,
                    quiet=True  # Não exibir mensagens internas do ADK
                )

                # Extrair resposta final dos eventos
                response_text = ""
                has_safety_response = False
                
                for event in events:
                    if hasattr(event, 'content') and event.content:
                        # Verificar se content.parts é iterável
                        parts = getattr(event.content, 'parts', None)
                        if parts:
                            try:
                                for part in parts:
                                    if hasattr(part, 'text') and part.text:
                                        text_content = part.text.strip()
                                        response_text += text_content
                                        # Detectar respostas de safety
                                        if text_content in ["SAFE", "UNSAFE"]:
                                            has_safety_response = True
                            except TypeError:
                                # Se parts não for iterável, ignorar
                                pass

                # Se não houver texto, buscar no último evento
                if not response_text and events:
                    last_event = events[-1] if isinstance(events, list) else events
                    if hasattr(last_event, 'text'):
                        response_text = last_event.text

                # Verificar se a resposta é apenas de safety
                if response_text in ["SAFE", "UNSAFE"]:
                    if response_text == "UNSAFE":
                        response_text = "*balança o rabo* [latido] Opa! Que tal falarmos sobre algo mais legal? Posso te contar sobre os projetos incríveis do Inteli! 🐕"
                    else:
                        response_text = "*balança o rabo* [latido] Como posso te ajudar com o tour do Inteli? 😊"
                elif not response_text:
                    response_text = "*balança o rabo* [latido] Hmm, não entendi bem. Pode reformular? 🐕"

                print(response_text + "\n")

                if self.debug:
                    print(f"[DEBUG] Session: {session_id}, User: {user_id}")

            except KeyboardInterrupt:
                print("\n\n🐕 Até logo! [latido]\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    async def interactive_mode_simple(self):
        """Run the robot dog in interactive mode using direct API (simple mode)."""
        print("\n" + "=" * 60)
        print("🐕 INTELI ROBOT DOG TOUR GUIDE")
        print("=" * 60)
        print("Type 'exit' to quit\n")
        
        history = []
        
        while True:
            user_input = input("👤 Você: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'sair']:
                print("\n🐕 Tchau! [latido] Foi ótimo conversar com você! *balança o rabo*\n")
                break
            
            # Add user message to history
            history.append({
                "role": "user",
                "parts": [{"text": user_input}]
            })
            
            print("\n🐕 Robot Dog: ", end="", flush=True)
            
            try:
                # Apply basic safety filtering
                unsafe_words = ['hack', 'droga', 'merda', 'bosta']
                if any(word in user_input.lower() for word in unsafe_words):
                    response_text = "*balança o rabo* [latido] Que tal falarmos sobre algo mais interessante? Posso te contar sobre os incríveis projetos do Inteli! 😊"
                    print(response_text + "\n")
                    history.append({
                        "role": "model",
                        "parts": [{"text": response_text}]
                    })
                    continue
                
                response_text, history = await self.chat_simple(history)
                print(response_text + "\n")
                
                if self.debug:
                    print(f"\n[DEBUG] History length: {len(history)} messages")
                
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                history.pop()  # Remove failed user message

    async def demo_conversation(self):
        """Run a demo conversation showing the robot dog's capabilities."""
        print("\n" + "=" * 60)
        print("🎬 DEMO: Inteli Robot Dog Tour Guide")
        print("=" * 60 + "\n")

        demo_messages = [
            "Olá! Estou aqui para o tour do Inteli!",
            "Como funciona o processo seletivo?",
            "E as bolsas? Tem auxílio financeiro?",
            "Quais são os cursos disponíveis?",
            "Obrigado pelo tour!"
        ]

        if self.mode == "full":
            # Demo with ADK runner using run_debug()
            user_id = "João"
            session_id = "demo_tour_joao"
            
            for i, message in enumerate(demo_messages, 1):
                print(f"👤 João: {message}\n")
                print("🐕 Robot Dog: ", end="", flush=True)
                
                try:
                    # Usar run_debug() que gerencia sessões automaticamente
                    events = await self.runner.run_debug(
                        user_messages=message,
                        user_id=user_id,
                        session_id=session_id,
                        quiet=True
                    )
                    
                    # Extrair resposta
                    response_text = ""
                    for event in events:
                        if hasattr(event, 'content') and event.content:
                            # Verificar se content.parts é iterável
                            parts = getattr(event.content, 'parts', None)
                            if parts:
                                try:
                                    for part in parts:
                                        if hasattr(part, 'text') and part.text:
                                            response_text += part.text.strip()
                                except TypeError:
                                    # Se parts não for iterável, ignorar
                                    pass
                    
                    # Se não houver texto, buscar no último evento
                    if not response_text and events:
                        last_event = events[-1] if isinstance(events, list) else events
                        if hasattr(last_event, 'text'):
                            response_text = last_event.text
                    
                    # Verificar se a resposta é apenas de safety
                    if response_text in ["SAFE", "UNSAFE"]:
                        if response_text == "UNSAFE":
                            response_text = "*balança o rabo* [latido] Opa! Que tal falarmos sobre algo mais legal? 🐕"
                        else:
                            response_text = "*balança o rabo* [latido] Como posso te ajudar? 😊"
                    elif not response_text:
                        response_text = "*balança o rabo* [latido]\n"
                    
                    print(response_text + "\n")
                    
                    if self.debug:
                        print(f"[DEBUG] Message {i}/{len(demo_messages)} completed\n")
                    
                    await asyncio.sleep(1.5)
                    
                except Exception as e:
                    print(f"Error: {e}\n")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
        else:
            # Demo with direct API
            history = []
            
            for i, message in enumerate(demo_messages, 1):
                print(f"👤 João: {message}\n")
                print("🐕 Robot Dog: ", end="", flush=True)
                
                history.append({
                    "role": "user",
                    "parts": [{"text": message}]
                })
                
                try:
                    response_text, history = await self.chat_simple(history)
                    print(response_text + "\n")
                    
                    if self.debug:
                        print(f"[DEBUG] Message {i}/{len(demo_messages)} completed\n")
                    
                    await asyncio.sleep(1.5)
                    
                except Exception as e:
                    print(f"Error: {e}\n")
                    if self.debug:
                        import traceback
                        traceback.print_exc()

        print("\n" + "=" * 60)
        print("✅ Demo completed!")
        print("=" * 60 + "\n")

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


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found in .env file!")
        print("Please ensure .env file exists in project root with:")
        print("GOOGLE_API_KEY=your_key_here")
        return False
    return True


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Inteli Robot Dog Tour Guide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agent_flow.app                    # Interactive mode (full)
  python -m agent_flow.app --mode demo        # Run demo conversation
  python -m agent_flow.app --mode simple      # Simplified mode for debugging
  python -m agent_flow.app --model gemini-pro # Use different model
  python -m agent_flow.app --debug            # Enable debug logging

Modes:
  full   - Multi-agent coordinator with full personality (uses direct API for reliability)
  simple - Streamlined direct API mode (faster, lighter)
  demo   - Automated demo conversation (fully functional)
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "simple", "demo"],
        default="full",
        help="Execution mode (default: full)"
    )
    
    parser.add_argument(
        "--model",
        default=os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-lite"),
        help="LLM model to use (default: gemini-2.5-flash-lite)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Inteli Robot Dog Tour Guide v1.0.0"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("🐕 INTELI ROBOT DOG TOUR GUIDE")
    print("=" * 60)
    print(f"🔧 Mode: {args.mode.upper()}")
    print(f"📍 Model: {args.model}")
    if args.debug:
        print("🐛 Debug: ENABLED")
    print("-" * 60 + "\n")
    
    # Create and run application
    try:
        app = InteliRobotDogApp(
            model=args.model,
            mode=args.mode,
            debug=args.debug
        )
        
        if args.mode == "demo":
            await app.demo_conversation()
        elif args.mode == "full":
            await app.interactive_mode_full()
        elif args.mode == "simple":
            await app.interactive_mode_simple()
            
    except KeyboardInterrupt:
        print("\n\n🐕 [latido] Até logo! *balança o rabo e corre*\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
