import os
import sys
import time

from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()
load_dotenv("agent_flow/.env", override=False)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_flow.agents.orchestrator_agent import OrchestratorAgent
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def print_header():
    print("\n" + "=" * 70)
    print("INTELI MULTI-AGENT CHAT SYSTEM")
    print("=" * 70)
    print("\nCommands:")
    print("  stats  - Show conversation statistics")
    print("  clear  - Clear conversation history")
    print("  exit   - Exit chat")
    print("-" * 70 + "\n")


def print_conversation_stats(orchestrator):
    """Show conversation statistics."""
    try:
        history = orchestrator.get_conversation_history()
        if not history:
            print("No conversation history yet.")
            return

        print("\nConversation Statistics:")
        print(f"  Total messages: {len(history)}")

        # Count by role
        user_msgs = sum(1 for msg in history if msg["role"] == "user")
        assistant_msgs = sum(1 for msg in history if msg["role"] == "assistant")
        print(f"  User: {user_msgs} | Assistant: {assistant_msgs}")

        # Count by agent used
        agents_used = {}
        for msg in history:
            if msg.get("agent_used"):
                agent = msg["agent_used"]
                agents_used[agent] = agents_used.get(agent, 0) + 1

        if agents_used:
            print("  Agents used:")
            for agent, count in agents_used.items():
                print(f"    - {agent}: {count}x")

        print("-" * 70)

    except Exception as e:
        print(f"Statistics error: {e}")


def print_bot_response(response: str, execution_time: float):
    """Print final response."""
    print(f"\nAssistant: {response}")
    print(f"Time: {execution_time:.2f}s")
    print("=" * 70)


def main():
    print_header()

    try:
        orchestrator = OrchestratorAgent()
        print("System initialized.\n")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()

            # Handle exit command
            if user_input.lower() in ["sair", "exit", "quit"]:
                break

            # Handle empty input
            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() == "stats":
                print_conversation_stats(orchestrator)
                continue

            if user_input.lower() in ["limpar", "clear", "reset"]:
                orchestrator.clear_history()
                print("History cleared.\n")
                continue

            # Process normal message
            start_time = time.time()
            response = orchestrator.process_message(user_input)
            end_time = time.time()

            print_bot_response(response, end_time - start_time)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
