"""
Simplified Inteli Robot Dog Tour Guide Application

This version uses a simpler approach without complex session management.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types
from agent_flow.agents.coordinator_agent import create_enhanced_coordinator

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Validate API key
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    print("Please create a .env file with your API key:")
    print("GOOGLE_API_KEY=your_api_key_here")
    exit(1)


async def chat_with_robot_dog(model: str = "gemini-2.0-flash-exp"):
    """
    Simple chat interface with the robot dog.
    
    Args:
        model: LLM model to use
    """
    print("🐕 Initializing Inteli Robot Dog Tour Guide...")
    agent = create_enhanced_coordinator(model)
    print("✅ Agent created successfully!\n")
    
    print("="*60)
    print("🐕 INTELI ROBOT DOG TOUR GUIDE")
    print("="*60)
    print("Type 'exit' to quit\n")
    
    # Conversation history
    history = []
    
    while True:
        # Get user input
        user_input = input("👤 Você: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit', 'sair']:
            print("\n🐕 Tchau! [latido] Foi ótimo conversar com você! *balança o rabo*\n")
            break
        
        # Add user message to history
        history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        print("\n🐕 Robot Dog: ", end="", flush=True)
        
        try:
            # Call the agent (simplified - without ADK runner)
            # For now, we'll use the agent's model directly
            from google import genai
            
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
            
            # Build conversation context
            messages = []
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })
            
            # Get agent's instruction as system prompt
            system_instruction = agent.instruction
            
            # Get tools from agent
            tools = []
            if hasattr(agent, 'tools') and agent.tools:
                for tool in agent.tools:
                    tools.append(tool)
            
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools,
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode=types.FunctionCallingConfig.Mode.AUTO
                        )
                    ),
                    temperature=0.7,
                )
            )
            
            assistant_response = response.text
            print(assistant_response + "\n")
            
            # Add assistant response to history
            history.append({
                "role": "model",
                "content": assistant_response
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            # Remove the failed user message from history
            history.pop()


async def demo_conversation_simple(model: str = "gemini-2.0-flash-exp"):
    """Run a simple demo conversation."""
    print("\n" + "="*60)
    print("🎬 DEMO: Inteli Robot Dog Tour Guide")
    print("="*60 + "\n")
    
    # Demo messages
    demo_messages = [
        "Olá! Estou aqui para o tour do Inteli!",
        "Como funciona o processo seletivo?",
        "E as bolsas? Tem auxílio financeiro?",
        "Quais são os cursos disponíveis?",
        "Obrigado pelo tour!"
    ]
    
    print("🐕 Initializing Inteli Robot Dog Tour Guide...")
    agent = create_enhanced_coordinator(model)
    print("✅ Agent created!\n")
    
    from google import genai
    import os
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    history = []
    
    for message in demo_messages:
        print(f"👤 João: {message}\n")
        print("🐕 Robot Dog: ", end="", flush=True)
        
        # Add to history
        history.append({
            "role": "user",
            "parts": [{"text": message}]
        })
        
        try:
            response = client.models.generate_content(
                model=model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=agent.instruction,
                    temperature=0.7,
                )
            )
            
            assistant_response = response.text
            print(assistant_response + "\n")
            
            history.append({
                "role": "model",
                "parts": [{"text": assistant_response}]
            })
            
            # Small delay for readability
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("="*60 + "\n")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inteli Robot Dog Tour Guide (Simplified)")
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
        await demo_conversation_simple(args.model)
    else:
        await chat_with_robot_dog(args.model)


if __name__ == "__main__":
    import os
    asyncio.run(main())
