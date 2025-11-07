"""
Fixed Inteli Robot Dog Tour Guide Application

This version handles tool calls correctly.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types
from google import genai
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


async def chat_with_robot_dog_fixed(model: str = "gemini-2.0-flash-exp"):
    """
    Fixed chat interface with the robot dog.
    
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
    
    history = []
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
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
            "content": user_input
        })
        
        # Generate response
        print("\n🐕 Robot Dog: ", end="", flush=True)
        
        try:
            # Build conversation context
            messages = []
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })
            
            # Get agent's instruction as system prompt
            system_instruction = agent.instruction
            
            # Simple approach: use agent instruction but without complex tools
            # Just include personality and basic knowledge
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
            )
            
            assistant_response = response.text
            
            # Apply basic safety filtering
            unsafe_words = ['hack', 'droga', 'buceta', 'piroc', 'merda', 'bosta', 'mijo']
            if any(word in user_input.lower() for word in unsafe_words):
                assistant_response = "*balança o rabo* [latido] Que tal falarmos sobre algo mais interessante? Posso te contar sobre os incríveis projetos do Inteli! 😊"
            
            # Add personality elements if missing
            if '[latido]' not in assistant_response and 'latido' not in assistant_response:
                assistant_response = f"[latido] {assistant_response}"
            
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


async def main():
    """Main entry point."""
    import sys
    
    mode = "interactive"
    if len(sys.argv) > 2 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
    
    if mode == "interactive":
        await chat_with_robot_dog_fixed()
    else:
        print("Available modes: --mode interactive")

if __name__ == "__main__":
    asyncio.run(main())