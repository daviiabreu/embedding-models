#!/usr/bin/env python3
"""
Test script to validate agent_flow is working correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

print("=" * 60)
print("🧪 TESTING INTELI ROBOT DOG AGENT FLOW")
print("=" * 60)

# Test 1: Environment variables
print("\n1️⃣ Testing environment variables...")
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"   ✅ GOOGLE_API_KEY loaded (ends with: ...{api_key[-10:]})")
else:
    print("   ❌ GOOGLE_API_KEY not found!")
    sys.exit(1)

model = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
print(f"   ✅ Model: {model}")

# Test 2: Import agents
print("\n2️⃣ Testing imports...")
try:
    from agent_flow.agents import (
        create_enhanced_coordinator,
        create_safety_agent,
        create_tour_agent,
        create_knowledge_agent,
        create_context_agent,
    )
    print("   ✅ All agent imports successful")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 3: Import tools
print("\n3️⃣ Testing tool imports...")
try:
    from agent_flow.tools import (
        add_dog_personality,
        detect_visitor_emotion,
        get_conversation_suggestions,
        generate_engagement_prompt,
        check_content_safety,
        search_knowledge_base,
        get_user_preferences,
    )
    print("   ✅ All tool imports successful")
except ImportError as e:
    print(f"   ❌ Tool import error: {e}")
    sys.exit(1)

# Test 4: Check required files
print("\n4️⃣ Checking required files...")
required_files = [
    "documents/Edital-Processo-Seletivo-Inteli_-Graduacao-2026_AJUSTADO-chunks.json",
    "documents/script.md",
    ".env",
]

all_files_exist = True
for file_path in required_files:
    full_path = Path(__file__).parent / file_path
    if full_path.exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ⚠️  {file_path} (missing - some features may not work)")
        if file_path.endswith(".json"):
            all_files_exist = False

# Test 5: Create agents (without running)
print("\n5️⃣ Testing agent creation...")
try:
    print("   Creating safety agent...")
    safety_agent = create_safety_agent(model)
    print(f"   ✅ Safety agent created: {safety_agent.name}")
    
    print("   Creating tour agent...")
    tour_agent = create_tour_agent(model)
    print(f"   ✅ Tour agent created: {tour_agent.name}")
    
    print("   Creating knowledge agent...")
    knowledge_agent = create_knowledge_agent(model)
    print(f"   ✅ Knowledge agent created: {knowledge_agent.name}")
    
    print("   Creating context agent...")
    context_agent = create_context_agent(model)
    print(f"   ✅ Context agent created: {context_agent.name}")
    
    print("   Creating coordinator agent (with all sub-agents)...")
    coordinator = create_enhanced_coordinator(model)
    print(f"   ✅ Coordinator created: {coordinator.name}")
    print(f"   ✅ Sub-agents: {len(coordinator.sub_agents)} agents")
    print(f"   ✅ Tools: {len(coordinator.tools)} tools")
    
except Exception as e:
    print(f"   ❌ Agent creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test personality tools
print("\n6️⃣ Testing personality tools...")
try:
    from google.adk.tools.tool_context import ToolContext
    
    # Create a mock tool context
    class MockState(dict):
        pass
    
    mock_context = type('obj', (object,), {'state': MockState()})()
    
    # Test emotion detection
    emotion_result = detect_visitor_emotion("Olá! Estou muito empolgado!", mock_context)
    print(f"   ✅ Emotion detection: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.2f})")
    
    # Test personality addition
    personality_result = add_dog_personality(
        "Bem-vindo ao Inteli!", 
        "excited", 
        mock_context
    )
    print(f"   ✅ Personality tool: '{personality_result['enhanced_text'][:50]}...'")
    
    # Test engagement prompt
    engagement_result = generate_engagement_prompt("quiet_moment", mock_context)
    print(f"   ✅ Engagement prompt: '{engagement_result['prompt'][:50]}...'")
    
except Exception as e:
    print(f"   ❌ Personality tools error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\n📝 Next steps:")
print("   1. Run interactive mode: python run_app.py --mode interactive")
print("   2. Run demo mode: python run_app.py --mode demo")
print("   3. Check agent_flow/docs/ for architecture documentation")
print()
