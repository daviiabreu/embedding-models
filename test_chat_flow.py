#!/usr/bin/env python3
"""
Comprehensive diagnostic test for the chat flow.
Tests all components and stages of the orchestrator.
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()
load_dotenv("agent_flow/.env", override=False)

print("=" * 70)
print("CHAT FLOW DIAGNOSTIC TEST")
print("=" * 70)

# Test 1: Environment
print("\n[Test 1] Environment Variables")
api_key = os.getenv("GOOGLE_API_KEY")
model = os.getenv("DEFAULT_MODEL")
print(f"  API Key: {'✓ Set' if api_key else '✗ Missing'}")
print(f"  Model: {model or '✗ Not set'}")

# Test 2: Imports
print("\n[Test 2] Module Imports")
try:
    import google.generativeai as genai

    print("  google.generativeai: ✓")
except ImportError as e:
    print(f"  google.generativeai: ✗ {e}")
    sys.exit(1)

try:
    from google.adk.agents import Agent

    print("  google.adk: ✓")
except ImportError as e:
    print(f"  google.adk: ✗ {e}")
    sys.exit(1)

try:
    from agent_flow.tools.knowledge_tools import rag_inference_pipeline

    print("  RAG pipeline: ✓")
except ImportError as e:
    print(f"  RAG pipeline: ✗ {e}")

try:
    from agent_flow.agents.orchestrator_agent import OrchestratorAgent

    print("  OrchestratorAgent: ✓")
except ImportError as e:
    print(f"  OrchestratorAgent: ✗ {e}")
    sys.exit(1)

# Test 3: Model Availability
print("\n[Test 3] Gemini Model Availability")
try:
    genai.configure(api_key=api_key)
    test_model = genai.GenerativeModel(model)
    response = test_model.generate_content("Say 'OK' if you work.")
    print(f"  Model '{model}': ✓")
    print(f"  Test response: {response.text[:50]}...")
except Exception as e:
    print(f"  Model '{model}': ✗ {e}")
    sys.exit(1)

# Test 4: Orchestrator Initialization
print("\n[Test 4] Orchestrator Initialization")
try:
    orch = OrchestratorAgent()
    print("  Orchestrator created: ✓")
    print(f"  Safety agent: {'✓' if orch.safety_agent else '✗'}")
    print(f"  Context agent: {'✓' if orch.context_agent else '✗'}")
    print(f"  Personality agent: {'✓' if orch.personality_agent else '✗'}")
    print(f"  Knowledge agent: {'✓' if orch.knowledge_agent else '✗'}")
    print(f"  Tour agent: {'✓' if orch.tour_agent else '✗'}")
    print(f"  LLM client: {'✓' if orch.llm else '✗'}")
except Exception as e:
    print(f"  Orchestrator: ✗ {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Safety Validation
print("\n[Test 5] Safety Validation")
try:
    safe_result = orch._validate_input_safety("Hello, how are you?")
    print(f"  Safe message: {safe_result}")
    print(f"  Result: {'✓' if safe_result.get('safe') else '✗'}")
except Exception as e:
    print(f"  Safety check: ✗ {e}")

# Test 6: Context Management
print("\n[Test 6] Context Management")
try:
    context = orch._manage_context_memory("Test message")
    print("  Context retrieved: ✓")
    print(f"  Context length: {len(context)} chars")
except Exception as e:
    print(f"  Context management: ✗ {e}")

# Test 7: Personality Detection
print("\n[Test 7] Personality Detection")
try:
    personality = orch._detect_personality_and_adapt("Who is Ana Garcia?", "")
    print("  Personality detected: ✓")
    print(f"  Tone: {personality.get('tone')}")
    print(f"  Style: {personality.get('style')}")
except Exception as e:
    print(f"  Personality detection: ✗ {e}")

# Test 8: Intent Classification
print("\n[Test 8] Intent Classification")
try:
    intent = orch._decide_intent("Who is Ana Garcia?")
    print("  Intent classified: ✓")
    print(f"  Intent: {intent}")
except Exception as e:
    print(f"  Intent classification: ✗ {e}")

# Test 9: RAG Pipeline (if available)
print("\n[Test 9] RAG Pipeline")
try:
    from agent_flow.tools.knowledge_tools import rag_inference_pipeline

    result = rag_inference_pipeline("test query")
    print("  RAG pipeline: ✓")
    print(f"  Result keys: {list(result.keys())}")
except Exception as e:
    print(f"  RAG pipeline: ⚠ {e} (may need Qdrant)")

# Test 10: Full Message Processing
print("\n[Test 10] Full Message Processing")
test_messages = [
    ("Oi!", "chitchat"),
    ("Quem é a Ana Garcia?", "knowledge"),
]

for msg, expected_type in test_messages:
    try:
        print(f"\n  Testing: '{msg}'")
        response = orch.process_message(msg)
        print(f"    Response: {response[:100]}...")
        print("    Status: ✓")
    except Exception as e:
        print(f"    Status: ✗ {e}")
        import traceback

        traceback.print_exc()

# Test 11: Conversation History
print("\n[Test 11] Conversation History")
history = orch.get_conversation_history()
print(f"  Messages in history: {len(history)}")
print("  History tracking: ✓")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 70)
print("\nIf all tests passed, the system is ready to use!")
print("Run: python chat_with_agents.py")
print("=" * 70)
