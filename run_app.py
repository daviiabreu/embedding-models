#!/usr/bin/env python3
"""
Inteli Robot Dog Tour Guide - Main Entry Point

This script loads environment variables from .env and runs the robot dog tour guide.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env in project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: GOOGLE_API_KEY not found in .env file!")
    print("Please ensure .env file exists in project root with GOOGLE_API_KEY=your_key_here")
    sys.exit(1)

# Import the app after loading env vars
from agent_flow.app import main

if __name__ == "__main__":
    import asyncio
    
    print("🔧 Environment loaded successfully")
    print(f"📍 Using model: {os.getenv('DEFAULT_MODEL', 'gemini-2.0-flash-exp')}")
    print("-" * 60)
    
    asyncio.run(main())
