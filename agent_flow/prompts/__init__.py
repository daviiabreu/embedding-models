"""Helper functions to load and manage prompts."""

import os
from pathlib import Path
from typing import Dict


def load_prompt_file(filename: str) -> str:
    """
    Load a prompt file from the prompts directory.
    
    Args:
        filename: Name of the prompt file (e.g., "base_personality.txt")
        
    Returns:
        Content of the prompt file
    """
    prompts_dir = Path(__file__).parent
    file_path = prompts_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_base_personality() -> str:
    """Get the base personality prompt for the robot dog."""
    return load_prompt_file("base_personality.txt")


def get_safety_guidelines() -> str:
    """Get the safety guidelines for content validation."""
    return load_prompt_file("safety_guidelines.txt")


def get_all_prompts() -> Dict[str, str]:
    """
    Load all available prompts.
    
    Returns:
        Dictionary mapping prompt names to their content
    """
    return {
        "base_personality": get_base_personality(),
        "safety_guidelines": get_safety_guidelines(),
    }


def format_instruction_with_personality(base_instruction: str) -> str:
    """
    Enhance an agent instruction with the base personality.
    
    Args:
        base_instruction: The agent's specific instruction
        
    Returns:
        Combined instruction with personality guidelines
    """
    personality = get_base_personality()
    
    return f"""
{base_instruction}

═══════════════════════════════════════════════════════════════
PERSONALITY GUIDELINES
═══════════════════════════════════════════════════════════════

{personality}
"""


def format_instruction_with_safety(base_instruction: str) -> str:
    """
    Enhance an agent instruction with safety guidelines.
    
    Args:
        base_instruction: The agent's specific instruction
        
    Returns:
        Combined instruction with safety guidelines
    """
    safety = get_safety_guidelines()
    
    return f"""
{base_instruction}

═══════════════════════════════════════════════════════════════
SAFETY GUIDELINES
═══════════════════════════════════════════════════════════════

{safety}
"""


# Example usage:
if __name__ == "__main__":
    print("Testing prompt loading...")
    
    try:
        personality = get_base_personality()
        print(f"✅ Base personality loaded ({len(personality)} chars)")
        
        safety = get_safety_guidelines()
        print(f"✅ Safety guidelines loaded ({len(safety)} chars)")
        
        all_prompts = get_all_prompts()
        print(f"✅ All prompts loaded: {list(all_prompts.keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
