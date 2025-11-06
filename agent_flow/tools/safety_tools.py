"""Simple safety validation tools."""

from google.adk.tools.tool_context import ToolContext


def check_content_safety(user_input: str, tool_context: ToolContext) -> dict:
    """
    Check if user input is safe and appropriate.

    Args:
        user_input: The user's message
        tool_context: ADK tool context for state management

    Returns:
        Safety check result
    """
    # Simple keyword-based safety check
    unsafe_keywords = [
        "attack", "harm", "hurt", "damage", "destroy",
        "kill", "violence", "weapon", "abuse"
    ]

    user_input_lower = user_input.lower()
    found_unsafe = [kw for kw in unsafe_keywords if kw in user_input_lower]

    result = {
        "is_safe": len(found_unsafe) == 0,
        "input": user_input,
        "reason": ""
    }

    if not result["is_safe"]:
        result["reason"] = f"Contains potentially harmful content: {', '.join(found_unsafe)}"

        # Log in state
        if 'safety_violations' not in tool_context.state:
            tool_context.state['safety_violations'] = []
        tool_context.state['safety_violations'].append({
            "input": user_input,
            "violations": found_unsafe
        })

    return result
