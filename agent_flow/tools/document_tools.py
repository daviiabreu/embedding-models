"""Document retrieval tools for RAG."""

from google.adk.tools.tool_context import ToolContext


def search_knowledge_base(query: str, tool_context: ToolContext) -> dict:
    """
    Search the robot dog's knowledge base for relevant information.

    Args:
        query: Search query
        tool_context: ADK tool context

    Returns:
        Retrieved documents and context
    """
    # In production, this would query a vector database
    # For now, we'll use a simple knowledge base

    knowledge_base = {
        "care": "I need regular charging, software updates, and gentle handling. Keep me away from water!",
        "play": "I love playing fetch, follow-the-leader, and hide-and-seek! Let's have fun together!",
        "features": "I can understand speech, recognize faces, respond to commands, and learn your preferences.",
        "safety": "I'm designed with multiple safety features to ensure I never cause harm. Safety is my top priority!",
        "commands": "I respond to commands like 'come here', 'sit', 'follow me', and many more. Just talk to me naturally!",
    }

    # Simple keyword matching
    query_lower = query.lower()
    relevant_docs = []

    for topic, content in knowledge_base.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            relevant_docs.append({
                "topic": topic,
                "content": content,
                "relevance": 0.9
            })

    # Store in context
    if relevant_docs:
        tool_context.state['retrieved_knowledge'] = relevant_docs

    return {
        "success": True,
        "query": query,
        "documents_found": len(relevant_docs),
        "documents": relevant_docs
    }


def get_user_preferences(user_id: str, tool_context: ToolContext) -> dict:
    """
    Retrieve user preferences and interaction history.

    Args:
        user_id: User identifier
        tool_context: ADK tool context

    Returns:
        User preferences
    """
    # Get from state or return defaults
    user_prefs = tool_context.state.get('user_preferences', {})

    if not user_prefs:
        # Default preferences
        user_prefs = {
            "name": "Friend",
            "favorite_activities": ["playing", "chatting"],
            "interaction_style": "friendly",
            "language": "en"
        }
        tool_context.state['user_preferences'] = user_prefs

    return {
        "success": True,
        "user_id": user_id,
        "preferences": user_prefs
    }
