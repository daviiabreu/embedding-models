"""Tools for personality, safety, and document retrieval."""

# Document tools
from .document_tools import search_knowledge_base, get_user_preferences

# Graph RAG tool
from .rag_tool import run_graph_rag_tool

# Personality tools
from .personality_tools import (
    add_dog_personality,
    detect_visitor_emotion,
    get_conversation_suggestions,
    generate_engagement_prompt,
)

# Safety tools
from .safety_tools import check_content_safety

__all__ = [
    # Document tools
    "search_knowledge_base",
    "get_user_preferences",
    # RAG
    "run_graph_rag_tool",
    # Personality tools
    "add_dog_personality",
    "detect_visitor_emotion",
    "get_conversation_suggestions",
    "generate_engagement_prompt",
    # Safety tools
    "check_content_safety",
]
