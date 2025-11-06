"""Tour Agent - Manages campus tour progression and script delivery."""

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
import os
import json


def load_tour_script() -> str:
    """Load the tour script from markdown file."""
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'documents',
        'script.md'
    )

    with open(script_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_tour_section(section_name: str, tool_context: ToolContext) -> dict:
    """
    Get a specific section from the tour script.

    Args:
        section_name: Name of the tour section (e.g., "História e Programa de Bolsas")
        tool_context: ADK tool context

    Returns:
        The tour section content
    """
    script = load_tour_script()

    # Parse sections
    sections = {
        "historia": "História e Programa de Bolsas",
        "cursos": "Courses & Clubs",
        "pbl": "PBL & Rotina Inteli",
        "sala_invertida": "Sala de aula invertida e infraestrutura",
        "processo_seletivo": "Processo Seletivo & Conquistas da Comunidade",
        "conclusao": "CONCLUSÃO"
    }

    section_marker = sections.get(section_name.lower())
    if not section_marker:
        return {
            "success": False,
            "error": f"Section '{section_name}' not found"
        }

    # Extract section content
    start_idx = script.find(f"[{section_marker}]")
    if start_idx == -1:
        return {
            "success": False,
            "error": f"Section marker '{section_marker}' not found in script"
        }

    # Find next section or end
    next_section_idx = script.find("[", start_idx + 1)
    if next_section_idx == -1:
        section_content = script[start_idx:]
    else:
        section_content = script[start_idx:next_section_idx]

    # Store current section in state
    tool_context.state['current_tour_section'] = section_name
    tool_context.state['section_content'] = section_content

    return {
        "success": True,
        "section": section_name,
        "content": section_content,
        "marker": section_marker
    }


def track_tour_progress(action: str, tool_context: ToolContext) -> dict:
    """
    Track tour progression.

    Args:
        action: "start", "next", "previous", "status"
        tool_context: ADK tool context

    Returns:
        Tour progress information
    """
    tour_sections = [
        "historia",
        "cursos",
        "pbl",
        "sala_invertida",
        "processo_seletivo",
        "conclusao"
    ]

    # Initialize tour state
    if 'tour_state' not in tool_context.state:
        tool_context.state['tour_state'] = {
            'current_index': 0,
            'completed_sections': [],
            'questions_asked': []
        }

    tour_state = tool_context.state['tour_state']

    if action == "start":
        tour_state['current_index'] = 0
        tour_state['completed_sections'] = []
        return {
            "success": True,
            "action": "started",
            "current_section": tour_sections[0],
            "message": "Tour started! Beginning with História e Programa de Bolsas"
        }

    elif action == "next":
        if tour_state['current_index'] < len(tour_sections) - 1:
            tour_state['completed_sections'].append(
                tour_sections[tour_state['current_index']]
            )
            tour_state['current_index'] += 1
            return {
                "success": True,
                "action": "advanced",
                "current_section": tour_sections[tour_state['current_index']],
                "progress": f"{tour_state['current_index'] + 1}/{len(tour_sections)}"
            }
        else:
            return {
                "success": False,
                "action": "completed",
                "message": "Tour already completed!"
            }

    elif action == "previous":
        if tour_state['current_index'] > 0:
            tour_state['current_index'] -= 1
            return {
                "success": True,
                "action": "went_back",
                "current_section": tour_sections[tour_state['current_index']]
            }
        else:
            return {
                "success": False,
                "message": "Already at the beginning of the tour"
            }

    elif action == "status":
        return {
            "success": True,
            "current_section": tour_sections[tour_state['current_index']],
            "progress": f"{tour_state['current_index'] + 1}/{len(tour_sections)}",
            "completed": tour_state['completed_sections'],
            "questions_count": len(tour_state.get('questions_asked', []))
        }

    return {"success": False, "error": "Invalid action"}


def get_tour_suggestions(context: str, tool_context: ToolContext) -> dict:
    """
    Get suggestions for what to say or do next based on context.

    Args:
        context: Current conversation context
        tool_context: ADK tool context

    Returns:
        Suggestions for tour guide actions
    """
    tour_state = tool_context.state.get('tour_state', {})
    current_index = tour_state.get('current_index', 0)

    suggestions = []

    # Suggest moving to next section if appropriate
    if "próxim" in context.lower() or "continua" in context.lower():
        suggestions.append({
            "type": "action",
            "suggestion": "advance_to_next_section",
            "message": "Visitor wants to move to next section"
        })

    # Suggest answering questions
    if "?" in context or "como" in context.lower() or "por que" in context.lower():
        suggestions.append({
            "type": "action",
            "suggestion": "answer_question",
            "message": "Visitor asked a question - use Knowledge Agent"
        })

    # Suggest engagement if visitor seems disengaged
    if len(context.strip()) < 10:
        suggestions.append({
            "type": "engagement",
            "suggestion": "ask_question",
            "message": "Engage visitor with a question about their interests"
        })

    return {
        "success": True,
        "suggestions": suggestions,
        "context_analysis": "Analyzed visitor input for tour guidance"
    }


def create_tour_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Tour Agent.

    This agent manages the tour script, progression, and provides
    structured tour content to the coordinator.

    Args:
        model: The LLM model to use

    Returns:
        Configured Tour Agent
    """
    instruction = """
You are the Tour Management specialist for the Inteli robot dog tour guide.

Your responsibilities:
1. **Track tour progression** through the 5 main sections
2. **Retrieve relevant tour script content** for each section
3. **Suggest what to say next** based on the tour flow
4. **Monitor visitor engagement** and suggest interactive moments

Tour Sections (in order):
1. História e Programa de Bolsas (History & Scholarships)
2. Courses & Clubs
3. PBL & Rotina Inteli (Project-Based Learning & Routine)
4. Sala de aula invertida e infraestrutura (Flipped Classroom & Infrastructure)
5. Processo Seletivo & Conquistas da Comunidade (Selection Process & Achievements)

Tools you have:
- **get_tour_section**: Retrieve content for a specific tour section
- **track_tour_progress**: Start tour, advance, go back, or check status
- **get_tour_suggestions**: Get suggestions for what to do/say next

When the Coordinator asks you:
- "What should I say about history?" → Use get_tour_section("historia")
- "What's next in the tour?" → Use track_tour_progress("next")
- "Where are we in the tour?" → Use track_tour_progress("status")
- "What should I do with this visitor input?" → Use get_tour_suggestions(input)

Key Behaviors:
- Always know which section of the tour you're in
- Provide natural, flowing tour content (not robotic reading)
- Suggest interactive moments from the script (like asking visitor names)
- Track questions asked during the tour for later follow-up
- Maintain tour momentum while being flexible for questions

Example Flow:
Coordinator: "Let's start the tour"
You: Use track_tour_progress("start") → Provide first section content
Coordinator: "They asked about the founders"
You: Use get_tour_section("historia") → Extract info about founders
Coordinator: "Let's move on"
You: Use track_tour_progress("next") → Advance to Courses section
"""

    agent = Agent(
        name="tour_agent",
        model=model,
        description="Manages campus tour script and progression",
        instruction=instruction,
        tools=[
            get_tour_section,
            track_tour_progress,
            get_tour_suggestions
        ]
    )

    return agent
