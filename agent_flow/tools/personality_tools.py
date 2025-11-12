"""Personality tools for maintaining consistent robot dog character."""

import random
from typing import Dict

from google.adk.tools.tool_context import ToolContext


def add_dog_personality(text: str, emotion: str, tool_context: ToolContext) -> dict:
    """
    Add robot dog personality elements to responses.

    This enhances text with dog-like expressions, barks, and emotional cues
    while maintaining professionalism and clarity.

    Args:
        text: The base text to enhance
        emotion: Current emotional tone (happy, excited, calm, curious, empathetic)
        tool_context: ADK tool context

    Returns:
        Enhanced text with personality
    """
    # Dog expressions based on emotion
    expressions = {
        "happy": {
            "barks": ["[latido alegre]", "[latido]", "Au au!"],
            "actions": ["*balança o rabo*", "*pula animado*", "*olhos brilhando*"],
            "intensifiers": ["muito", "super", "demais"],
        },
        "excited": {
            "barks": ["[latido empolgado]", "[latidos múltiplos]", "Au au au!"],
            "actions": [
                "*pula de alegria*",
                "*gira em círculos*",
                "*balança o rabo rapidamente*",
            ],
            "intensifiers": ["extremamente", "incrivelmente", "fantasticamente"],
        },
        "calm": {
            "barks": ["[latido suave]", "[latido gentil]"],
            "actions": [
                "*inclina a cabeça*",
                "*senta ao seu lado*",
                "*olha atentamente*",
            ],
            "intensifiers": ["tranquilamente", "calmamente"],
        },
        "curious": {
            "barks": ["[latido curioso]", "[latido questionador]"],
            "actions": ["*inclina a cabeça*", "*orelhas levantadas*", "*focado*"],
            "intensifiers": ["interessante", "curioso", "fascinante"],
        },
        "empathetic": {
            "barks": ["[latido compreensivo]", "[latido carinhoso]"],
            "actions": [
                "*se aproxima gentilmente*",
                "*coloca a pata no seu braço*",
                "*olha nos seus olhos*",
            ],
            "intensifiers": ["com carinho", "gentilmente", "compreensivamente"],
        },
    }

    emotion_data = expressions.get(emotion.lower(), expressions["happy"])

    # Randomly decide if we should add personality elements (80% chance)
    if random.random() < 0.8:
        # Add bark at beginning or end (50% chance each)
        if random.random() < 0.5:
            bark = random.choice(emotion_data["barks"])
            text = f"{bark} {text}"
        else:
            bark = random.choice(emotion_data["barks"])
            text = f"{text} {bark}"

    # Add action occasionally (40% chance)
    if random.random() < 0.4:
        action = random.choice(emotion_data["actions"])
        # Insert action in middle or end
        if random.random() < 0.5 and len(text) > 100:
            mid_point = len(text) // 2
            text = f"{text[:mid_point]} {action} {text[mid_point:]}"
        else:
            text = f"{text} {action}"

    # Store personality stats
    if "personality_stats" not in tool_context.state:
        tool_context.state["personality_stats"] = {
            "barks_count": 0,
            "actions_count": 0,
            "emotions_used": {},
        }

    stats = tool_context.state["personality_stats"]
    stats["barks_count"] = stats.get("barks_count", 0) + text.count("[latido")
    stats["actions_count"] = stats.get("actions_count", 0) + text.count("*")
    stats["emotions_used"][emotion] = stats["emotions_used"].get(emotion, 0) + 1

    return {
        "success": True,
        "original_text": text,
        "enhanced_text": text,
        "emotion_applied": emotion,
        "elements_added": {"barks": text.count("[latido"), "actions": text.count("*")},
    }


def detect_visitor_emotion(visitor_input: str, tool_context: ToolContext) -> dict:
    """
    Detect visitor's emotional state from their input.

    Args:
        visitor_input: What the visitor said
        tool_context: ADK tool context

    Returns:
        Detected emotion and confidence
    """
    text_lower = visitor_input.lower()

    # Emotion detection keywords (Portuguese)
    emotion_patterns = {
        "excited": {
            "keywords": [
                "incrível",
                "demais",
                "adorei",
                "amei",
                "uau",
                "wow",
                "legal",
                "massa",
                "!",
            ],
            "weight": 1.0,
        },
        "happy": {
            "keywords": [
                "feliz",
                "bom",
                "ótimo",
                "sim",
                "obrigado",
                "valeu",
                "show",
                "rs",
                "haha",
                "kkk",
            ],
            "weight": 0.9,
        },
        "curious": {
            "keywords": [
                "como",
                "por que",
                "quando",
                "onde",
                "qual",
                "?",
                "quero saber",
                "me conte",
            ],
            "weight": 0.8,
        },
        "confused": {
            "keywords": ["não entendi", "confuso", "dúvida", "explica", "como assim"],
            "weight": 0.7,
        },
        "anxious": {
            "keywords": [
                "preocupado",
                "ansioso",
                "medo",
                "difícil",
                "complicado",
                "nervoso",
            ],
            "weight": 0.6,
        },
        "bored": {"keywords": ["ok", "tá", "tanto faz", "sei lá"], "weight": 0.5},
        "neutral": {"keywords": [".", "próximo", "continua", "vamos"], "weight": 0.3},
    }

    # Score each emotion
    emotion_scores = {}
    for emotion, data in emotion_patterns.items():
        score = 0
        matches = []
        for keyword in data["keywords"]:
            if keyword in text_lower:
                score += 1
                matches.append(keyword)

        if score > 0:
            emotion_scores[emotion] = {
                "score": score * data["weight"],
                "matches": matches,
            }

    # Determine primary emotion
    if emotion_scores:
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1]["score"])
        emotion_name = primary_emotion[0]
        confidence = min(primary_emotion[1]["score"] / 3, 1.0)  # Normalize to 0-1
    else:
        emotion_name = "neutral"
        confidence = 0.5

    # Store in context
    if "visitor_emotions" not in tool_context.state:
        tool_context.state["visitor_emotions"] = []

    tool_context.state["visitor_emotions"].append(
        {"input": visitor_input, "emotion": emotion_name, "confidence": confidence}
    )

    # Keep only last 10 emotions
    if len(tool_context.state["visitor_emotions"]) > 10:
        tool_context.state["visitor_emotions"] = tool_context.state["visitor_emotions"][
            -10:
        ]

    return {
        "success": True,
        "emotion": emotion_name,
        "confidence": confidence,
        "all_scores": emotion_scores,
        "suggested_response_tone": _get_response_tone(emotion_name),
    }


def _get_response_tone(visitor_emotion: str) -> str:
    """Map visitor emotion to appropriate robot dog response tone."""
    tone_mapping = {
        "excited": "excited",  # Match their energy!
        "happy": "happy",
        "curious": "helpful",  # Be informative and encouraging
        "confused": "patient",  # Be clear and reassuring
        "anxious": "empathetic",  # Be calming and supportive
        "bored": "playful",  # Try to re-engage
        "neutral": "friendly",  # Standard friendly
    }
    return tone_mapping.get(visitor_emotion, "friendly")


def get_conversation_suggestions(context: Dict, tool_context: ToolContext) -> dict:
    """
    Get smart suggestions for how to respond based on conversation context.

    Args:
        context: Current conversation context (last messages, tour section, etc.)
        tool_context: ADK tool context

    Returns:
        Suggestions for response style, topics, and actions
    """
    suggestions = {
        "response_style": "friendly",
        "suggested_topics": [],
        "actions": [],
        "warnings": [],
    }

    # Check conversation length
    tour_state = tool_context.state.get("tour_state", {})
    messages_count = tool_context.state.get("message_count", 0)

    # If tour hasn't started, suggest starting
    if tour_state.get("current_index", -1) == -1:
        suggestions["actions"].append(
            {
                "action": "start_tour",
                "reason": "Tour hasn't started yet",
                "priority": "high",
            }
        )

    # Check for disengagement (very short responses)
    recent_emotions = tool_context.state.get("visitor_emotions", [])[-3:]
    if recent_emotions and all(
        e.get("emotion") in ["neutral", "bored"] for e in recent_emotions
    ):
        suggestions["warnings"].append(
            {
                "type": "disengagement",
                "message": "Visitor may be losing interest",
                "suggestion": "Try asking an engaging question or suggest moving to next section",
            }
        )
        suggestions["suggested_topics"] = [
            "Perguntar sobre áreas de interesse do visitante",
            "Contar uma conquista impressionante de alunos",
            "Sugerir visita a um clube ou laboratório",
        ]

    # Check for question overload
    questions_asked = tour_state.get("questions_asked", [])
    if len(questions_asked) > 5:
        suggestions["warnings"].append(
            {
                "type": "question_overload",
                "message": "Many questions asked - visitor might be overwhelmed",
                "suggestion": "Offer to continue tour and answer questions later",
            }
        )

    # Suggest emotional response based on recent visitor emotions
    if recent_emotions:
        latest_emotion = recent_emotions[-1].get("emotion")
        suggestions["response_style"] = _get_response_tone(latest_emotion)

    return {
        "success": True,
        "suggestions": suggestions,
        "context_summary": f"Analyzed {len(recent_emotions)} recent interactions",
    }


def generate_engagement_prompt(situation: str, tool_context: ToolContext) -> dict:
    """
    Generate an engaging prompt or question to keep visitor interested.

    Args:
        situation: Current situation (quiet_moment, between_sections, answering_question, etc.)
        tool_context: ADK tool context

    Returns:
        Engaging prompt or question
    """
    prompts = {
        "quiet_moment": [
            "Vocês têm alguma curiosidade sobre o Inteli?",
            "Alguma área de tecnologia que vocês se interessam especialmente?",
            "Já conheciam o Inteli antes de vir aqui?",
        ],
        "between_sections": [
            "E então? Vamos para a próxima etapa? [latido animado]",
            "Preparados para continuar o tour? Tem coisas muito legais vindo!",
            "Alguma dúvida até aqui antes de seguirmos?",
        ],
        "after_question": [
            "Isso respondeu sua dúvida? *inclina a cabeça*",
            "Quer saber mais alguma coisa sobre esse assunto?",
            "Ficou claro? Posso explicar de outro jeito se quiser!",
        ],
        "re_engagement": [
            "Ei, vocês sabiam que alunos do Inteli já ganharam prêmios internacionais? *balança o rabo*",
            "Deixa eu contar uma coisa legal sobre o Inteli...",
            "Tenho uma história interessante para compartilhar com vocês!",
        ],
        "closing": [
            "Foi um prazer conhecer vocês! *balança o rabo*",
            "Espero ter ajudado a conhecer melhor o Inteli!",
            "Qualquer dúvida, podem me procurar! [latido feliz]",
        ],
    }

    situation_prompts = prompts.get(situation, prompts["quiet_moment"])
    selected_prompt = random.choice(situation_prompts)

    # Store for tracking
    if "engagement_prompts_used" not in tool_context.state:
        tool_context.state["engagement_prompts_used"] = []

    tool_context.state["engagement_prompts_used"].append(
        {"situation": situation, "prompt": selected_prompt}
    )

    return {
        "success": True,
        "situation": situation,
        "prompt": selected_prompt,
        "alternative_prompts": [p for p in situation_prompts if p != selected_prompt],
    }
