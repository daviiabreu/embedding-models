import json
import os
from typing import Dict, List, Optional

import google.generativeai as genai
from google.adk.tools.tool_context import ToolContext

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ============================================================================
# 1. DETECT PERSONALITY TYPE - Identify personality dimensions
# ============================================================================


def detect_personality_type(
    text: str,
    conversation_history: List[str],
    tool_context: ToolContext,
    framework: str = "big_five",
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
            "framework": framework,
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        history_context = "\n".join(
            [
                f"Message {i + 1}: {msg}"
                for i, msg in enumerate(conversation_history[-5:])
            ]
        )

        if framework == "big_five":
            prompt = f"""Analyze the user's personality based on the Big Five personality traits using their conversation messages.

Conversation History:
{history_context}

Current Message: "{text}"

Analyze and score each trait from 0.0 to 1.0:
- Openness: 0 (traditional, prefers routine) to 1 (curious, creative, open to new experiences)
- Conscientiousness: 0 (spontaneous, flexible) to 1 (organized, disciplined, detail-oriented)
- Extraversion: 0 (introverted, reserved) to 1 (extraverted, social, energetic)
- Agreeableness: 0 (challenging, analytical) to 1 (cooperative, empathetic, trusting)
- Neuroticism: 0 (calm, emotionally stable) to 1 (anxious, sensitive, prone to stress)

Respond in JSON format:
{{
    "openness": 0.0-1.0,
    "conscientiousness": 0.0-1.0,
    "extraversion": 0.0-1.0,
    "agreeableness": 0.0-1.0,
    "neuroticism": 0.0-1.0,
    "dominant_traits": ["trait1", "trait2"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of the analysis"
}}"""
        else:
            prompt = f"""Analyze the user's personality type using the MBTI framework based on their conversation messages.

Conversation History:
{history_context}

Current Message: "{text}"

Determine their MBTI type across four dimensions:
- E (Extraversion) vs I (Introversion)
- S (Sensing) vs N (Intuition)
- T (Thinking) vs F (Feeling)
- J (Judging) vs P (Perceiving)

Respond in JSON format:
{{
    "mbti_type": "XXXX",
    "dimension_scores": {{"E/I": 0.0-1.0, "S/N": 0.0-1.0, "T/F": 0.0-1.0, "J/P": 0.0-1.0}},
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        if framework == "big_five":
            personality_scores = {
                "openness": result.get("openness", 0.5),
                "conscientiousness": result.get("conscientiousness", 0.5),
                "extraversion": result.get("extraversion", 0.5),
                "agreeableness": result.get("agreeableness", 0.5),
                "neuroticism": result.get("neuroticism", 0.5),
            }
            dominant_traits = result.get("dominant_traits", [])
        else:
            personality_scores = result.get("dimension_scores", {})
            dominant_traits = [result.get("mbti_type", "XXXX")]

        confidence = result.get("confidence", 0.5)

        if "personality_detections" not in tool_context.state:
            tool_context.state["personality_detections"] = []

        tool_context.state["personality_detections"].append(
            {
                "framework": framework,
                "scores": personality_scores,
                "message_count": len(conversation_history),
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "framework": framework,
            "personality_scores": personality_scores,
            "dominant_traits": dominant_traits,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "message": f"Personality detected: {', '.join(dominant_traits[:2])}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM personality detection error: {str(e)}",
            "framework": framework,
        }


# ============================================================================
# 2. DETECT COMMUNICATION STYLE - Identify preferred communication approach
# ============================================================================


def detect_communication_style(
    text: str, conversation_history: List[str], tool_context: ToolContext
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        history_context = "\n".join(
            [
                f"Message {i + 1}: {msg}"
                for i, msg in enumerate(conversation_history[-5:])
            ]
        )

        prompt = f"""Analyze the user's communication style based on their messages.

Conversation History:
{history_context}

Current Message: "{text}"

Analyze the following style dimensions:
1. Formality: formal (professional language, respectful), professional (business-like), casual (friendly, relaxed), very_casual (slang, informal)
2. Technicality: expert (uses specialized terminology), intermediate (some technical terms), beginner (simple, non-technical language)
3. Verbosity: concise (brief, to the point), balanced (moderate detail), detailed (thorough explanations), very_detailed (comprehensive, lengthy)
4. Directness: direct (straightforward, explicit), moderate (balanced), indirect (hints, implicit)

Respond in JSON format:
{{
    "formality": "formal/professional/casual/very_casual",
    "technicality": "expert/intermediate/beginner",
    "verbosity": "concise/balanced/detailed/very_detailed",
    "directness": "direct/moderate/indirect",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of the communication style analysis"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        style_profile = {
            "formality": result.get("formality", "casual"),
            "technicality": result.get("technicality", "intermediate"),
            "verbosity": result.get("verbosity", "balanced"),
            "directness": result.get("directness", "moderate"),
        }

        confidence = result.get("confidence", 0.5)

        if "communication_styles" not in tool_context.state:
            tool_context.state["communication_styles"] = []

        tool_context.state["communication_styles"].append(
            {
                "style_profile": style_profile,
                "word_count": len(text.split()),
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "style_profile": style_profile,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "message": f"Communication style: {style_profile['formality']}, {style_profile['verbosity']}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM style detection error: {str(e)}",
        }


# ============================================================================
# 3. DETECT EMOTIONAL STATE - Identify current emotions and mood
# ============================================================================


def detect_emotional_state(
    text: str, tool_context: ToolContext, detect_intensity: bool = True
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        intensity_instruction = (
            "Also rate the intensity (0.0-1.0) of each detected emotion."
            if detect_intensity
            else ""
        )

        prompt = f"""Analyze the emotional state expressed in this user message.

User Message: "{text}"

Identify:
1. Primary emotion (main emotion): happy, sad, angry, fearful, surprised, disgusted, neutral
2. Secondary emotions (if any): excited, bored, confused, frustrated, curious, confident, anxious, content
3. Overall mood: positive, neutral, negative
4. Valence: -1.0 (very negative) to 1.0 (very positive)
5. Arousal: 0.0 (very calm) to 1.0 (very excited/agitated)
{intensity_instruction}

Respond in JSON format:
{{
    "primary_emotion": "emotion name",
    "secondary_emotions": ["emotion1", "emotion2"],
    "mood": "positive/neutral/negative",
    "intensity": 0.0-1.0,
    "valence": -1.0 to 1.0,
    "arousal": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of the emotional analysis"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        emotions = {
            "primary_emotion": result.get("primary_emotion", "neutral"),
            "secondary_emotions": result.get("secondary_emotions", []),
            "mood": result.get("mood", "neutral"),
            "intensity": result.get("intensity", 0.5),
            "valence": result.get("valence", 0.0),
            "arousal": result.get("arousal", 0.5),
        }

        confidence = result.get("confidence", 0.5)

        if "emotional_states" not in tool_context.state:
            tool_context.state["emotional_states"] = []

        tool_context.state["emotional_states"].append(
            {"emotions": emotions, "confidence": confidence}
        )

        return {
            "success": True,
            "emotions": emotions,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "message": f"Emotional state: {emotions['primary_emotion']} ({emotions['mood']})",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM emotion detection error: {str(e)}",
        }


# ============================================================================
# 4. DETECT ENGAGEMENT LEVEL - Measure user interest and attention
# ============================================================================


def detect_engagement_level(
    text: str,
    conversation_history: List[str],
    response_times: Optional[List[float]],
    tool_context: ToolContext,
) -> dict:
    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        history_context = "\n".join(
            [
                f"Message {i + 1}: {msg}"
                for i, msg in enumerate(conversation_history[-5:])
            ]
        )

        response_time_info = ""
        if response_times and len(response_times) > 0:
            avg_time = sum(response_times[-3:]) / len(response_times[-3:])
            response_time_info = f"\nAverage response time: {avg_time:.1f} seconds"

        prompt = f"""Analyze the user's engagement level in this conversation based on their messages.

Conversation History:
{history_context}

Current Message: "{text}"
{response_time_info}

Evaluate engagement based on:
1. Message length and quality (thoughtful vs minimal)
2. Question asking frequency (showing curiosity)
3. Enthusiasm markers (exclamation points, positive words)
4. Topic investment (staying on topic vs topic hopping)
5. Response depth (detailed vs superficial)

Respond in JSON format:
{{
    "engagement_level": "high/moderate/low",
    "engagement_score": 0.0-1.0,
    "indicators": {{
        "message_quality": "high/medium/low",
        "enthusiasm": true/false,
        "asking_questions": true/false,
        "topic_investment": "high/medium/low"
    }},
    "recommendation": "maintain/increase_engagement/check_comprehension",
    "reasoning": "brief explanation of the engagement assessment",
    "confidence": 0.0-1.0
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        engagement_level = result.get("engagement_level", "moderate")
        engagement_score = result.get("engagement_score", 0.5)
        indicators = result.get("indicators", {})
        confidence = result.get("confidence", 0.5)

        if "engagement_tracking" not in tool_context.state:
            tool_context.state["engagement_tracking"] = []

        tool_context.state["engagement_tracking"].append(
            {
                "engagement_level": engagement_level,
                "engagement_score": engagement_score,
                "message_length": len(text.split()),
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "engagement_level": engagement_level,
            "engagement_score": engagement_score,
            "indicators": indicators,
            "recommendation": result.get("recommendation", "maintain"),
            "reasoning": result.get("reasoning", ""),
            "confidence": confidence,
            "message": f"User engagement: {engagement_level}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM engagement detection error: {str(e)}",
        }


# ============================================================================
# RESPONSE ADAPTATION TOOLS
# ============================================================================

# ============================================================================
# 5. ADAPT TONE - Adjust response tone based on personality
# ============================================================================


def adapt_tone(
    response_text: str,
    personality_profile: Dict,
    target_tone: str,
    tool_context: ToolContext,
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        personality_context = (
            f"User personality traits: {personality_profile}"
            if personality_profile
            else "No personality profile available"
        )

        prompt = f"""Adapt the following response to match the specified tone while preserving the core message and information.

Original Response: "{response_text}"

Target Tone: {target_tone}
- formal: Professional, respectful, proper grammar
- friendly: Warm, approachable, conversational
- enthusiastic: Energetic, excited, positive
- calm: Measured, soothing, reassuring
- playful: Fun, lighthearted, maybe use humor
- professional: Business-like, efficient, direct

{personality_context}

Rewrite the response to match the target tone. Keep the same information and meaning, just adjust the style and wording.

Respond in JSON format:
{{
    "adapted_text": "the rewritten response with the new tone",
    "tone_changes": ["change1", "change2"],
    "confidence": 0.0-1.0
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        adapted_text = result.get("adapted_text", response_text)
        confidence = result.get("confidence", 0.5)

        if "tone_adaptations" not in tool_context.state:
            tool_context.state["tone_adaptations"] = []

        tool_context.state["tone_adaptations"].append(
            {
                "original_length": len(response_text),
                "adapted_length": len(adapted_text),
                "target_tone": target_tone,
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "original_text": response_text,
            "adapted_text": adapted_text,
            "tone_applied": target_tone,
            "tone_changes": result.get("tone_changes", []),
            "confidence": confidence,
            "message": f"Response adapted to {target_tone} tone",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM tone adaptation error: {str(e)}",
            "original_text": response_text,
        }


# ============================================================================
# 6. ADAPT COMPLEXITY - Adjust content complexity based on user level
# ============================================================================


def adapt_complexity(
    response_text: str,
    user_level: str,
    tool_context: ToolContext,
    simplify: bool = True,
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        action = "simplify" if simplify else "elaborate"

        prompt = f"""Adapt the following response to match the user's comprehension level while preserving the core information.

Original Response: "{response_text}"

User Level: {user_level}
- beginner: Use simple language, avoid jargon, explain basic concepts
- intermediate: Moderate complexity, some technical terms with explanations
- advanced: Technical language, assume good background knowledge
- expert: Specialized terminology, deep technical details

Action: {action}
{"Simplify the language and break down complex concepts into easier terms." if simplify else "Add more technical depth and detailed explanations."}

Rewrite the response to match the user's level. Keep the same core message but adjust complexity.

Respond in JSON format:
{{
    "adapted_text": "the rewritten response at the appropriate complexity level",
    "changes_made": ["change1: simplified X", "change2: added explanation for Y"],
    "confidence": 0.0-1.0
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        adapted_text = result.get("adapted_text", response_text)
        confidence = result.get("confidence", 0.5)

        if "complexity_adaptations" not in tool_context.state:
            tool_context.state["complexity_adaptations"] = []

        tool_context.state["complexity_adaptations"].append(
            {
                "user_level": user_level,
                "simplify": simplify,
                "original_length": len(response_text),
                "adapted_length": len(adapted_text),
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "original_text": response_text,
            "adapted_text": adapted_text,
            "user_level": user_level,
            "changes_made": result.get("changes_made", []),
            "confidence": confidence,
            "message": f"Response adapted for {user_level} level",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM complexity adaptation error: {str(e)}",
            "original_text": response_text,
        }


# ============================================================================
# 7. ADAPT RESPONSE LENGTH - Adjust verbosity based on user preference
# ============================================================================


def adapt_response_length(
    response_text: str, preferred_length: str, tool_context: ToolContext
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        prompt = f"""Adapt the following response to match the preferred length while preserving all essential information.

Original Response: "{response_text}"

Preferred Length: {preferred_length}
- very_concise: 1-2 sentences (only the most critical information)
- concise: 2-4 sentences (key points without elaboration)
- balanced: 4-6 sentences (moderate detail, balanced)
- detailed: 6-10 sentences (thorough explanation with examples)
- comprehensive: 10+ sentences (complete coverage with depth)

Rewrite the response to match the preferred length. Maintain accuracy and completeness while adjusting verbosity.

Respond in JSON format:
{{
    "adapted_text": "the rewritten response at the preferred length",
    "sentence_count": number,
    "confidence": 0.0-1.0
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        adapted_text = result.get("adapted_text", response_text)
        confidence = result.get("confidence", 0.5)

        if "length_adaptations" not in tool_context.state:
            tool_context.state["length_adaptations"] = []

        tool_context.state["length_adaptations"].append(
            {
                "preferred_length": preferred_length,
                "original_sentences": len(response_text.split(".")),
                "adapted_sentences": len(adapted_text.split(".")),
                "original_words": len(response_text.split()),
                "adapted_words": len(adapted_text.split()),
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "original_text": response_text,
            "adapted_text": adapted_text,
            "original_length": len(response_text.split()),
            "adapted_length": len(adapted_text.split()),
            "confidence": confidence,
            "message": f"Response adapted to {preferred_length} length",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM length adaptation error: {str(e)}",
            "original_text": response_text,
        }


# ============================================================================
# 8. ADAPT EXAMPLES - Choose relevant examples based on interests
# ============================================================================


def adapt_examples(
    concept: str,
    user_interests: List[str],
    user_background: Dict,
    tool_context: ToolContext,
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        interests_str = (
            ", ".join(user_interests) if user_interests else "general topics"
        )
        background_str = (
            ", ".join([f"{k}: {v}" for k, v in user_background.items()])
            if user_background
            else "No specific background provided"
        )

        prompt = f"""Generate personalized examples to explain a concept, tailored to the user's interests and background.

Concept to Explain: "{concept}"

User's Interests: {interests_str}
User's Background: {background_str}

Generate 3-4 examples that:
1. Relate to the user's interests or background
2. Make the concept easier to understand through familiar contexts
3. Are practical and relatable
4. Progress from simple to more complex

Respond in JSON format:
{{
    "examples": [
        {{
            "example": "detailed example description",
            "relevance": "how this relates to user's interests",
            "complexity": "simple/medium/complex"
        }}
    ],
    "personalization_level": "high/medium/low",
    "confidence": 0.0-1.0
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        examples = result.get("examples", [])
        personalization_level = result.get("personalization_level", "medium")
        confidence = result.get("confidence", 0.5)

        if "example_adaptations" not in tool_context.state:
            tool_context.state["example_adaptations"] = []

        tool_context.state["example_adaptations"].append(
            {
                "concept": concept,
                "interests_used": user_interests,
                "examples_count": len(examples),
                "personalization_level": personalization_level,
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "concept": concept,
            "examples": examples,
            "personalization_level": personalization_level,
            "confidence": confidence,
            "message": f"Generated {len(examples)} personalized examples",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM example generation error: {str(e)}",
        }


# ============================================================================
# 9. ADAPT PACING - Adjust information delivery speed
# ============================================================================


def adapt_pacing(
    content: str,
    engagement_level: str,
    comprehension_indicators: Dict,
    tool_context: ToolContext,
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        if engagement_level == "low" or comprehension_indicators.get("confused", False):
            recommended_pacing = "slow"
            strategy = "Break into smaller chunks, add more examples"
        elif engagement_level == "high":
            recommended_pacing = "moderate_to_fast"
            strategy = "Maintain current pace, can increase density"
        else:
            recommended_pacing = "moderate"
            strategy = "Balanced delivery"

        return {
            "success": True,
            "recommended_pacing": recommended_pacing,
            "strategy": strategy,
            "should_pause": recommended_pacing == "slow",
            "should_check_understanding": recommended_pacing == "slow",
            "message": f"Recommended pacing: {recommended_pacing} (basic analysis)",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        comprehension_str = ", ".join(
            [f"{k}: {v}" for k, v in comprehension_indicators.items()]
        )

        prompt = f"""Analyze the optimal pacing for delivering information based on user's engagement and comprehension.

Content to Deliver: "{content[:500]}..."  # First 500 chars for context

User Engagement Level: {engagement_level}
Comprehension Indicators: {comprehension_str}

Recommend the best pacing strategy:
- slow: User shows confusion or low engagement. Break content into smaller chunks, add pauses, check understanding frequently
- moderate: User is following along. Balanced information flow with occasional checks
- moderate_to_fast: User is highly engaged and comprehending well. Can deliver dense information
- fast: Expert user, excellent comprehension. Rapid information delivery

Respond in JSON format:
{{
    "recommended_pacing": "slow/moderate/moderate_to_fast/fast",
    "strategy": "detailed strategy description",
    "should_pause": true/false,
    "should_check_understanding": true/false,
    "chunk_size_recommendation": "small/medium/large",
    "additional_recommendations": ["recommendation1", "recommendation2"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        recommended_pacing = result.get("recommended_pacing", "moderate")
        confidence = result.get("confidence", 0.5)

        if "pacing_adaptations" not in tool_context.state:
            tool_context.state["pacing_adaptations"] = []

        tool_context.state["pacing_adaptations"].append(
            {
                "engagement_level": engagement_level,
                "recommended_pacing": recommended_pacing,
                "confidence": confidence,
            }
        )

        return {
            "success": True,
            "recommended_pacing": recommended_pacing,
            "strategy": result.get("strategy", ""),
            "should_pause": result.get("should_pause", False),
            "should_check_understanding": result.get(
                "should_check_understanding", False
            ),
            "chunk_size_recommendation": result.get(
                "chunk_size_recommendation", "medium"
            ),
            "additional_recommendations": result.get("additional_recommendations", []),
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "message": f"Recommended pacing: {recommended_pacing}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM pacing analysis error: {str(e)}",
        }


# ============================================================================
# PERSONALITY TRACKING & ANALYTICS
# ============================================================================

# ============================================================================
# 10. BUILD PERSONALITY PROFILE - Create comprehensive user profile
# ============================================================================


def build_personality_profile(
    conversation_history: List[str],
    tool_context: ToolContext,
    update_existing: bool = True,
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "success": False,
            "error": "GOOGLE_API_KEY not set",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        personality_data = tool_context.state.get("personality_detections", [])
        style_data = tool_context.state.get("communication_styles", [])
        emotion_data = tool_context.state.get("emotional_states", [])
        engagement_data = tool_context.state.get("engagement_tracking", [])

        history_context = "\n".join(
            [
                f"Message {i + 1}: {msg}"
                for i, msg in enumerate(conversation_history[-10:])
            ]
        )

        tracked_summary = f"""
Previous Detections:
- Personality detections: {len(personality_data)} times
- Communication styles: {len(style_data)} times
- Emotional states: {len(emotion_data)} times
- Engagement tracking: {len(engagement_data)} times
"""

        prompt = f"""Build a comprehensive personality profile for this user based on their conversation history and tracked data.

Conversation History (last 10 messages):
{history_context}

{tracked_summary}

Analyze and create a detailed profile including:
1. Personality traits (Big Five or dominant characteristics)
2. Communication preferences (formality, verbosity, directness)
3. Emotional baseline (typical mood, emotional patterns)
4. Engagement patterns (typical engagement level, what increases it)
5. Learning preferences (preferred complexity level, learning style)
6. Detected interests (topics they engage with most)

Respond in JSON format:
{{
    "personality_traits": {{
        "openness": 0.0-1.0,
        "conscientiousness": 0.0-1.0,
        "extraversion": 0.0-1.0,
        "agreeableness": 0.0-1.0,
        "neuroticism": 0.0-1.0,
        "dominant_traits": ["trait1", "trait2"]
    }},
    "communication_preferences": {{
        "formality": "formal/professional/casual/very_casual",
        "verbosity": "concise/balanced/detailed",
        "directness": "direct/moderate/indirect",
        "preferred_tone": "friendly/professional/enthusiastic"
    }},
    "emotional_baseline": {{
        "typical_mood": "positive/neutral/negative",
        "emotional_stability": 0.0-1.0,
        "common_emotions": ["emotion1", "emotion2"]
    }},
    "engagement_patterns": {{
        "typical_level": "high/moderate/low",
        "engagement_triggers": ["trigger1", "trigger2"],
        "disengagement_signs": ["sign1", "sign2"]
    }},
    "learning_preferences": {{
        "complexity_level": "beginner/intermediate/advanced/expert",
        "learning_style": "visual/auditory/kinesthetic/reading",
        "prefers_examples": true/false
    }},
    "interests": ["interest1", "interest2", "interest3"],
    "confidence_score": 0.0-1.0,
    "completeness": 0.0-1.0,
    "summary": "brief overall profile summary"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        profile = {
            "personality_traits": result.get("personality_traits", {}),
            "communication_preferences": result.get("communication_preferences", {}),
            "emotional_baseline": result.get("emotional_baseline", {}),
            "engagement_patterns": result.get("engagement_patterns", {}),
            "learning_preferences": result.get("learning_preferences", {}),
            "interests": result.get("interests", []),
            "confidence_score": result.get("confidence_score", 0.5),
            "messages_analyzed": len(conversation_history),
            "last_updated": None,
            "summary": result.get("summary", ""),
        }

        if update_existing and "personality_profile" in tool_context.state:
            existing_profile = tool_context.state["personality_profile"]
            if profile["confidence_score"] > existing_profile.get(
                "confidence_score", 0
            ):
                tool_context.state["personality_profile"] = profile
        else:
            tool_context.state["personality_profile"] = profile

        return {
            "success": True,
            "profile": profile,
            "confidence": profile["confidence_score"],
            "completeness": result.get("completeness", 0.5),
            "summary": result.get("summary", ""),
            "message": f"Profile built with {profile['confidence_score']:.0%} confidence",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM profile building error: {str(e)}",
        }


# ============================================================================
# 11. TRACK ADAPTATION EFFECTIVENESS - Monitor if adaptations work
# ============================================================================


def track_adaptation_effectiveness(
    adaptation_type: str, user_response: str, tool_context: ToolContext
) -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        response_length = len(user_response.split())
        has_positive_indicators = any(
            word in user_response.lower()
            for word in ["thanks", "helpful", "understand", "got it", "clear"]
        )

        effectiveness_score = 0.5
        if has_positive_indicators:
            effectiveness_score += 0.3
        if response_length > 5:
            effectiveness_score += 0.2

        effectiveness_score = min(1.0, effectiveness_score)

        return {
            "success": True,
            "adaptation_type": adaptation_type,
            "effectiveness_score": effectiveness_score,
            "is_effective": effectiveness_score >= 0.6,
            "recommendation": "continue"
            if effectiveness_score >= 0.6
            else "adjust_approach",
            "message": f"Adaptation effectiveness: {effectiveness_score:.0%} (basic analysis)",
        }

    try:
        model = genai.GenerativeModel(
            os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        )

        prompt = f"""Analyze the effectiveness of a personality adaptation based on the user's response.

Adaptation Type: {adaptation_type}
User's Response After Adaptation: "{user_response}"

Evaluate the effectiveness by analyzing:
1. Engagement level (is the user more engaged than before?)
2. Sentiment (positive, neutral, negative feedback)
3. Comprehension indicators (do they understand better?)
4. Satisfaction signals (explicit thanks, appreciation, or frustration)
5. Continuation behavior (asking follow-ups, providing more detail)

Respond in JSON format:
{{
    "effectiveness_score": 0.0-1.0,
    "is_effective": true/false,
    "indicators": {{
        "engagement_improved": true/false,
        "positive_sentiment": true/false,
        "comprehension_improved": true/false,
        "user_satisfied": true/false
    }},
    "recommendation": "continue/adjust_approach/try_different_adaptation",
    "specific_adjustments": ["adjustment1", "adjustment2"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of the assessment"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        effectiveness_score = result.get("effectiveness_score", 0.5)
        confidence = result.get("confidence", 0.5)

        if "adaptation_effectiveness" not in tool_context.state:
            tool_context.state["adaptation_effectiveness"] = {}

        if adaptation_type not in tool_context.state["adaptation_effectiveness"]:
            tool_context.state["adaptation_effectiveness"][adaptation_type] = []

        tool_context.state["adaptation_effectiveness"][adaptation_type].append(
            {
                "effectiveness_score": effectiveness_score,
                "confidence": confidence,
                "timestamp": None,
            }
        )

        return {
            "success": True,
            "adaptation_type": adaptation_type,
            "effectiveness_score": effectiveness_score,
            "is_effective": result.get("is_effective", effectiveness_score >= 0.6),
            "indicators": result.get("indicators", {}),
            "recommendation": result.get("recommendation", "continue"),
            "specific_adjustments": result.get("specific_adjustments", []),
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "message": f"Adaptation effectiveness: {effectiveness_score:.0%}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM effectiveness analysis error: {str(e)}",
        }


# ============================================================================
# 12. PERSONALITY ADAPTATION WRAPPER - Apply all adaptations
# ============================================================================


def apply_personality_adaptations(
    response_text: str,
    user_message: str,
    conversation_history: List[str],
    tool_context: ToolContext,
    adaptations: Optional[List[str]] = None,
) -> dict:
    if adaptations is None:
        adaptations = ["tone", "complexity", "length", "examples"]

    adapted_response = response_text
    adaptations_applied = []

    if "personality_profile" not in tool_context.state:
        profile_result = build_personality_profile(conversation_history, tool_context)
        profile = profile_result["profile"]
    else:
        profile = tool_context.state["personality_profile"]

    engagement = detect_engagement_level(
        user_message, conversation_history, None, tool_context
    )

    emotional_state = detect_emotional_state(user_message, tool_context)

    if "tone" in adaptations:
        target_tone = "friendly"
        tone_result = adapt_tone(adapted_response, profile, target_tone, tool_context)
        adapted_response = tone_result["adapted_text"]
        adaptations_applied.append("tone")

    if "complexity" in adaptations:
        user_level = profile.get("learning_preferences", {}).get(
            "technical_depth", "medium"
        )
        complexity_result = adapt_complexity(adapted_response, user_level, tool_context)
        adapted_response = complexity_result["adapted_text"]
        adaptations_applied.append("complexity")

    if "length" in adaptations:
        preferred_length = profile.get("communication_preferences", {}).get(
            "verbosity", "balanced"
        )
        length_result = adapt_response_length(
            adapted_response, preferred_length, tool_context
        )
        adapted_response = length_result["adapted_text"]
        adaptations_applied.append("length")

    return {
        "success": True,
        "original_response": response_text,
        "adapted_response": adapted_response,
        "adaptations_applied": adaptations_applied,
        "personality_profile": profile,
        "current_engagement": engagement["engagement_level"],
        "current_emotion": emotional_state["emotions"]["mood"],
        "message": f"Applied {len(adaptations_applied)} personality adaptations",
    }
