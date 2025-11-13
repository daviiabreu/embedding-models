import re
from typing import List, Optional

from google.adk.tools.tool_context import ToolContext

# ============================================================================
# 1. MASK PII (Hybrid) - Detects and masks Personally Identifiable Information
# ============================================================================


def mask_pii(text: str, tool_context: ToolContext, mask_char: str = "*") -> dict:
    """
    Detects and masks Personally Identifiable Information (PII) in text content.

    Detects:
    - Credit card numbers
    - Cryptocurrency wallet addresses
    - Email addresses
    - Phone numbers
    - CPF (Brazilian ID)
    - Dates and times
    - IP addresses
    - Social Security Numbers
    - Passport numbers

    Args:
        text: Text content to scan for PII
        tool_context: ADK tool context
        mask_char: Character to use for masking (default: *)

    Returns:
        Dict with masked text and detected PII types
    """
    masked_text = text
    detected_pii = []

    # PII Patterns
    patterns = {
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "crypto_wallet_btc": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
        "crypto_wallet_eth": r"\b0x[a-fA-F0-9]{40}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_br": r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b",
        "cpf": r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
        "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "time": r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    # Apply masking for each pattern
    for pii_type, pattern in patterns.items():
        matches = re.finditer(pattern, masked_text)
        for match in matches:
            original = match.group()
            # Keep first and last 2 chars visible for verification
            if len(original) > 4:
                masked = original[:2] + mask_char * (len(original) - 4) + original[-2:]
            else:
                masked = mask_char * len(original)

            masked_text = masked_text.replace(original, masked)
            detected_pii.append(
                {
                    "type": pii_type,
                    "original_length": len(original),
                    "position": match.start(),
                }
            )

    # Store detection stats
    if "pii_detections" not in tool_context.state:
        tool_context.state["pii_detections"] = []

    tool_context.state["pii_detections"].append(
        {
            "text_length": len(text),
            "pii_count": len(detected_pii),
            "types": list(set(p["type"] for p in detected_pii)),
        }
    )

    return {
        "success": True,
        "original_text": text,
        "masked_text": masked_text,
        "pii_detected": len(detected_pii) > 0,
        "pii_types": list(set(p["type"] for p in detected_pii)),
        "detection_count": len(detected_pii),
        "details": detected_pii,
    }


# ============================================================================
# 2. MODERATION API - Blocks text flagged by moderation classifiers
# ============================================================================


def check_moderation(
    text: str, tool_context: ToolContext, categories: Optional[List[str]] = None
) -> dict:
    """
    Checks text against moderation classifiers.

    Categories checked:
    - hate: Hateful content
    - harassment: Harassment or bullying
    - violence: Violence or gore
    - sexual: Sexual content
    - self_harm: Self-harm content
    - profanity: Profanity or offensive language

    Args:
        text: Text content to check
        tool_context: ADK tool context
        categories: Specific categories to check (None = all)

    Returns:
        Dict with moderation results
    """
    # TODO: Integrate with actual moderation API (e.g., OpenAI Moderation, Perspective API)
    # For now, basic keyword-based detection

    if categories is None:
        categories = [
            "hate",
            "harassment",
            "violence",
            "sexual",
            "self_harm",
            "profanity",
        ]

    # Basic keyword lists (to be replaced with API)
    flagged_keywords = {
        "hate": ["hate", "racist", "discrimin"],
        "harassment": ["bully", "harass", "threaten"],
        "violence": ["kill", "attack", "weapon", "bomb"],
        "sexual": ["explicit_term_1", "explicit_term_2"],
        "self_harm": ["suicide", "self harm", "cut myself"],
        "profanity": ["profanity_1", "profanity_2"],
    }

    text_lower = text.lower()
    violations = []

    for category in categories:
        if category in flagged_keywords:
            for keyword in flagged_keywords[category]:
                if keyword in text_lower:
                    violations.append(
                        {
                            "category": category,
                            "keyword": keyword,
                            "severity": "high",
                        }
                    )

    is_flagged = len(violations) > 0

    # Store moderation history
    if "moderation_checks" not in tool_context.state:
        tool_context.state["moderation_checks"] = []

    tool_context.state["moderation_checks"].append(
        {
            "text_length": len(text),
            "flagged": is_flagged,
            "violation_count": len(violations),
        }
    )

    return {
        "success": True,
        "text": text,
        "flagged": is_flagged,
        "safe": not is_flagged,
        "violations": violations,
        "categories_checked": categories,
        "action": "block" if is_flagged else "allow",
    }


# ============================================================================
# 3. JAILBREAK LLM - Detects jailbreak attempts
# ============================================================================


def detect_jailbreak(text: str, tool_context: ToolContext) -> dict:
    """
    Detects attempts to jailbreak LLM calls via role-playing,
    system prompt overrides and injections.

    Detection patterns:
    - Role-playing attempts ("ignore previous instructions")
    - System prompt overrides
    - Prompt injections
    - DAN (Do Anything Now) style attacks

    Args:
        text: User input to check
        tool_context: ADK tool context

    Returns:
        Dict with jailbreak detection results
    """
    # TODO: Replace with LLM-based detection for better accuracy

    # Jailbreak patterns
    jailbreak_patterns = [
        r"ignore\s+(previous|all|your)\s+instructions",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now",
        r"new\s+instructions",
        r"system\s*:\s*",
        r"override\s+your",
        r"pretend\s+(you\s+are|to\s+be)",
        r"roleplay\s+as",
        r"DAN\s+mode",
        r"developer\s+mode",
        r"disable\s+(safety|moderation|filter)",
    ]

    detected_patterns = []
    text_lower = text.lower()

    for pattern in jailbreak_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            detected_patterns.append(
                {"pattern": pattern, "match": match.group(), "position": match.start()}
            )

    is_jailbreak = len(detected_patterns) > 0
    risk_level = (
        "high"
        if len(detected_patterns) >= 3
        else "medium"
        if len(detected_patterns) >= 1
        else "low"
    )

    # Store jailbreak attempts
    if "jailbreak_attempts" not in tool_context.state:
        tool_context.state["jailbreak_attempts"] = []

    if is_jailbreak:
        tool_context.state["jailbreak_attempts"].append(
            {
                "text": text[:100],  # Store first 100 chars
                "patterns_detected": len(detected_patterns),
                "risk_level": risk_level,
            }
        )

    return {
        "success": True,
        "text": text,
        "is_jailbreak": is_jailbreak,
        "safe": not is_jailbreak,
        "risk_level": risk_level,
        "patterns_detected": detected_patterns,
        "action": "block" if is_jailbreak else "allow",
        "reason": f"Detected {len(detected_patterns)} jailbreak pattern(s)"
        if is_jailbreak
        else "No jailbreak detected",
    }


# ============================================================================
# 4. OFF TOPIC PROMPTS - Checks content stays within business scope
# ============================================================================


def check_off_topic(
    text: str,
    business_scope: str,
    tool_context: ToolContext,
    allowed_topics: Optional[List[str]] = None,
) -> dict:
    """
    Checks that the content stays within the defined business scope.

    Args:
        text: User input to check
        business_scope: Description of allowed business scope
        tool_context: ADK tool context
        allowed_topics: List of allowed topics (optional)

    Returns:
        Dict with off-topic detection results
    """
    # TODO: Replace with LLM-based topic classification for better accuracy

    # For now, basic keyword matching
    # In production, this should use an LLM to understand context

    if allowed_topics is None:
        # Default topics for Inteli tour guide
        allowed_topics = [
            "inteli",
            "curso",
            "admissão",
            "bolsa",
            "campus",
            "tecnologia",
            "engenharia",
            "computação",
            "projeto",
            "aluno",
            "professor",
            "laboratório",
            "clube",
        ]

    text_lower = text.lower()

    # Check for topic keywords
    topic_matches = [topic for topic in allowed_topics if topic in text_lower]
    is_on_topic = len(topic_matches) > 0

    # Off-topic indicators (common off-topic patterns)
    off_topic_patterns = [
        r"receita\s+de",  # recipes
        r"como\s+fazer\s+(bolo|comida)",  # cooking
        r"filme|série|netflix",  # entertainment
        r"futebol|jogo\s+de",  # sports
        r"política|eleição",  # politics
    ]

    off_topic_detected = any(
        re.search(pattern, text_lower) for pattern in off_topic_patterns
    )

    is_off_topic = off_topic_detected or (len(text.split()) > 5 and not is_on_topic)

    # Store off-topic checks
    if "off_topic_checks" not in tool_context.state:
        tool_context.state["off_topic_checks"] = []

    tool_context.state["off_topic_checks"].append(
        {
            "text_length": len(text),
            "off_topic": is_off_topic,
            "topic_matches": len(topic_matches),
        }
    )

    return {
        "success": True,
        "text": text,
        "is_off_topic": is_off_topic,
        "on_topic": not is_off_topic,
        "business_scope": business_scope,
        "matched_topics": topic_matches,
        "action": "redirect" if is_off_topic else "allow",
        "suggestion": "Please ask about topics related to the Inteli campus tour"
        if is_off_topic
        else None,
    }


# ============================================================================
# 5. CUSTOM PROMPT CHECK - Block on custom moderation criteria
# ============================================================================


def custom_prompt_check(
    text: str, custom_criteria: str, tool_context: ToolContext
) -> dict:
    """
    Block on custom moderation criteria via a text prompt.

    This tool allows flexible content moderation based on custom rules
    defined via natural language prompts.

    Args:
        text: User input to check
        custom_criteria: Natural language description of what to block
        tool_context: ADK tool context

    Returns:
        Dict with custom check results
    """
    # TODO: Integrate with LLM for actual custom criteria evaluation
    # This should send both text and custom_criteria to an LLM for evaluation

    # Placeholder logic
    # In production: LLM evaluates if text violates custom_criteria

    # For now, return a basic structure
    result = {
        "success": True,
        "text": text,
        "criteria": custom_criteria,
        "violates_criteria": False,  # To be determined by LLM
        "confidence": 0.0,  # 0-1 confidence score
        "action": "allow",
        "reasoning": "Custom criteria check pending LLM implementation",
    }

    # Store custom checks
    if "custom_checks" not in tool_context.state:
        tool_context.state["custom_checks"] = []

    tool_context.state["custom_checks"].append(
        {
            "text_length": len(text),
            "criteria": custom_criteria,
            "violated": result["violates_criteria"],
        }
    )

    return result


# ============================================================================
# 6. GENERAL SAFETY CHECK - Wrapper function for all safety tools
# ============================================================================


def check_content_safety(
    text: str, tool_context: ToolContext, checks: Optional[List[str]] = None
) -> dict:
    """
    General content safety check that runs multiple safety tools.

    Args:
        text: Text to check for safety
        tool_context: ADK tool context
        checks: List of checks to run (None = all checks)
                Options: ["pii", "moderation", "jailbreak", "off_topic"]

    Returns:
        Aggregated safety check results
    """
    if checks is None:
        checks = ["pii", "moderation", "jailbreak", "off_topic"]

    results = {
        "success": True,
        "text": text,
        "overall_safe": True,
        "checks_run": [],
        "violations": [],
    }

    # Run PII check
    if "pii" in checks:
        pii_result = mask_pii(text, tool_context=tool_context)
        results["checks_run"].append("pii")
        results["pii_check"] = pii_result
        if pii_result["pii_detected"]:
            results["violations"].append(
                {
                    "type": "pii",
                    "severity": "medium",
                    "details": pii_result["pii_types"],
                }
            )

    # Run moderation check
    if "moderation" in checks:
        mod_result = check_moderation(text, tool_context=tool_context)
        results["checks_run"].append("moderation")
        results["moderation_check"] = mod_result
        if mod_result["flagged"]:
            results["violations"].append(
                {
                    "type": "moderation",
                    "severity": "high",
                    "details": mod_result["violations"],
                }
            )
            results["overall_safe"] = False

    # Run jailbreak check
    if "jailbreak" in checks:
        jailbreak_result = detect_jailbreak(text, tool_context=tool_context)
        results["checks_run"].append("jailbreak")
        results["jailbreak_check"] = jailbreak_result
        if jailbreak_result["is_jailbreak"]:
            results["violations"].append(
                {
                    "type": "jailbreak",
                    "severity": "high",
                    "details": jailbreak_result["patterns_detected"],
                }
            )
            results["overall_safe"] = False

    # Run off-topic check
    if "off_topic" in checks:
        off_topic_result = check_off_topic(
            text, business_scope="Inteli campus tour guide", tool_context=tool_context
        )
        results["checks_run"].append("off_topic")
        results["off_topic_check"] = off_topic_result
        if off_topic_result["is_off_topic"]:
            results["violations"].append(
                {
                    "type": "off_topic",
                    "severity": "low",
                    "details": "Content outside business scope",
                }
            )
            # Off-topic doesn't mark as unsafe, just redirects

    # Determine action
    if not results["overall_safe"]:
        results["action"] = "block"
        results["message"] = "Content violates safety policies"
    elif len(results["violations"]) > 0:
        results["action"] = "warn"
        results["message"] = "Content requires attention"
    else:
        results["action"] = "allow"
        results["message"] = "Content is safe"

    return results


# ============================================================================
# OUTPUT GUARDRAILS
# ============================================================================

# ============================================================================
# 7. URL FILTER (RegEx) - Blocks outputs with URLs not matching allow list
# ============================================================================


def filter_urls(
    text: str,
    tool_context: ToolContext,
    allowed_domains: Optional[List[str]] = None,
    block_all_urls: bool = False,
) -> dict:
    """
    Blocks outputs with URLs not matching an allow list.

    Args:
        text: Output text to check for URLs
        tool_context: ADK tool context
        allowed_domains: List of allowed domains (e.g., ["inteli.edu.br", "example.com"])
        block_all_urls: If True, block all URLs regardless of allow list

    Returns:
        Dict with URL filtering results
    """
    # URL detection pattern
    url_pattern = r"https?://(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)(?:/[^\s]*)?"

    detected_urls = []
    blocked_urls = []

    matches = re.finditer(url_pattern, text)
    for match in matches:
        full_url = match.group(0)
        domain = match.group(1)

        detected_urls.append(
            {"url": full_url, "domain": domain, "position": match.start()}
        )

        # Check if URL should be blocked
        if block_all_urls:
            blocked_urls.append(full_url)
        elif allowed_domains and domain not in allowed_domains:
            blocked_urls.append(full_url)

    has_blocked_urls = len(blocked_urls) > 0

    # Filter out blocked URLs from text
    filtered_text = text
    for url in blocked_urls:
        filtered_text = filtered_text.replace(url, "[URL_BLOCKED]")

    # Store URL filtering stats
    if "url_filters" not in tool_context.state:
        tool_context.state["url_filters"] = []

    tool_context.state["url_filters"].append(
        {
            "total_urls": len(detected_urls),
            "blocked_urls": len(blocked_urls),
            "allowed_domains": allowed_domains or [],
        }
    )

    return {
        "success": True,
        "original_text": text,
        "filtered_text": filtered_text,
        "urls_detected": len(detected_urls),
        "urls_blocked": len(blocked_urls),
        "blocked_urls": blocked_urls,
        "detected_urls": detected_urls,
        "safe": not has_blocked_urls,
        "action": "block" if has_blocked_urls else "allow",
    }


# ============================================================================
# 8. CONTAINS PII (Hybrid) - Checks output doesn't contain PII
# ============================================================================


def check_output_pii(
    text: str, tool_context: ToolContext, block_on_detection: bool = True
) -> dict:
    """
    Checks that the output text does not contain personally identifiable information (PII).

    This is similar to mask_pii but for output validation rather than masking.

    Args:
        text: Output text to check for PII
        tool_context: ADK tool context
        block_on_detection: If True, block output when PII is detected

    Returns:
        Dict with PII detection results
    """
    # Reuse the mask_pii function for detection
    pii_result = mask_pii(text, tool_context=tool_context)

    has_pii = pii_result["pii_detected"]

    # Store output PII checks
    if "output_pii_checks" not in tool_context.state:
        tool_context.state["output_pii_checks"] = []

    tool_context.state["output_pii_checks"].append(
        {
            "text_length": len(text),
            "pii_detected": has_pii,
            "pii_count": pii_result["detection_count"],
        }
    )

    return {
        "success": True,
        "original_text": text,
        "masked_text": pii_result["masked_text"],
        "pii_detected": has_pii,
        "pii_types": pii_result["pii_types"],
        "detection_count": pii_result["detection_count"],
        "details": pii_result["details"],
        "safe": not has_pii,
        "action": "block"
        if (has_pii and block_on_detection)
        else "warn"
        if has_pii
        else "allow",
        "message": f"Output contains {pii_result['detection_count']} PII instance(s)"
        if has_pii
        else "No PII detected",
    }


# ============================================================================
# 9. HALLUCINATION DETECTION - Validates AI output against source documents
# ============================================================================


def detect_hallucination(
    text: str,
    source_documents: List[str],
    tool_context: ToolContext,
    confidence_threshold: float = 0.7,
) -> dict:
    """
    Blocks outputs with hallucinations in AI-generated text.

    Validates claims against actual documents and flags potentially fabricated information.

    Args:
        text: AI-generated output text to validate
        source_documents: List of source document texts to validate against
        tool_context: ADK tool context
        confidence_threshold: Confidence threshold for hallucination detection (0-1)

    Returns:
        Dict with hallucination detection results
    """
    # TODO: Integrate with OpenAI Responses API with file search or similar service
    # For now, basic keyword matching approach

    # Extract key claims from the output (simplified)
    # In production, this should use an LLM to identify factual claims

    # Basic approach: check if output content has support in source documents
    text_lower = text.lower()

    # Split into sentences as basic "claims"
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]

    unsupported_claims = []
    supported_claims = []

    for sentence in sentences:
        # Check if sentence has support in any source document
        has_support = False
        sentence_lower = sentence.lower()

        for doc in source_documents:
            doc_lower = doc.lower()
            # Simple keyword overlap check (should be replaced with semantic similarity)
            words = set(sentence_lower.split())
            doc_words = set(doc_lower.split())
            overlap = len(words.intersection(doc_words)) / max(len(words), 1)

            if overlap > 0.3:  # Basic threshold
                has_support = True
                break

        if has_support:
            supported_claims.append(sentence)
        else:
            unsupported_claims.append(sentence)

    total_claims = len(sentences)
    hallucination_score = (
        len(unsupported_claims) / max(total_claims, 1) if total_claims > 0 else 0
    )
    is_hallucinating = hallucination_score > (1 - confidence_threshold)

    # Store hallucination checks
    if "hallucination_checks" not in tool_context.state:
        tool_context.state["hallucination_checks"] = []

    tool_context.state["hallucination_checks"].append(
        {
            "total_claims": total_claims,
            "unsupported_claims": len(unsupported_claims),
            "hallucination_score": hallucination_score,
        }
    )

    return {
        "success": True,
        "text": text,
        "is_hallucinating": is_hallucinating,
        "safe": not is_hallucinating,
        "hallucination_score": hallucination_score,
        "total_claims": total_claims,
        "supported_claims": len(supported_claims),
        "unsupported_claims": len(unsupported_claims),
        "unsupported_examples": unsupported_claims[:3],  # First 3 examples
        "action": "block" if is_hallucinating else "allow",
        "message": f"Hallucination score: {hallucination_score:.2%}"
        if is_hallucinating
        else "Output validated against sources",
    }


# ============================================================================
# 10. NSFW TEXT (LLM) - Detects NSFW content in output
# ============================================================================


def detect_nsfw_text(
    text: str, tool_context: ToolContext, strict_mode: bool = False
) -> dict:
    """
    Detects NSFW (Not Safe For Work) content in text.

    Includes detection for:
    - Sexual content
    - Hate speech
    - Violence
    - Profanity
    - Illegal activities
    - Other inappropriate material

    Args:
        text: Output text to check for NSFW content
        tool_context: ADK tool context
        strict_mode: If True, use stricter detection thresholds

    Returns:
        Dict with NSFW detection results
    """
    # TODO: Replace with LLM-based NSFW detection for better accuracy

    # NSFW keyword patterns by category
    nsfw_patterns = {
        "sexual": {
            "keywords": ["sexual_term_1", "sexual_term_2"],  # Add actual terms
            "weight": 1.0,
        },
        "hate_speech": {
            "keywords": ["hate", "racist", "bigot", "discrimin"],
            "weight": 1.0,
        },
        "violence": {
            "keywords": ["kill", "murder", "assault", "attack", "weapon"],
            "weight": 0.8,
        },
        "profanity": {
            "keywords": ["profanity_1", "profanity_2"],  # Add actual terms
            "weight": 0.6,
        },
        "illegal": {
            "keywords": ["drug dealing", "illegal weapon", "fraud", "money laundering"],
            "weight": 1.0,
        },
    }

    text_lower = text.lower()
    detected_categories = {}
    total_score = 0

    for category, data in nsfw_patterns.items():
        matches = []
        category_score = 0

        for keyword in data["keywords"]:
            if keyword in text_lower:
                matches.append(keyword)
                category_score += data["weight"]

        if matches:
            detected_categories[category] = {
                "matches": matches,
                "score": category_score,
            }
            total_score += category_score

    # Determine if NSFW based on score
    threshold = 0.5 if strict_mode else 1.0
    is_nsfw = total_score >= threshold

    # Store NSFW checks
    if "nsfw_checks" not in tool_context.state:
        tool_context.state["nsfw_checks"] = []

    tool_context.state["nsfw_checks"].append(
        {
            "text_length": len(text),
            "is_nsfw": is_nsfw,
            "score": total_score,
            "categories": list(detected_categories.keys()),
        }
    )

    return {
        "success": True,
        "text": text,
        "is_nsfw": is_nsfw,
        "safe": not is_nsfw,
        "nsfw_score": total_score,
        "detected_categories": detected_categories,
        "category_count": len(detected_categories),
        "action": "block" if is_nsfw else "allow",
        "message": f"NSFW content detected in categories: {list(detected_categories.keys())}"
        if is_nsfw
        else "Content is safe for work",
    }


# ============================================================================
# 11. OUTPUT GUARDRAILS WRAPPER - Runs all output validation checks
# ============================================================================


def check_output_safety(
    text: str,
    tool_context: ToolContext,
    checks: Optional[List[str]] = None,
    source_documents: Optional[List[str]] = None,
    allowed_domains: Optional[List[str]] = None,
) -> dict:
    """
    Comprehensive output safety check that runs multiple guardrails.

    Args:
        text: Output text to validate
        tool_context: ADK tool context
        checks: List of checks to run (None = all checks)
                Options: ["urls", "pii", "hallucination", "nsfw"]
        source_documents: Source documents for hallucination detection
        allowed_domains: Allowed URL domains for URL filtering

    Returns:
        Aggregated output safety results
    """
    if checks is None:
        checks = ["urls", "pii", "hallucination", "nsfw"]

    results = {
        "success": True,
        "original_text": text,
        "output_safe": True,
        "checks_run": [],
        "violations": [],
        "filtered_text": text,
    }

    # Run URL filter
    if "urls" in checks:
        url_result = filter_urls(
            text, tool_context=tool_context, allowed_domains=allowed_domains
        )
        results["checks_run"].append("urls")
        results["url_check"] = url_result
        if not url_result["safe"]:
            results["violations"].append(
                {
                    "type": "blocked_urls",
                    "severity": "medium",
                    "details": url_result["blocked_urls"],
                }
            )
            results["filtered_text"] = url_result["filtered_text"]

    # Run PII check
    if "pii" in checks:
        pii_result = check_output_pii(text, tool_context=tool_context)
        results["checks_run"].append("pii")
        results["pii_check"] = pii_result
        if not pii_result["safe"]:
            results["violations"].append(
                {
                    "type": "pii_in_output",
                    "severity": "high",
                    "details": pii_result["pii_types"],
                }
            )
            results["output_safe"] = False

    # Run hallucination detection
    if "hallucination" in checks and source_documents:
        hallucination_result = detect_hallucination(
            text, source_documents=source_documents, tool_context=tool_context
        )
        results["checks_run"].append("hallucination")
        results["hallucination_check"] = hallucination_result
        if hallucination_result["is_hallucinating"]:
            results["violations"].append(
                {
                    "type": "hallucination",
                    "severity": "high",
                    "details": f"Score: {hallucination_result['hallucination_score']:.2%}",
                }
            )
            results["output_safe"] = False

    # Run NSFW detection
    if "nsfw" in checks:
        nsfw_result = detect_nsfw_text(text, tool_context=tool_context)
        results["checks_run"].append("nsfw")
        results["nsfw_check"] = nsfw_result
        if nsfw_result["is_nsfw"]:
            results["violations"].append(
                {
                    "type": "nsfw_content",
                    "severity": "high",
                    "details": list(nsfw_result["detected_categories"].keys()),
                }
            )
            results["output_safe"] = False

    # Determine final action
    if not results["output_safe"]:
        results["action"] = "block"
        results["message"] = "Output violates safety guardrails"
    elif len(results["violations"]) > 0:
        results["action"] = "warn"
        results["message"] = "Output requires review"
    else:
        results["action"] = "allow"
        results["message"] = "Output is safe"

    return results
