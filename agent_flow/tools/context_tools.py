import json
import os
from typing import Dict, List, Optional

import google.generativeai as genai
from google.adk.tools.tool_context import ToolContext

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ============================================================================
# 1. RETRIEVE RELEVANT CONTEXT - Get relevant information from knowledge base
# ============================================================================


def retrieve_relevant_context(
    query: str,
    tool_context: ToolContext,
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    sources: Optional[List[str]] = None,
) -> dict:
    try:
        from .knowledge_tools import retrieve_inteli_knowledge
    except ImportError:
        retrieved_chunks = []
        if "context_retrievals" not in tool_context.state:
            tool_context.state["context_retrievals"] = []

        tool_context.state["context_retrievals"].append(
            {"query": query, "top_k": top_k, "chunks_retrieved": 0}
        )

        return {
            "success": False,
            "query": query,
            "chunks": [],
            "total_retrieved": 0,
            "sources_searched": sources or ["all"],
            "message": "Knowledge tools not available",
            "error": "Could not import retrieve_inteli_knowledge",
        }

    try:
        rag_result = retrieve_inteli_knowledge(query, tool_context)

        retrieved_chunks = rag_result.get("chunks", [])

        if similarity_threshold > 0:
            filtered_chunks = [
                chunk
                for chunk in retrieved_chunks
                if chunk.get("score", 0) >= similarity_threshold
            ]
            retrieved_chunks = filtered_chunks

        retrieved_chunks = retrieved_chunks[:top_k]

        if "context_retrievals" not in tool_context.state:
            tool_context.state["context_retrievals"] = []

        tool_context.state["context_retrievals"].append(
            {
                "query": query,
                "top_k": top_k,
                "chunks_retrieved": len(retrieved_chunks),
                "similarity_threshold": similarity_threshold,
            }
        )

        return {
            "success": True,
            "query": query,
            "chunks": retrieved_chunks,
            "total_retrieved": len(retrieved_chunks),
            "sources_searched": sources or ["all"],
            "context": rag_result.get("context", ""),
            "query_embedding": rag_result.get("query_embedding"),
            "message": f"Retrieved {len(retrieved_chunks)} relevant context chunks from RAG system",
        }

    except Exception as e:
        if "context_retrievals" not in tool_context.state:
            tool_context.state["context_retrievals"] = []

        tool_context.state["context_retrievals"].append(
            {
                "query": query,
                "top_k": top_k,
                "chunks_retrieved": 0,
                "error": str(e),
            }
        )

        return {
            "success": False,
            "query": query,
            "chunks": [],
            "total_retrieved": 0,
            "sources_searched": sources or ["all"],
            "message": f"Context retrieval failed: {str(e)}",
            "error": str(e),
        }


# ============================================================================
# 2. RANK CONTEXT CHUNKS - Prioritize context by relevance
# ============================================================================


def rank_context_chunks(
    query: str,
    chunks: List[Dict],
    tool_context: ToolContext,
    ranking_method: str = "semantic",
) -> dict:
    ranked_chunks = []

    try:
        if ranking_method == "semantic":
            import os

            from sentence_transformers import SentenceTransformer, util

            model_name = os.getenv(
                "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            model = SentenceTransformer(model_name)

            query_embedding = model.encode(query, convert_to_tensor=True)

            scored_chunks = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if not content:
                    continue

                if "embedding" in chunk and chunk["embedding"]:
                    chunk_embedding = chunk["embedding"]
                else:
                    chunk_embedding = model.encode(content, convert_to_tensor=True)

                similarity = util.cos_sim(query_embedding, chunk_embedding).item()

                chunk_copy = chunk.copy()
                chunk_copy["ranking_score"] = similarity
                scored_chunks.append(chunk_copy)

            ranked_chunks = sorted(
                scored_chunks, key=lambda x: x.get("ranking_score", 0), reverse=True
            )

        elif ranking_method == "keyword":
            query_keywords = set(query.lower().split())

            scored_chunks = []
            for chunk in chunks:
                content = chunk.get("content", "").lower()
                content_keywords = set(content.split())

                overlap = len(query_keywords.intersection(content_keywords))
                total = len(query_keywords.union(content_keywords))
                score = overlap / total if total > 0 else 0

                chunk_copy = chunk.copy()
                chunk_copy["ranking_score"] = score
                scored_chunks.append(chunk_copy)

            ranked_chunks = sorted(
                scored_chunks, key=lambda x: x.get("ranking_score", 0), reverse=True
            )

        elif ranking_method == "hybrid":
            import os

            from sentence_transformers import SentenceTransformer, util

            model_name = os.getenv(
                "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            model = SentenceTransformer(model_name)
            query_embedding = model.encode(query, convert_to_tensor=True)
            query_keywords = set(query.lower().split())

            scored_chunks = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if not content:
                    continue

                if "embedding" in chunk and chunk["embedding"]:
                    chunk_embedding = chunk["embedding"]
                else:
                    chunk_embedding = model.encode(content, convert_to_tensor=True)
                semantic_score = util.cos_sim(query_embedding, chunk_embedding).item()

                content_keywords = set(content.lower().split())
                overlap = len(query_keywords.intersection(content_keywords))
                total = len(query_keywords.union(content_keywords))
                keyword_score = overlap / total if total > 0 else 0

                combined_score = 0.7 * semantic_score + 0.3 * keyword_score

                chunk_copy = chunk.copy()
                chunk_copy["ranking_score"] = combined_score
                chunk_copy["semantic_score"] = semantic_score
                chunk_copy["keyword_score"] = keyword_score
                scored_chunks.append(chunk_copy)

            ranked_chunks = sorted(
                scored_chunks, key=lambda x: x.get("ranking_score", 0), reverse=True
            )

        elif ranking_method == "recency":
            ranked_chunks = sorted(
                chunks,
                key=lambda x: x.get("metadata", {}).get("timestamp", 0),
                reverse=True,
            )

        elif ranking_method == "popularity":
            ranked_chunks = sorted(
                chunks,
                key=lambda x: x.get("metadata", {}).get("access_count", 0),
                reverse=True,
            )

        else:
            ranked_chunks = sorted(
                chunks, key=lambda x: x.get("score", 0), reverse=True
            )

    except Exception as e:
        ranked_chunks = chunks
        error_message = f"Ranking failed ({str(e)}), returning original order"
    else:
        error_message = None

    if "context_rankings" not in tool_context.state:
        tool_context.state["context_rankings"] = []

    tool_context.state["context_rankings"].append(
        {
            "ranking_method": ranking_method,
            "chunks_ranked": len(chunks),
            "error": error_message,
        }
    )

    return {
        "success": error_message is None,
        "ranked_chunks": ranked_chunks,
        "ranking_method": ranking_method,
        "total_ranked": len(ranked_chunks),
        "message": error_message
        or f"Ranked {len(ranked_chunks)} chunks using {ranking_method} method",
    }


# ============================================================================
# 3. FILTER CONTEXT BY RELEVANCE - Remove low-quality context
# ============================================================================


def filter_context_by_relevance(
    chunks: List[Dict],
    query: str,
    tool_context: ToolContext,
    min_score: float = 0.5,
    max_chunks: int = 10,
) -> dict:
    try:
        from datetime import datetime

        from sentence_transformers import SentenceTransformer, util

        if not chunks:
            return {
                "success": True,
                "filtered_chunks": [],
                "original_count": 0,
                "filtered_count": 0,
                "removed_count": 0,
                "message": "No chunks to filter",
            }

        score_filtered = [
            chunk for chunk in chunks if chunk.get("score", 0) >= min_score
        ]

        current_time = datetime.now()
        for chunk in score_filtered:
            metadata = chunk.get("metadata", {})
            freshness_score = 1.0
            timestamp = None
            for field in ["timestamp", "created_at", "updated_at", "last_modified"]:
                timestamp = metadata.get(field)
                if timestamp:
                    break
            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)):
                        chunk_date = datetime.fromtimestamp(timestamp)
                    elif isinstance(timestamp, str):
                        chunk_date = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                    else:
                        chunk_date = None
                    if chunk_date:
                        age_days = (current_time - chunk_date).days
                        freshness_score = max(0.1, 1.0 / (1.0 + age_days / 180.0))
                except (ValueError, TypeError, AttributeError):
                    pass
            chunk["freshness_score"] = freshness_score

        for chunk in score_filtered:
            relevance = chunk.get("score", 0)
            freshness = chunk.get("freshness_score", 1.0)
            chunk["combined_score"] = 0.7 * relevance + 0.3 * freshness

        score_filtered.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        if len(score_filtered) <= max_chunks:
            filtered_chunks = score_filtered
        else:
            model_name = os.getenv(
                "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            model = SentenceTransformer(model_name)
            query_embedding = model.encode(query, convert_to_tensor=True)
            chunk_embeddings = []
            for chunk in score_filtered:
                if "embedding" in chunk and chunk["embedding"]:
                    chunk_embeddings.append(chunk["embedding"])
                else:
                    embedding = model.encode(
                        chunk.get("content", ""), convert_to_tensor=True
                    )
                    chunk_embeddings.append(embedding)
            selected_indices = [0]
            remaining_indices = list(range(1, len(score_filtered)))
            lambda_param = 0.7
            while len(selected_indices) < max_chunks and remaining_indices:
                best_score = -float("inf")
                best_idx = None
                for idx in remaining_indices:
                    relevance = util.cos_sim(
                        query_embedding, chunk_embeddings[idx]
                    ).item()
                    max_similarity = max(
                        util.cos_sim(chunk_embeddings[idx], chunk_embeddings[s]).item()
                        for s in selected_indices
                    )
                    mmr_score = (
                        lambda_param * relevance - (1 - lambda_param) * max_similarity
                    )
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
            filtered_chunks = [score_filtered[idx] for idx in selected_indices]
        success = True
        error_message = None
    except Exception as e:
        filtered_chunks = [
            chunk for chunk in chunks if chunk.get("score", 0) >= min_score
        ][:max_chunks]
        success = False
        error_message = f"Advanced filtering failed ({str(e)}), used simple filtering"

    if "context_filtering" not in tool_context.state:
        tool_context.state["context_filtering"] = []

    tool_context.state["context_filtering"].append(
        {
            "original_count": len(chunks),
            "filtered_count": len(filtered_chunks),
            "min_score": min_score,
            "error": error_message,
        }
    )

    return {
        "success": success,
        "filtered_chunks": filtered_chunks,
        "original_count": len(chunks),
        "filtered_count": len(filtered_chunks),
        "removed_count": len(chunks) - len(filtered_chunks),
        "message": error_message
        or f"Filtered from {len(chunks)} to {len(filtered_chunks)} chunks (MMR + freshness)",
    }


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

# ============================================================================
# 4. MANAGE CONVERSATION MEMORY - Track conversation history
# ============================================================================


def manage_conversation_memory(
    current_message: str,
    tool_context: ToolContext,
    memory_type: str = "sliding_window",
    max_messages: int = 10,
) -> dict:
    if "conversation_history" not in tool_context.state:
        tool_context.state["conversation_history"] = []

    history = tool_context.state["conversation_history"]

    if memory_type in ["selective", "summary"]:
        try:
            if os.getenv("GOOGLE_API_KEY"):
                model = genai.GenerativeModel(
                    os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
                )

                importance_prompt = f"""Rate the importance of this message on a scale of 0.0 to 1.0 for conversation memory.

Message: "{current_message}"

Consider:
- Does it contain factual information?
- Does it represent a key decision or conclusion?
- Is it a greeting or filler?
- Does it introduce new topics?

Respond with ONLY a number between 0.0 and 1.0."""

                response = model.generate_content(importance_prompt)
                try:
                    importance = float(response.text.strip())
                    importance = max(0.0, min(1.0, importance))
                except ValueError:
                    importance = 0.5
            else:
                importance = 0.5
        except Exception:
            importance = 0.5
    else:
        importance = 0.5

    history.append(
        {"message": current_message, "timestamp": None, "importance": importance}
    )

    if memory_type == "sliding_window" and len(history) > max_messages:
        history = history[-max_messages:]

    elif memory_type == "summary" and len(history) > max_messages:
        try:
            if os.getenv("GOOGLE_API_KEY"):
                model = genai.GenerativeModel(
                    os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
                )

                older_messages = history[:-max_messages]
                recent_messages = history[-max_messages:]

                messages_text = "\n".join([msg["message"] for msg in older_messages])
                summary_prompt = f"""Summarize the following conversation messages concisely (max 200 words):

{messages_text}

Focus on key facts, decisions, and important points."""

                response = model.generate_content(summary_prompt)
                summary = response.text.strip()

                history = [
                    {
                        "message": f"[Summary of previous messages: {summary}]",
                        "timestamp": None,
                        "importance": 1.0,
                    }
                ] + recent_messages
            else:
                history = history[-max_messages:]
        except Exception:
            history = history[-max_messages:]

    elif memory_type == "selective" and len(history) > max_messages:
        sorted_history = sorted(
            history, key=lambda x: x.get("importance", 0), reverse=True
        )
        important_messages = sorted_history[:max_messages]
        history = sorted(important_messages, key=lambda x: history.index(x))

    tool_context.state["conversation_history"] = history

    return {
        "success": True,
        "memory_type": memory_type,
        "current_size": len(history),
        "max_size": max_messages,
        "recent_messages": history[-5:],
        "message": f"Managing {len(history)} messages in memory ({memory_type})",
    }


# ============================================================================
# 5. TRACK TOPICS DISCUSSED - Monitor conversation topics
# ============================================================================


def track_topics_discussed(
    conversation_history: List[str],
    tool_context: ToolContext,
    extract_subtopics: bool = True,
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
                for i, msg in enumerate(conversation_history[-10:])
            ]
        )

        subtopic_instruction = (
            "Also extract subtopics for each main topic." if extract_subtopics else ""
        )

        prompt = f"""Analyze the conversation history and extract all topics discussed.

Conversation History (last 10 messages):
{history_context}

Identify:
1. Main topics discussed
2. Subtopics (if requested)
3. Topic transitions (when the conversation shifted from one topic to another)
4. Coverage level for each topic (how deeply was it covered: superficial, moderate, deep)

{subtopic_instruction}

Respond in JSON format:
{{
    "main_topics": ["topic1", "topic2", "topic3"],
    "subtopics": {{"main_topic": ["subtopic1", "subtopic2"]}},
    "topic_transitions": [
        {{"from": "topic1", "to": "topic2", "message_index": 3}}
    ],
    "coverage": {{"topic1": "deep", "topic2": "moderate", "topic3": "superficial"}},
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of topic analysis"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        topics_discussed = {
            "main_topics": result.get("main_topics", []),
            "subtopics": result.get("subtopics", {}),
            "topic_transitions": result.get("topic_transitions", []),
            "coverage": result.get("coverage", {}),
        }

        if "topics_tracking" not in tool_context.state:
            tool_context.state["topics_tracking"] = []

        tool_context.state["topics_tracking"].append(
            {
                "topics": topics_discussed,
                "message_count": len(conversation_history),
                "confidence": result.get("confidence", 0.5),
            }
        )

        return {
            "success": True,
            "topics_discussed": topics_discussed,
            "total_topics": len(topics_discussed["main_topics"]),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", ""),
            "message": f"Tracked {len(topics_discussed['main_topics'])} main topics",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM topic extraction error: {str(e)}",
        }


# ============================================================================
# 6. DETECT CONTEXT GAPS - Identify missing information
# ============================================================================


def detect_context_gaps(
    query: str,
    tool_context: ToolContext,
    available_context: str = "",
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

        # Handle both string and list inputs
        if isinstance(available_context, str):
            context_summary = available_context
        elif isinstance(available_context, list):
            context_summary = "\n\n".join(
                [
                    f"Context {i + 1}: {ctx.get('text', str(ctx))[:200]}..."
                    for i, ctx in enumerate(available_context[:5])
                ]
            )
        else:
            context_summary = str(available_context)

        prompt = f"""Analyze if the available context is sufficient to answer the user's query.

User Query: "{query}"

Available Context:
{context_summary}

Identify:
1. What information is MISSING that would be needed to fully answer the query
2. What topics are INCOMPLETE (partially covered but need more detail)
3. What points are AMBIGUOUS or unclear
4. Your confidence level in the available context (0.0 = no confidence, 1.0 = fully confident)

Respond in JSON format:
{{
    "missing_information": ["missing info 1", "missing info 2"],
    "incomplete_topics": ["incomplete topic 1", "incomplete topic 2"],
    "ambiguous_points": ["ambiguous point 1"],
    "confidence_score": 0.0-1.0,
    "can_answer": true/false,
    "reasoning": "brief explanation of the gaps analysis"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        gaps_detected = {
            "missing_information": result.get("missing_information", []),
            "incomplete_topics": result.get("incomplete_topics", []),
            "ambiguous_points": result.get("ambiguous_points", []),
            "confidence_score": result.get("confidence_score", 0.0),
        }

        if "context_gaps" not in tool_context.state:
            tool_context.state["context_gaps"] = []

        tool_context.state["context_gaps"].append(
            {
                "query": query,
                "gaps": gaps_detected,
                "context_count": len(available_context)
                if isinstance(available_context, list)
                else 1,
            }
        )

        has_gaps = (
            len(gaps_detected["missing_information"]) > 0
            or len(gaps_detected["incomplete_topics"]) > 0
        )

        return {
            "success": True,
            "gaps": gaps_detected,
            "has_gaps": has_gaps,
            "confidence": gaps_detected["confidence_score"],
            "can_answer": result.get("can_answer", not has_gaps),
            "recommendation": "retrieve_more"
            if gaps_detected["confidence_score"] < 0.7
            else "proceed",
            "reasoning": result.get("reasoning", ""),
            "message": f"Gap analysis complete. Confidence: {gaps_detected['confidence_score']:.0%}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM gap detection error: {str(e)}",
        }


# ============================================================================
# CONTEXT ANALYSIS & SUMMARIZATION
# ============================================================================

# ============================================================================
# 7. SUMMARIZE CONTEXT - Condense context for efficiency
# ============================================================================


def summarize_context(
    context_chunks: List[Dict],
    tool_context: ToolContext,
    max_length: int = 500,
    summary_type: str = "abstractive",
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

        combined_text = "\n".join(
            [
                f"Chunk {i + 1}: {chunk.get('text', str(chunk))}"
                for i, chunk in enumerate(context_chunks[:10])
            ]
        )

        if summary_type == "extractive":
            approach_instruction = "Extract the most important sentences directly from the text without paraphrasing."
        elif summary_type == "abstractive":
            approach_instruction = (
                "Generate a new, concise summary that captures the key information."
            )
        else:  # hybrid
            approach_instruction = "Combine extracted key sentences with generated explanations to create a comprehensive summary."

        prompt = f"""Summarize the following context chunks into a concise summary.

Context to Summarize:
{combined_text[:3000]}

Summary Requirements:
- Maximum length: {max_length} words
- Type: {summary_type}
- {approach_instruction}
- Preserve key facts and important details
- Maintain logical flow

Respond in JSON format:
{{
    "summary": "the generated summary text",
    "key_points": ["key point 1", "key point 2", "key point 3"],
    "word_count": number,
    "compression_ratio": 0.0-1.0 (how much was compressed),
    "confidence": 0.0-1.0
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        summary = result.get("summary", "")

        if "context_summaries" not in tool_context.state:
            tool_context.state["context_summaries"] = []

        tool_context.state["context_summaries"].append(
            {
                "chunks_summarized": len(context_chunks),
                "summary_type": summary_type,
                "summary_length": len(summary.split()),
                "compression_ratio": result.get("compression_ratio", 0.0),
            }
        )

        return {
            "success": True,
            "summary": summary,
            "key_points": result.get("key_points", []),
            "original_chunks": len(context_chunks),
            "summary_type": summary_type,
            "word_count": result.get("word_count", len(summary.split())),
            "compression_ratio": result.get("compression_ratio", 0.0),
            "confidence": result.get("confidence", 0.5),
            "message": f"Summarized {len(context_chunks)} chunks into {result.get('word_count', 0)} words",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM summarization error: {str(e)}",
        }


# ============================================================================
# 8. EXTRACT KEY INFORMATION - Pull out important facts
# ============================================================================


def extract_key_information(
    context: str,
    query: str,
    tool_context: ToolContext,
    info_types: Optional[List[str]] = None,
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

        if info_types is None:
            info_types = ["facts", "dates", "numbers", "entities", "definitions"]

        info_types_str = ", ".join(info_types)

        prompt = f"""Extract key information from the context that is relevant to answering the query.

User Query: "{query}"

Context:
{context[:2000]}

Extract the following types of information:
{info_types_str}

- Facts: Factual statements or claims
- Dates: Temporal information (dates, times, periods)
- Numbers: Quantitative data (statistics, measurements, counts)
- Entities: People, places, organizations, products
- Definitions: Definitions of important terms

Respond in JSON format:
{{
    "facts": ["fact 1", "fact 2"],
    "dates": ["date 1", "date 2"],
    "numbers": ["number 1 (with context)", "number 2 (with context)"],
    "entities": ["entity 1", "entity 2"],
    "definitions": [{{"term": "term name", "definition": "definition text"}}],
    "confidence": 0.0-1.0
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        extracted_info = {
            "facts": result.get("facts", []),
            "dates": result.get("dates", []),
            "numbers": result.get("numbers", []),
            "entities": result.get("entities", []),
            "definitions": result.get("definitions", []),
        }

        if "info_extractions" not in tool_context.state:
            tool_context.state["info_extractions"] = []

        tool_context.state["info_extractions"].append(
            {
                "info_types": info_types,
                "total_extracted": sum(len(v) for v in extracted_info.values()),
                "confidence": result.get("confidence", 0.5),
            }
        )

        return {
            "success": True,
            "extracted_info": extracted_info,
            "info_types_requested": info_types,
            "total_items": sum(len(v) for v in extracted_info.values()),
            "confidence": result.get("confidence", 0.5),
            "message": f"Extracted {sum(len(v) for v in extracted_info.values())} items",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM information extraction error: {str(e)}",
        }


# ============================================================================
# 9. DEDUPLICATE CONTEXT - Remove redundant information
# ============================================================================


def deduplicate_context(
    chunks: List[Dict], tool_context: ToolContext, similarity_threshold: float = 0.9
) -> dict:
    try:
        import os

        from sentence_transformers import SentenceTransformer, util

        if not chunks:
            return {
                "success": True,
                "deduplicated_chunks": [],
                "original_count": 0,
                "unique_count": 0,
                "duplicates_removed": 0,
                "message": "No chunks to deduplicate",
            }

        model_name = os.getenv(
            "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        model = SentenceTransformer(model_name)

        seen_content = set()
        unique_chunks = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if content and content not in seen_content:
                seen_content.add(content)
                unique_chunks.append(chunk)

        exact_duplicates_removed = len(chunks) - len(unique_chunks)

        if len(unique_chunks) > 1:
            embeddings = []
            for chunk in unique_chunks:
                if "embedding" in chunk and chunk["embedding"]:
                    embeddings.append(chunk["embedding"])
                else:
                    content = chunk.get("content", "")
                    embedding = model.encode(content, convert_to_tensor=True)
                    embeddings.append(embedding)

            deduplicated_chunks = []
            skip_indices = set()

            for i, chunk_i in enumerate(unique_chunks):
                if i in skip_indices:
                    continue

                for j in range(i + 1, len(unique_chunks)):
                    if j in skip_indices:
                        continue

                    similarity = util.cos_sim(embeddings[i], embeddings[j]).item()

                    if similarity >= similarity_threshold:
                        skip_indices.add(j)

                deduplicated_chunks.append(chunk_i)

            semantic_duplicates_removed = len(unique_chunks) - len(deduplicated_chunks)
        else:
            deduplicated_chunks = unique_chunks
            semantic_duplicates_removed = 0

        total_removed = exact_duplicates_removed + semantic_duplicates_removed

    except Exception as e:
        seen_content = set()
        deduplicated_chunks = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if content and content not in seen_content:
                seen_content.add(content)
                deduplicated_chunks.append(chunk)

        total_removed = len(chunks) - len(deduplicated_chunks)
        error_message = (
            f"Semantic deduplication failed ({str(e)}), used exact matching only"
        )
    else:
        error_message = None

    if "context_deduplication" not in tool_context.state:
        tool_context.state["context_deduplication"] = []

    tool_context.state["context_deduplication"].append(
        {
            "original_count": len(chunks),
            "deduplicated_count": len(deduplicated_chunks),
            "removed_count": total_removed,
            "similarity_threshold": similarity_threshold,
            "error": error_message,
        }
    )

    return {
        "success": error_message is None,
        "deduplicated_chunks": deduplicated_chunks,
        "original_count": len(chunks),
        "unique_count": len(deduplicated_chunks),
        "duplicates_removed": total_removed,
        "message": error_message
        or f"Removed {total_removed} duplicates (exact: {exact_duplicates_removed}, semantic: {semantic_duplicates_removed})",
    }


# ============================================================================
# CONTEXT PREPARATION & FORMATTING
# ============================================================================

# ============================================================================
# 10. MANAGE CONTEXT WINDOW - Handle token limits
# ============================================================================


def manage_context_window(
    context_chunks: List[Dict],
    conversation_history: List[str],
    tool_context: ToolContext,
    max_tokens: int = 8000,
    priority: str = "recent",
) -> dict:
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        word_count = len(text.split())
        char_count = len(text)
        return int((word_count * 1.3 + char_count / 4) / 2)

    try:
        context_tokens = sum(
            estimate_tokens(chunk.get("content", "")) for chunk in context_chunks
        )
        history_tokens = sum(estimate_tokens(msg) for msg in conversation_history)
        total_tokens = context_tokens + history_tokens

        if total_tokens <= max_tokens:
            optimized_context = {
                "context_chunks": context_chunks,
                "conversation_history": conversation_history,
                "estimated_tokens": total_tokens,
            }
        else:
            if priority == "recent":
                kept_history = []
                history_budget = max_tokens // 3
                current_tokens = 0
                for msg in reversed(conversation_history):
                    msg_tokens = estimate_tokens(msg)
                    if current_tokens + msg_tokens <= history_budget:
                        kept_history.insert(0, msg)
                        current_tokens += msg_tokens
                    else:
                        break

                kept_chunks = []
                chunk_budget = max_tokens - current_tokens
                current_tokens = 0
                for chunk in reversed(context_chunks):
                    chunk_tokens = estimate_tokens(chunk.get("content", ""))
                    if current_tokens + chunk_tokens <= chunk_budget:
                        kept_chunks.insert(0, chunk)
                        current_tokens += chunk_tokens
                    else:
                        break

                optimized_context = {
                    "context_chunks": kept_chunks,
                    "conversation_history": kept_history,
                    "estimated_tokens": estimate_tokens(" ".join(kept_history))
                    + sum(estimate_tokens(c.get("content", "")) for c in kept_chunks),
                }

            elif priority == "relevant":
                sorted_chunks = sorted(
                    context_chunks, key=lambda x: x.get("score", 0), reverse=True
                )
                kept_chunks = []
                kept_history = conversation_history[-3:]
                history_tokens = sum(estimate_tokens(msg) for msg in kept_history)
                chunk_budget = max_tokens - history_tokens
                current_tokens = 0

                for chunk in sorted_chunks:
                    chunk_tokens = estimate_tokens(chunk.get("content", ""))
                    if current_tokens + chunk_tokens <= chunk_budget:
                        kept_chunks.append(chunk)
                        current_tokens += chunk_tokens
                    else:
                        break

                optimized_context = {
                    "context_chunks": kept_chunks,
                    "conversation_history": kept_history,
                    "estimated_tokens": history_tokens + current_tokens,
                }

            elif priority == "balanced":
                mid_point = len(context_chunks) // 2
                recent_chunks = context_chunks[mid_point:]
                relevant_chunks = sorted(
                    context_chunks[:mid_point],
                    key=lambda x: x.get("score", 0),
                    reverse=True,
                )
                mixed_chunks = recent_chunks + relevant_chunks

                kept_chunks = []
                kept_history = conversation_history[-5:]
                history_tokens = sum(estimate_tokens(msg) for msg in kept_history)
                chunk_budget = max_tokens - history_tokens
                current_tokens = 0

                for chunk in mixed_chunks:
                    chunk_tokens = estimate_tokens(chunk.get("content", ""))
                    if current_tokens + chunk_tokens <= chunk_budget:
                        kept_chunks.append(chunk)
                        current_tokens += chunk_tokens
                    else:
                        break

                optimized_context = {
                    "context_chunks": kept_chunks,
                    "conversation_history": kept_history,
                    "estimated_tokens": history_tokens + current_tokens,
                }

            else:
                important_chunks = [
                    c for c in context_chunks if c.get("metadata", {}).get("important")
                ]
                other_chunks = [
                    c
                    for c in context_chunks
                    if not c.get("metadata", {}).get("important")
                ]

                kept_chunks = []
                kept_history = conversation_history[-5:]
                history_tokens = sum(estimate_tokens(msg) for msg in kept_history)
                chunk_budget = max_tokens - history_tokens
                current_tokens = 0

                for chunk in important_chunks:
                    chunk_tokens = estimate_tokens(chunk.get("content", ""))
                    if current_tokens + chunk_tokens <= chunk_budget:
                        kept_chunks.append(chunk)
                        current_tokens += chunk_tokens

                for chunk in sorted(
                    other_chunks, key=lambda x: x.get("score", 0), reverse=True
                ):
                    chunk_tokens = estimate_tokens(chunk.get("content", ""))
                    if current_tokens + chunk_tokens <= chunk_budget:
                        kept_chunks.append(chunk)
                        current_tokens += chunk_tokens
                    else:
                        break

                optimized_context = {
                    "context_chunks": kept_chunks,
                    "conversation_history": kept_history,
                    "estimated_tokens": history_tokens + current_tokens,
                }

        utilization = (
            optimized_context["estimated_tokens"] / max_tokens if max_tokens > 0 else 0
        )
        success = True
        error_message = None

    except Exception as e:
        optimized_context = {
            "context_chunks": context_chunks[:5],
            "conversation_history": conversation_history[-5:],
            "estimated_tokens": 0,
        }
        utilization = 0.0
        success = False
        error_message = f"Token optimization failed ({str(e)}), used simple truncation"

    if "context_window_management" not in tool_context.state:
        tool_context.state["context_window_management"] = []

    tool_context.state["context_window_management"].append(
        {
            "max_tokens": max_tokens,
            "priority": priority,
            "chunks_included": len(optimized_context["context_chunks"]),
            "estimated_tokens": optimized_context["estimated_tokens"],
            "utilization": utilization,
            "error": error_message,
        }
    )

    return {
        "success": success,
        "optimized_context": optimized_context,
        "estimated_tokens": optimized_context["estimated_tokens"],
        "max_tokens": max_tokens,
        "utilization": utilization,
        "priority_used": priority,
        "message": error_message
        or f"Context optimized to {optimized_context['estimated_tokens']}/{max_tokens} tokens ({utilization:.1%} utilization)",
    }


# ============================================================================
# 11. PREPARE CONTEXT FOR LLM - Format for consumption
# ============================================================================


def prepare_context_for_llm(
    context_chunks: List[Dict],
    query: str,
    tool_context: ToolContext,
    format_style: str = "structured",
) -> dict:
    try:
        if not context_chunks:
            formatted_context = "No context available."
        elif format_style == "structured":
            sections = [f"# Relevant Context for: {query}\n"]
            for i, chunk in enumerate(context_chunks, 1):
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                citations = []
                if metadata.get("section"):
                    citations.append(f"Section: {metadata['section']}")
                if metadata.get("page_number"):
                    citations.append(f"Page: {metadata['page_number']}")
                if metadata.get("document"):
                    citations.append(f"Document: {metadata['document']}")
                citation_str = " | ".join(citations) if citations else f"Source {i}"
                sections.append(f"\n## Context {i}: {citation_str}\n\n{content}")
            formatted_context = "\n".join(sections)
        elif format_style == "conversational":
            if len(context_chunks) == 1:
                formatted_context = f"Based on the available information, {context_chunks[0].get('content', '')}"
            else:
                parts = ["Here's what I found:\n"]
                for chunk in context_chunks:
                    parts.append(f"\n{chunk.get('content', '')}")
                formatted_context = "".join(parts)
        elif format_style == "bullet_points":
            formatted_context = f"Key information about '{query}':\n\n"
            for i, chunk in enumerate(context_chunks, 1):
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                source = (
                    metadata.get("section") or metadata.get("document") or f"Source {i}"
                )
                sentences = [s.strip() for s in content.split(".") if s.strip()]
                for sentence in sentences[:3]:
                    formatted_context += f"• {sentence}. [{source}]\n"
        elif format_style == "qa_format":
            formatted_context = f"Q: {query}\n\nA: "
            answers = []
            for chunk in context_chunks:
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                source = metadata.get("section") or metadata.get("document") or "Source"
                answers.append(f"{content} (Source: {source})")
            formatted_context += " ".join(answers)
        elif format_style == "json":
            formatted_data = {
                "query": query,
                "context_chunks": [
                    {
                        "content": chunk.get("content", ""),
                        "metadata": chunk.get("metadata", {}),
                        "score": chunk.get("score"),
                        "id": chunk.get("id"),
                    }
                    for chunk in context_chunks
                ],
            }
            formatted_context = json.dumps(formatted_data, indent=2, ensure_ascii=False)
        elif format_style == "markdown":
            formatted_context = f"# Query: {query}\n\n---\n\n"
            for i, chunk in enumerate(context_chunks, 1):
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                formatted_context += f"### Source {i}\n\n{content}\n\n"
                if metadata:
                    formatted_context += "_Metadata: "
                    meta_parts = []
                    for key, value in metadata.items():
                        if value and key not in ["chunk_id", "embedding"]:
                            meta_parts.append(f"{key}: {value}")
                    formatted_context += ", ".join(meta_parts) + "_\n\n"
                formatted_context += "---\n\n"
        else:
            formatted_context = "\n\n".join(
                [chunk.get("content", "") for chunk in context_chunks]
            )
        includes_citations = format_style in [
            "structured",
            "bullet_points",
            "qa_format",
            "markdown",
        ]
        success = True
        error_message = None
    except Exception as e:
        formatted_context = "\n\n".join(
            [chunk.get("content", "") for chunk in context_chunks]
        )
        includes_citations = False
        success = False
        error_message = f"Formatting failed ({str(e)}), used simple concatenation"

    if "context_formatting" not in tool_context.state:
        tool_context.state["context_formatting"] = []

    tool_context.state["context_formatting"].append(
        {
            "format_style": format_style,
            "chunks_formatted": len(context_chunks),
            "error": error_message,
        }
    )

    return {
        "success": success,
        "formatted_context": formatted_context,
        "format_style": format_style,
        "chunks_included": len(context_chunks),
        "includes_citations": includes_citations,
        "message": error_message
        or f"Context formatted using {format_style} style with {len(context_chunks)} chunks",
    }


# ============================================================================
# 12. BUILD CONTEXT PROFILE - Track user's knowledge state
# ============================================================================


def build_context_profile(
    conversation_history: List[str],
    topics_discussed: List[str],
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

        history_context = "\n".join(
            [
                f"Message {i + 1}: {msg}"
                for i, msg in enumerate(conversation_history[-10:])
            ]
        )

        topics_str = (
            ", ".join(topics_discussed)
            if topics_discussed
            else "No specific topics yet"
        )

        prompt = f"""Build a comprehensive profile of the user's knowledge and context based on the conversation history.

Conversation History (last 10 messages):
{history_context}

Topics Discussed: {topics_str}

Analyze and create a profile including:
1. Topics covered and their knowledge level (beginner/intermediate/advanced per topic)
2. Questions the user has asked (what they're curious about)
3. Information already provided to the user (so we don't repeat)
4. Gaps in understanding (what they still need to learn)
5. Overall knowledge completeness (how complete is their understanding: 0.0-1.0)

Respond in JSON format:
{{
    "topics_covered": ["topic1", "topic2"],
    "knowledge_level": {{"topic1": "intermediate", "topic2": "beginner"}},
    "questions_asked": ["question 1", "question 2"],
    "information_provided": ["info 1", "info 2"],
    "knowledge_gaps": ["gap 1", "gap 2"],
    "knowledge_completeness": 0.0-1.0,
    "learning_progress": "slow/moderate/fast",
    "recommended_next_topics": ["topic to cover next"],
    "confidence": 0.0-1.0,
    "summary": "brief summary of user's knowledge state"
}}"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        result = json.loads(result_text)

        context_profile = {
            "topics_covered": result.get("topics_covered", topics_discussed),
            "knowledge_level": result.get("knowledge_level", {}),
            "questions_asked": result.get("questions_asked", []),
            "information_provided": result.get("information_provided", []),
            "knowledge_gaps": result.get("knowledge_gaps", []),
            "knowledge_completeness": result.get("knowledge_completeness", 0.0),
            "learning_progress": result.get("learning_progress", "moderate"),
            "recommended_next_topics": result.get("recommended_next_topics", []),
            "last_updated": None,
        }

        tool_context.state["context_profile"] = context_profile

        return {
            "success": True,
            "profile": context_profile,
            "topics_covered": len(context_profile["topics_covered"]),
            "knowledge_completeness": context_profile["knowledge_completeness"],
            "confidence": result.get("confidence", 0.5),
            "summary": result.get("summary", ""),
            "message": f"Profile built: {context_profile['knowledge_completeness']:.0%} complete",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM profile building error: {str(e)}",
        }


# ============================================================================
# 13. CHECK CONTEXT FRESHNESS - Verify context is up-to-date
# ============================================================================


def check_context_freshness(
    context_chunks: List[Dict], tool_context: ToolContext, max_age_days: int = 90
) -> dict:
    from datetime import datetime, timedelta

    try:
        freshness_assessment = {
            "fresh_chunks": [],
            "stale_chunks": [],
            "unknown_age": [],
            "overall_freshness": 0.0,
        }

        if not context_chunks:
            return {
                "success": True,
                "freshness": freshness_assessment,
                "fresh_count": 0,
                "stale_count": 0,
                "overall_score": 0.0,
                "recommendation": "no_context",
                "message": "No chunks to check",
            }

        current_time = datetime.now()
        max_age_delta = timedelta(days=max_age_days)

        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            chunk_age_days = None
            is_fresh = None

            timestamp = None
            for field in [
                "timestamp",
                "created_at",
                "updated_at",
                "last_modified",
                "date",
            ]:
                timestamp = metadata.get(field)
                if timestamp:
                    break

            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)):
                        chunk_date = datetime.fromtimestamp(timestamp)
                    elif isinstance(timestamp, str):
                        for fmt in [
                            "%Y-%m-%d",
                            "%Y-%m-%dT%H:%M:%S",
                            "%Y-%m-%d %H:%M:%S",
                            "%d/%m/%Y",
                            "%m/%d/%Y",
                        ]:
                            try:
                                chunk_date = datetime.strptime(timestamp, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            chunk_date = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                    else:
                        chunk_date = None

                    if chunk_date:
                        age_delta = current_time - chunk_date
                        chunk_age_days = age_delta.days

                        is_fresh = age_delta <= max_age_delta

                        chunk_with_age = chunk.copy()
                        chunk_with_age["age_days"] = chunk_age_days
                        chunk_with_age["age_category"] = (
                            "fresh" if is_fresh else "stale"
                        )

                        if is_fresh:
                            freshness_assessment["fresh_chunks"].append(chunk_with_age)
                        else:
                            freshness_assessment["stale_chunks"].append(chunk_with_age)
                    else:
                        freshness_assessment["unknown_age"].append(chunk)

                except (ValueError, TypeError, AttributeError):
                    freshness_assessment["unknown_age"].append(chunk)
            else:
                freshness_assessment["unknown_age"].append(chunk)

        total_chunks = len(context_chunks)
        fresh_count = len(freshness_assessment["fresh_chunks"])
        stale_count = len(freshness_assessment["stale_chunks"])
        unknown_count = len(freshness_assessment["unknown_age"])

        overall_freshness = (
            (fresh_count * 1.0 + unknown_count * 0.5) / total_chunks
            if total_chunks > 0
            else 0.0
        )
        freshness_assessment["overall_freshness"] = overall_freshness

        # Determine recommendation
        if overall_freshness >= 0.8:
            recommendation = "context_fresh"
        elif overall_freshness >= 0.5:
            recommendation = "mostly_fresh"
        elif overall_freshness >= 0.3:
            recommendation = "partially_stale"
        else:
            recommendation = "update_required"

        success = True
        error_message = None

    except Exception as e:
        freshness_assessment = {
            "fresh_chunks": [],
            "stale_chunks": [],
            "unknown_age": context_chunks,
            "overall_freshness": 0.5,
        }
        fresh_count = 0
        stale_count = 0
        unknown_count = len(context_chunks)
        overall_freshness = 0.5
        recommendation = "unknown"
        success = False
        error_message = f"Freshness check failed ({str(e)}), marked all as unknown"

    if "context_freshness_checks" not in tool_context.state:
        tool_context.state["context_freshness_checks"] = []

    tool_context.state["context_freshness_checks"].append(
        {
            "chunks_checked": len(context_chunks),
            "max_age_days": max_age_days,
            "fresh_count": fresh_count,
            "stale_count": stale_count,
            "unknown_count": unknown_count,
            "overall_freshness": overall_freshness,
            "error": error_message,
        }
    )

    return {
        "success": success,
        "freshness": freshness_assessment,
        "fresh_count": fresh_count,
        "stale_count": stale_count,
        "unknown_count": unknown_count,
        "overall_score": overall_freshness,
        "recommendation": recommendation,
        "message": error_message
        or f"Freshness check: {fresh_count} fresh, {stale_count} stale, {unknown_count} unknown (score: {overall_freshness:.1%})",
    }


# ============================================================================
# 14. CONTEXT MANAGEMENT WRAPPER - Orchestrate all context operations
# ============================================================================


def manage_context(
    query: str,
    conversation_history: List[str],
    tool_context: ToolContext,
    operations: Optional[List[str]] = None,
    max_tokens: int = 8000,
) -> dict:
    if operations is None:
        operations = ["retrieve", "rank", "filter", "deduplicate", "optimize", "format"]

    managed_context = {
        "query": query,
        "context_chunks": [],
        "conversation_memory": [],
        "formatted_context": "",
        "metadata": {},
    }

    if "retrieve" in operations:
        retrieval_result = retrieve_relevant_context(query, tool_context)
        managed_context["context_chunks"] = retrieval_result["chunks"]

    if "rank" in operations and managed_context["context_chunks"]:
        ranking_result = rank_context_chunks(
            query, managed_context["context_chunks"], tool_context
        )
        managed_context["context_chunks"] = ranking_result["ranked_chunks"]

    if "filter" in operations and managed_context["context_chunks"]:
        filter_result = filter_context_by_relevance(
            managed_context["context_chunks"], query, tool_context
        )
        managed_context["context_chunks"] = filter_result["filtered_chunks"]

    if "deduplicate" in operations and managed_context["context_chunks"]:
        dedup_result = deduplicate_context(
            managed_context["context_chunks"], tool_context
        )
        managed_context["context_chunks"] = dedup_result["deduplicated_chunks"]

    memory_result = manage_conversation_memory(query, tool_context)
    managed_context["conversation_memory"] = memory_result["recent_messages"]

    if "optimize" in operations:
        window_result = manage_context_window(
            managed_context["context_chunks"],
            conversation_history,
            tool_context,
            max_tokens,
        )
        managed_context["context_chunks"] = window_result["optimized_context"][
            "context_chunks"
        ]
        managed_context["conversation_memory"] = window_result["optimized_context"][
            "conversation_history"
        ]

    if "format" in operations:
        format_result = prepare_context_for_llm(
            managed_context["context_chunks"], query, tool_context
        )
        managed_context["formatted_context"] = format_result["formatted_context"]

    return {
        "success": True,
        "managed_context": managed_context,
        "operations_performed": operations,
        "total_chunks": len(managed_context["context_chunks"]),
        "memory_items": len(managed_context["conversation_memory"]),
        "ready_for_llm": "format" in operations,
        "message": f"Context managed with {len(operations)} operations",
    }
