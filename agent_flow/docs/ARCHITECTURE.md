# 🏗️ Inteli Robot Dog - Architecture Deep Dive

## Overview

This document provides a detailed technical architecture of the Inteli Robot Dog Tour Guide system, optimized specifically for the Inteli campus tour use case.

## Design Principles

1. **Separation of Concerns**: Each agent handles a specific domain
2. **Personality First**: Consistent character maintained throughout
3. **Safety by Default**: All inputs validated before processing
4. **Context Awareness**: Emotion detection and adaptive responses
5. **Scalability**: Modular design allows easy extension

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                     (CLI / API / Physical Robot)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER (app.py)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         InMemoryRunner + InMemorySessionService          │  │
│  │         - Session management                              │  │
│  │         - Message routing                                 │  │
│  │         - State persistence                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                  ENHANCED COORDINATOR AGENT                    │
│               (enhanced_coordinator.py)                        │
│                                                                │
│  Role: Main robot dog personality & orchestrator               │
│  Model: gemini-2.0-flash-exp                                   │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Core Responsibilities:                                 │   │
│  │  • Maintain robot dog character                         │   │
│  │  • Orchestrate sub-agents                               │   │
│  │  • Emotion-aware responses                              │   │
│  │  • Conversation flow management                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  Sub-Agents:                   Tools:                          │
│  ├── Safety Agent               ├── add_dog_personality()      │
│  ├── Tour Agent                 ├── detect_visitor_emotion()   │
│  └── Knowledge Agent            ├── get_conversation_suggestions() │
│                                 └── generate_engagement_prompt()│
└─────┬───────────────┬───────────────┬───────────────────────────┘
      │               │               │
      ▼               ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Safety   │    │  Tour    │    │  Knowledge   │
│  Agent   │    │  Agent   │    │    Agent     │
└──────────┘    └──────────┘    └──────────────┘
```

---

## Agent Details

### 1. Enhanced Coordinator Agent

**File**: `agents/enhanced_coordinator.py`

**Purpose**: Main orchestrator that maintains the robot dog personality while coordinating all interactions.

**Key Features**:
- 🐕 Consistent robot dog character
- 🎯 Intelligent delegation to sub-agents
- ❤️ Emotion-aware response generation
- 🔄 Conversation flow management

**Decision Flow**:
```python
User Input
    ↓
1. Detect visitor emotion (detect_visitor_emotion)
    ↓
2. Check safety (→ safety_agent)
    ↓ [if safe]
3. Analyze intent
    ↓
    ├─→ Tour-related? → tour_agent
    │
    └─→ Question? → knowledge_agent
    ↓
4. Add personality (add_dog_personality)
    ↓
Response to User
```

**State Managed**:
```python
{
    "visitor_emotions": [],        # Last 10 emotions
    "personality_stats": {},        # Barks/actions count
    "tour_state": {},              # Current section, progress
    "message_count": 0,            # Total messages
    "retrieved_knowledge": []      # Last Q&A results
}
```

---

### 2. Safety Agent

**File**: `agents/safety_agent.py`

**Purpose**: Validates all user inputs for safety and appropriateness.

**Architecture**:
```
User Input
    ↓
check_content_safety()
    ↓
├─→ Keyword matching (unsafe patterns)
├─→ Context analysis
└─→ State logging
    ↓
Return: {"is_safe": bool, "reason": str}
```

**Unsafe Patterns** (examples):
- Violence keywords
- Harmful commands
- Inappropriate content

**Safety Violations Logged**:
```python
tool_context.state['safety_violations'] = [
    {
        "input": "user message",
        "violations": ["keyword1", "keyword2"]
    }
]
```

**Integration**:
- Coordinator **always** checks safety first
- If unsafe → polite redirect response
- All violations logged for monitoring

---

### 3. Tour Agent

**File**: `agents/tour_agent.py`

**Purpose**: Manages the campus tour script, progression, and tour-related interactions.

**Components**:

#### Tour Script Structure
```
documents/script.md
    ├── [História e Programa de Bolsas]
    ├── [Courses & Clubs]
    ├── [PBL & Rotina Inteli]
    ├── [Sala de aula invertida e infraestrutura]
    ├── [Processo Seletivo & Conquistas da Comunidade]
    └── [CONCLUSÃO]
```

#### Tools Provided

**1. `get_tour_section(section_name)`**
```python
# Retrieves specific tour section content
Result: {
    "success": True,
    "section": "historia",
    "content": "Que alegria receber vocês...",
    "marker": "História e Programa de Bolsas"
}
```

**2. `track_tour_progress(action)`**
```python
# Actions: "start", "next", "previous", "status"
# Tracks progression through tour sections

Result: {
    "success": True,
    "current_section": "cursos",
    "progress": "2/6"
}
```

**3. `get_tour_suggestions(context)`**
```python
# Analyzes context and suggests next actions
Result: {
    "suggestions": [
        {"type": "action", "suggestion": "advance_to_next_section"},
        {"type": "engagement", "suggestion": "ask_question"}
    ]
}
```

**Tour State**:
```python
{
    "tour_state": {
        "current_index": 2,                    # Currently at section 2
        "completed_sections": ["historia", "cursos"],
        "questions_asked": ["Como funciona?", "Tem bolsa?"]
    }
}
```

**Usage Pattern**:
```python
# Coordinator delegates to tour_agent when:
- Starting the tour
- Visitor says "vamos", "próximo", "continua"
- Need to know current tour position
- Need script content for current section
```

---

### 4. Knowledge Agent

**File**: `agents/knowledge_agent.py`

**Purpose**: RAG-powered Q&A system for Inteli information.

**Architecture**:

```
User Question
    ↓
Intent Analysis (by Coordinator)
    ↓
Knowledge Agent
    ↓
    ├─→ Specific topic? → get_specific_info()
    ├─→ General search? → search_inteli_knowledge()
    └─→ Complex question? → answer_question()
    ↓
Retrieval from:
    ├─→ General Knowledge Base (hardcoded topics)
    └─→ Document Chunks (Edital PDF)
    ↓
Rank by relevance
    ↓
Return top results + compiled answer
```

#### Knowledge Sources

**1. General Knowledge Base**
```python
{
    "inteli": "Fundado em 2019...",
    "cursos": "5 graduações...",
    "bolsas": "Maior programa do Brasil...",
    "pbl": "Metodologia baseada em projetos...",
    "clubes": "20+ clubes estudantis...",
    "conquistas": "Alunos premiados..."
}
```

**2. Document Chunks** (from Edital)
```python
# Loaded from:
documents/Edital-...-chunks.json

# Structure:
{
    "id": "chunk_42",
    "content": "O processo seletivo...",
    "metadata": {
        "section": "4.1 Processo Seletivo",
        "page_number": 12
    }
}
```

#### Tools Provided

**1. `search_inteli_knowledge(query, top_k=3)`**
```python
# Searches all knowledge for query
# Returns top K most relevant documents

Algorithm:
1. Search general knowledge (keyword matching)
2. Search document chunks (word overlap scoring)
3. Sort by relevance
4. Return top K

Result: {
    "documents": [
        {
            "source": "knowledge_base_bolsas",
            "content": "...",
            "relevance": 0.95
        }
    ]
}
```

**2. `get_specific_info(topic)`**
```python
# Fast retrieval for known topics
# Topics: processo_seletivo, bolsas, cursos, inteli_historia, conquistas

Result: {
    "title": "Programa de Bolsas",
    "summary": "Detailed info...",
    "related_topics": ["processo_seletivo"]
}
```

**3. `answer_question(question)`**
```python
# Comprehensive Q&A
# Uses search_inteli_knowledge internally
# Compiles answer from multiple sources

Result: {
    "answer": "Compiled answer...",
    "sources": [...],
    "confidence": 0.92
}
```

#### RAG Implementation

**Current (Simple)**:
```python
# Keyword-based search
query_words = set(query.lower().split())
chunk_words = set(chunk_text.split())
relevance = len(query_words & chunk_words) / len(query_words)
```

**Production Recommended**:
```python
# Vector embedding-based semantic search
from vertexai.preview.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("text-embedding-004")
query_embedding = model.get_embeddings([query])[0].values

# Use Cloud Spanner or Vertex AI Vector Search
# with COSINE_DISTANCE for ranking
```

---

## Personality System

**File**: `tools/personality_tools.py`

### Components

#### 1. Emotion Detection
```python
def detect_visitor_emotion(input) -> dict:
    """
    Analyzes visitor's emotional state from text.

    Emotions detected:
    - excited: "incrível", "demais", "uau"
    - happy: "feliz", "bom", "obrigado"
    - curious: "como", "por que", "?"
    - confused: "não entendi", "dúvida"
    - anxious: "preocupado", "medo"
    - bored: "ok", "tanto faz"
    - neutral: default
    """
```

**Algorithm**:
1. Keyword matching for each emotion
2. Score = (matches * emotion_weight)
3. Primary emotion = highest score
4. Confidence = normalized score

**Response Tone Mapping**:
```python
visitor_emotion → suggested_response_tone
excited         → excited      # Match their energy!
curious         → helpful      # Be informative
confused        → patient      # Be clear
anxious         → empathetic   # Be calming
bored           → playful      # Re-engage
```

#### 2. Personality Enhancement

```python
def add_dog_personality(text, emotion) -> dict:
    """
    Adds dog-like expressions to responses.

    Elements added:
    - Barks: [latido], [latido alegre], Au au!
    - Actions: *balança o rabo*, *pula animado*
    - Emotion-appropriate tone
    """
```

**Emotion-Based Expressions**:
```python
"happy": {
    "barks": ["[latido alegre]", "Au au!"],
    "actions": ["*balança o rabo*", "*pula animado*"]
}

"excited": {
    "barks": ["[latidos múltiplos]", "Au au au!"],
    "actions": ["*gira em círculos*", "*pula de alegria*"]
}

"empathetic": {
    "barks": ["[latido carinhoso]"],
    "actions": ["*coloca a pata no seu braço*", "*olha nos seus olhos*"]
}
```

**Application Strategy**:
- 80% chance to add bark (beginning or end)
- 40% chance to add action
- Randomly selected from emotion-appropriate lists

#### 3. Conversation Suggestions

```python
def get_conversation_suggestions(context) -> dict:
    """
    Provides smart suggestions for response style and actions.

    Analyzes:
    - Tour progression
    - Recent visitor emotions
    - Question count
    - Engagement level
    """
```

**Disengagement Detection**:
```python
recent_emotions = last 3 emotions
if all emotions in [neutral, bored]:
    warning = "Visitor may be losing interest"
    suggestion = "Ask engaging question or move to next section"
```

#### 4. Engagement Prompts

```python
def generate_engagement_prompt(situation) -> dict:
    """
    Generates prompts for different situations.

    Situations:
    - quiet_moment: Keep conversation going
    - between_sections: Transition smoothly
    - after_question: Confirm understanding
    - re_engagement: Bring back interest
    - closing: Friendly goodbye
    """
```

---

## Data Flow Examples

### Example 1: Tour Start

```
User: "Olá! Estou aqui para o tour"

Coordinator:
  1. detect_visitor_emotion("Olá! Estou aqui para o tour")
     → emotion: happy, confidence: 0.7

  2. safety_agent.check_content_safety(...)
     → is_safe: True

  3. Analyze intent: Starting tour
     → Delegate to tour_agent

  4. tour_agent.track_tour_progress("start")
     → current_section: "historia"

  5. tour_agent.get_tour_section("historia")
     → content: "Que alegria receber vocês..."

  6. add_dog_personality(response, "happy")
     → Adds: [latido alegre], *balança o rabo*

Response: "[latido alegre] Olá! Bem-vindos ao Inteli!
           *balança o rabo* Qual o nome de vocês?"
```

### Example 2: Question Answering

```
User: "Como funciona o processo seletivo?"

Coordinator:
  1. detect_visitor_emotion(...)
     → emotion: curious, confidence: 0.85

  2. safety_agent.check_content_safety(...)
     → is_safe: True

  3. Analyze intent: Question about admission
     → Delegate to knowledge_agent

  4. knowledge_agent.get_specific_info("processo_seletivo")
     → Returns: Detailed info about 3 eixos

  5. add_dog_personality(answer, "helpful")
     → Adds: [latido curioso], *inclina a cabeça*

Response: "[latido curioso] Ótima pergunta! O processo
           seletivo tem 3 eixos: Prova (Matemática e Lógica),
           Perfil (redações), e Projeto (dinâmica em grupo).
           *inclina a cabeça* Quer saber mais sobre algum deles?"
```

### Example 3: Re-engagement

```
User: "ok"

Coordinator:
  1. detect_visitor_emotion("ok")
     → emotion: bored, confidence: 0.6

  2. safety_agent.check_content_safety(...)
     → is_safe: True

  3. get_conversation_suggestions(...)
     → warning: disengagement detected
     → suggestion: re_engagement needed

  4. generate_engagement_prompt("re_engagement")
     → "Ei, vocês sabiam que alunos do Inteli já ganharam prêmios internacionais?"

  5. add_dog_personality(prompt, "excited")
     → Adds: [latido empolgado], *balança o rabo rapidamente*

Response: "*balança o rabo rapidamente* [latido empolgado]
           Ei, vocês sabiam que alunos do Inteli já ganharam
           prêmios internacionais? Uma aluna foi até pro CERN
           na Suíça! *pula animado* Querem saber mais?"
```

---

## State Management

### Session State Structure

```python
tool_context.state = {
    # Tour tracking
    "tour_state": {
        "current_index": 2,
        "completed_sections": ["historia", "cursos"],
        "questions_asked": ["Como funciona?", "Tem bolsa?"]
    },

    # Emotion tracking
    "visitor_emotions": [
        {"input": "...", "emotion": "happy", "confidence": 0.8},
        {"input": "...", "emotion": "curious", "confidence": 0.9},
        # ... last 10 emotions
    ],

    # Personality stats
    "personality_stats": {
        "barks_count": 15,
        "actions_count": 8,
        "emotions_used": {"happy": 5, "excited": 3, "curious": 7}
    },

    # Knowledge retrieval
    "retrieved_knowledge": [
        {"source": "...", "content": "...", "relevance": 0.95}
    ],
    "last_query": "Como funciona o processo seletivo?",
    "last_answer": {...},

    # Safety tracking
    "safety_violations": [],  # Logged violations

    # Misc
    "message_count": 12,
    "user_preferences": {...},
    "engagement_prompts_used": [...]
}
```

### State Persistence

**Current**: `InMemorySessionService` (ephemeral)
- State persists during session
- Lost when app restarts

**Production**: Persistent storage recommended
```python
# Option 1: Cloud Firestore
from google.cloud import firestore
db = firestore.Client()

# Option 2: Cloud Spanner
from google.cloud import spanner
```

---

## Performance Considerations

### Response Time Optimization

**Target**: Sub-2 second response time

**Strategies**:
1. **Fast model**: Use `gemini-2.0-flash-exp`
2. **Parallel tool calls**: ADK supports concurrent execution
3. **Caching**: Cache frequently accessed knowledge
4. **Lazy loading**: Load tour script sections on-demand

### Token Optimization

**Techniques**:
1. **Concise instructions**: Clear, brief agent instructions
2. **Selective context**: Only include relevant state in context
3. **Chunked documents**: Pre-process documents into optimal chunks
4. **Progressive disclosure**: Show information incrementally

---

## Scalability

### Horizontal Scaling

**For multiple simultaneous tours**:
```
Load Balancer
    ├─→ App Instance 1 (Session A, B, C)
    ├─→ App Instance 2 (Session D, E, F)
    └─→ App Instance 3 (Session G, H, I)
```

**Shared resources**:
- Document chunks (read-only, cacheable)
- Tour script (read-only, cacheable)
- LLM API (Google manages scaling)

### Vertical Scaling

**Optimization for single instance**:
- In-memory caching for hot documents
- Connection pooling for API calls
- Async/await for I/O operations

---

## Security & Safety

### Multi-Layer Security

```
Layer 7: Logging & Monitoring
    ↓
Layer 6: Safety violations tracking
    ↓
Layer 5: Content filtering (safety_agent)
    ↓
Layer 4: Input validation
    ↓
Layer 3: Authentication (future)
    ↓
Layer 2: API rate limiting (Google managed)
    ↓
Layer 1: Network security (HTTPS)
```

### Safety Agent Implementation

**Checks performed**:
1. ✅ Keyword-based harmful content detection
2. ✅ Context analysis for safety
3. ✅ Violation logging
4. ✅ State tracking

**Future enhancements**:
- LLM-based safety check (Gemini safety settings)
- PII detection and redaction
- Rate limiting per user
- Comprehensive audit logs

---

## Extension Points

### Adding New Agents

```python
# 1. Create new agent file
def create_my_agent(model: str) -> Agent:
    agent = Agent(
        name="my_agent",
        model=model,
        instruction="...",
        tools=[...]
    )
    return agent

# 2. Add to coordinator
coordinator = Agent(
    ...,
    sub_agents=[
        safety_agent,
        tour_agent,
        knowledge_agent,
        my_agent  # ← New agent
    ]
)
```

### Adding New Tools

```python
# 1. Create tool function
def my_tool(param: str, tool_context: ToolContext) -> dict:
    # Tool logic
    return {"result": "..."}

# 2. Add to agent's tools
agent = Agent(
    ...,
    tools=[existing_tools, my_tool]
)
```

### Adding New Knowledge

```python
# Option 1: Update general knowledge
general_knowledge["new_topic"] = {
    "keywords": [...],
    "content": "..."
}

# Option 2: Add documents to chunks
# Preprocess new documents with main.py
# Chunks automatically loaded
```

---

## Monitoring & Observability

### Key Metrics to Track

**Performance**:
- Response latency (target: < 2s)
- Tool execution time
- LLM token usage

**User Experience**:
- Tour completion rate
- Questions asked per tour
- Emotion distribution
- Re-engagement success rate

**Safety**:
- Safety violations count
- Safety check pass rate

**Agent Behavior**:
- Personality elements usage (barks, actions)
- Agent delegation frequency
- Knowledge retrieval success rate

### Logging Strategy

```python
# Application logs
logger.info(f"User {user_id} started tour")
logger.info(f"Question answered: {question}")

# Safety logs
logger.warning(f"Safety violation: {violation}")

# Performance logs
logger.info(f"Response time: {duration}ms")
```

---

## Conclusion

This architecture provides:
- ✅ **Modular design** for easy extension
- ✅ **Consistent personality** throughout interactions
- ✅ **Safety-first approach** with multi-layer validation
- ✅ **Intelligent Q&A** with RAG capabilities
- ✅ **Emotion awareness** for adaptive responses
- ✅ **Scalable structure** for production deployment

**Next steps**: See [README.md](README.md) for usage and [QUICKSTART.md](QUICKSTART.md) for getting started!

---

**Architecture Version**: 1.0
**Last Updated**: 2025-01
**Status**: Production-Ready 🐕
