# 🐕 Inteli Robot Dog Tour Guide

An intelligent, personality-driven robot dog tour guide for Inteli campus built with Google's Agent Development Kit (ADK).

## 🌟 Features

### Multi-Agent Architecture
```
┌──────────────────────────────────────────────┐
│     Enhanced Coordinator (Main Agent)        │
│  🐕 Friendly robot dog personality           │
│  🎯 Tour orchestration & conversation flow   │
└────────┬──────────┬──────────┬───────────────┘
         │          │          │
    ┌────┴────┐ ┌──┴───┐ ┌────┴─────┐
    │ Safety  │ │ Tour │ │Knowledge │
    │  Agent  │ │Agent │ │  Agent   │
    └─────────┘ └──────┘ └──────────┘
         │          │           │
    Content    Script      RAG-powered
    Validation  Manager    Q&A System
```

### Core Capabilities

1. **🗺️ Guided Campus Tours**
   - 5 structured tour sections
   - Natural tour script delivery
   - Progression tracking
   - Interactive Q&A during tour

2. **🧠 Intelligent Q&A (RAG-Powered)**
   - Semantic search over Inteli admission documents
   - Answers about: courses, scholarships, admission process, clubs, teaching methodology
   - Cites sources for transparency

3. **🎭 Consistent Robot Dog Personality**
   - Playful barks and actions [latido], *balança o rabo*
   - Emotion detection and adaptive responses
   - Engagement monitoring and re-engagement tactics
   - Maintains character throughout conversation

4. **🛡️ Safety & Content Moderation**
   - Input validation
   - Inappropriate content filtering
   - Family-friendly interactions

5. **❤️ Emotion-Aware Interactions**
   - Detects visitor emotions (excited, curious, bored, anxious, etc.)
   - Adapts response tone accordingly
   - Suggests engagement strategies

## 🏗️ Architecture Details

### Agents

#### 1. **Enhanced Coordinator** (`enhanced_coordinator.py`)
- **Role**: Main orchestrator and robot dog personality
- **Responsibilities**:
  - Manages conversation flow
  - Delegates to specialist agents
  - Maintains consistent character
  - Emotion-aware response generation

#### 2. **Safety Agent** (`safety_agent.py`)
- **Role**: Content validation
- **Responsibilities**:
  - Validates all user inputs
  - Blocks harmful content
  - Ensures family-friendly interactions

#### 3. **Tour Agent** (`tour_agent.py`)
- **Role**: Tour script manager
- **Responsibilities**:
  - Loads and manages tour script sections
  - Tracks tour progression
  - Suggests next tour actions
  - Monitors visitor engagement

**Tour Sections**:
1. História e Programa de Bolsas (History & Scholarships)
2. Courses & Clubs
3. PBL & Rotina Inteli (Teaching Methodology)
4. Sala de Aula Invertida (Flipped Classroom)
5. Processo Seletivo & Conquistas (Admission & Achievements)

#### 4. **Knowledge Agent** (`knowledge_agent.py`)
- **Role**: RAG-powered information retrieval
- **Responsibilities**:
  - Searches knowledge base (Edital document + general info)
  - Answers questions accurately
  - Provides structured topic information
  - Cites sources

**Knowledge Topics**:
- ✅ Admission process (3 evaluation axes)
- ✅ Scholarship programs
- ✅ 5 undergraduate courses
- ✅ 20+ student clubs
- ✅ PBL methodology
- ✅ Inteli history
- ✅ Student achievements

### Tools

#### Personality Tools (`personality_tools.py`)

1. **`add_dog_personality(text, emotion)`**
   - Enhances responses with dog-like expressions
   - Adds barks, tail wags, and actions
   - Emotion-appropriate embellishments

2. **`detect_visitor_emotion(input)`**
   - Analyzes visitor's emotional state
   - Returns emotion and confidence score
   - Suggests appropriate response tone

3. **`get_conversation_suggestions(context)`**
   - Provides smart response guidance
   - Detects disengagement
   - Suggests re-engagement tactics

4. **`generate_engagement_prompt(situation)`**
   - Creates engaging questions/prompts
   - Situations: greeting, between sections, re-engagement, closing

## 🚀 Setup & Installation

### Prerequisites

```bash
# Python 3.11+
python --version

# Google Cloud credentials (for ADK)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

# API keys (if using external services)
export GOOGLE_API_KEY="your-api-key"
```

### Installation

```bash
# Clone repository
cd robot_dog_adk

# Install dependencies
pip install -r requirements.txt

# Prepare document chunks (if not already done)
cd ..
python main.py  # This creates the chunks from Edital PDF
```

### Requirements

```txt
google-adk>=0.1.0
google-generativeai>=0.8.0
google-genai>=0.2.0
```

## 💻 Usage

### Interactive Mode (Recommended)

```bash
python app.py --mode interactive
```

**Example interaction**:
```
🐕 Welcome to Inteli Robot Dog Tour Guide!
What's your name? João

🐕 Robot Dog: [latido alegre] Olá! Bem-vindos ao Inteli!
   *balança o rabo* Qual o nome de vocês?

👤 João: João!

🐕 Robot Dog: [latido] João! Que alegria receber você aqui hoje!
   Vamos começar o tour? *pula animado*

👤 João: Como funciona o processo seletivo?

🐕 Robot Dog: [latido curioso] Ótima pergunta! O processo seletivo
   tem 3 eixos: Prova (Matemática e Lógica), Perfil (redações e
   atividades), e Projeto (dinâmica em grupo). *inclina a cabeça*
   Quer saber mais detalhes sobre algum deles?
```

### Demo Mode

```bash
python app.py --mode demo
```

Runs a pre-scripted demo conversation showcasing key features.

### Custom Model

```bash
python app.py --model gemini-1.5-pro
```

## 📊 Architecture Highlights

### 1. **State Management**

The system tracks:
- Tour progression (current section, completed sections)
- Visitor emotions (last 10 interactions)
- Questions asked
- Personality statistics (barks, actions used)
- Retrieved knowledge

### 2. **Retrieval-Augmented Generation (RAG)**

**Simple Implementation** (current):
- Keyword-based search over document chunks
- General knowledge base for common topics
- Relevance scoring

**Production Enhancement** (recommended):
- Vector embeddings (e.g., `text-embedding-004`)
- Semantic similarity search
- Cloud Spanner or Vertex AI Vector Search

### 3. **Conversation Flow**

```python
User Input
    ↓
Emotion Detection
    ↓
Safety Check (Safety Agent)
    ↓
Intent Analysis (Coordinator)
    ↓
┌──────────────┬───────────────┐
│ Tour-related │ Question      │
│              │               │
↓              ↓               ↓
Tour Agent     Knowledge Agent
    ↓              ↓
Retrieve       Search RAG +
Script         Answer Question
    ↓              ↓
└──────────────┴───────────────┘
    ↓
Add Personality
    ↓
Response to User
```

## 🎯 Use Cases

### 1. Campus Tours
- Automated campus orientation for prospective students
- Consistent tour experience
- Scalable to multiple simultaneous tours

### 2. Information Desk
- Answer common questions about admission
- Provide course information
- Explain scholarship programs

### 3. Engagement & Marketing
- Create memorable visitor experiences
- Showcase Inteli's innovation culture
- Social media content generation

## 🔧 Customization

### Adding New Tour Sections

Edit `documents/script.md`:
```markdown
[New Section Name]
Content for the new section...
```

Update `tour_agent.py`:
```python
sections = {
    "new_section": "New Section Name",
    # ... existing sections
}
```

### Enhancing Knowledge Base

Add to `knowledge_agent.py`:
```python
general_knowledge = {
    "new_topic": {
        "keywords": ["keyword1", "keyword2"],
        "content": "Information about new topic..."
    }
}
```

### Customizing Personality

Modify `personality_tools.py`:
```python
expressions = {
    "happy": {
        "barks": ["[custom bark]"],
        "actions": ["*custom action*"],
        # ...
    }
}
```

## 📈 Future Enhancements

### Recommended Improvements

1. **Vector Embeddings**
   - Implement semantic search using `text-embedding-004`
   - Use Cloud Spanner or Vertex AI Vector Search
   - Better relevance matching

2. **Multi-Modal Interaction**
   - Voice input/output (STT/TTS)
   - Physical robot integration
   - Display screen with visuals

3. **Advanced Analytics**
   - Track tour effectiveness
   - Common question analysis
   - Visitor engagement metrics

4. **Personalization**
   - Remember returning visitors
   - Tailor tours to interests
   - Progressive disclosure of information

5. **Multi-Language Support**
   - English tours
   - Auto-detect language preference

## 🐛 Troubleshooting

### Common Issues

**Issue**: `File not found: chunks.json`
```bash
# Solution: Generate chunks first
cd ..
python main.py
```

**Issue**: ADK import errors
```bash
# Solution: Ensure google-adk is installed
pip install google-adk --upgrade
```

**Issue**: Authentication errors
```bash
# Solution: Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
# OR
export GOOGLE_API_KEY="your-api-key"
```

## 📝 Development Notes

### Code Structure

```
robot_dog_adk/
├── agents/
│   ├── coordinator_agent.py      # Original coordinator
│   ├── enhanced_coordinator.py   # ⭐ Main coordinator
│   ├── safety_agent.py
│   ├── context_agent.py          # Original context agent
│   ├── tour_agent.py             # ⭐ Tour management
│   └── knowledge_agent.py        # ⭐ RAG Q&A
├── tools/
│   ├── safety_tools.py
│   ├── document_tools.py         # Original document tools
│   └── personality_tools.py      # ⭐ Personality & emotion
├── prompts/                       # (Optional) Prompt templates
├── app.py                         # ⭐ Main application
└── README.md                      # This file
```

### Best Practices

1. **Always use safety_agent first** - Validate all user input
2. **Cite sources** - When using knowledge_agent, mention source
3. **Monitor engagement** - Use emotion detection to adapt
4. **Stay in character** - Maintain dog personality throughout
5. **Track state** - Use tool_context.state for persistence

## 🤝 Contributing

Ideas for contributions:
- Add more tour sections
- Enhance emotion detection
- Improve RAG relevance
- Add voice integration
- Create visualization dashboard

## 📄 License

This project is part of the Inteli Computer Engineering program.

## 🙏 Acknowledgments

- Built by Computer Engineering students at Inteli
- Powered by Google ADK and Gemini
- Inspired by Inteli's innovative, hands-on learning culture

---

**Made with ❤️ (and lots of [latidos]!) by Inteli Computer Engineering students** 🐕
