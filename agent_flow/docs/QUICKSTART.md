# 🚀 Quick Start Guide - Inteli Robot Dog

Get your robot dog tour guide running in 5 minutes!

## Prerequisites

```bash
# Ensure Python 3.11+
python --version

# Ensure you have Google Cloud credentials
export GOOGLE_API_KEY="your-gemini-api-key"
```

## Installation

```bash
# 1. Navigate to the robot_dog_adk folder
cd robot_dog_adk

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "from google.adk.agents import Agent; print('✅ ADK installed!')"
```

## First Run - Demo Mode

```bash
# Run the demo conversation
python app.py --mode demo
```

Expected output:
```
🐕 Initializing Inteli Robot Dog Tour Guide...
✅ Agent created successfully!
✅ Runner initialized!

============================================================
🎬 DEMO: Inteli Robot Dog Tour Guide
============================================================

👤 João: Olá! Estou aqui para o tour do Inteli!

🐕 Robot Dog: [latido alegre] Olá! Bem-vindos ao Inteli!
   *balança o rabo* ...
```

## Interactive Mode

```bash
# Start interactive conversation
python app.py --mode interactive
```

Then chat naturally:
```
What's your name? Maria

🐕 Robot Dog: [latido] Olá, Maria! ...

👤 Maria: Como funciona o processo seletivo?
👤 Maria: Quais são os cursos?
👤 Maria: Vamos para a próxima parte do tour
```

## Example Conversations

### 1. Tour Flow

```python
# Visitor arrives
"Olá! Estou aqui para o tour"
→ Robot dog greets and starts tour

# During tour
"Vamos para a próxima parte?"
→ Advances to next section

# Questions
"Quanto custa?"
→ Explains scholarship program
```

### 2. Q&A Focus

```python
# Admission questions
"Como funciona o processo seletivo?"
→ Explains 3 axes (Prova, Perfil, Projeto)

# Course information
"Quais cursos vocês oferecem?"
→ Lists 5 courses with details

# Scholarships
"Tem bolsa?"
→ Explains comprehensive scholarship program
```

### 3. Engagement

```python
# Short responses
"ok"
→ Robot detects disengagement, re-engages with interesting fact

# Excited visitor
"Uau! Incrível!"
→ Matches energy level with excited response
```

## Troubleshooting

### Issue: Import errors

```bash
# Solution 1: Reinstall ADK
pip install google-adk --upgrade

# Solution 2: Check Python version
python --version  # Must be 3.11+

# Solution 3: Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Issue: Document chunks not found

```bash
# Run the preprocessing script first
cd ..
python main.py

# This creates:
# documents/Edital-...-chunks.json
```

### Issue: API authentication

```bash
# Set your API key
export GOOGLE_API_KEY="your-key-here"

# OR create .env file
echo "GOOGLE_API_KEY=your-key-here" > .env
```

## Next Steps

1. **Customize the tour script**: Edit `../documents/script.md`
2. **Add knowledge**: Update `agents/knowledge_agent.py`
3. **Adjust personality**: Modify `tools/personality_tools.py`
4. **Test different models**: `python app.py --model gemini-1.5-pro`

## Understanding the Output

### Conversation Format

```
👤 Visitor: [User input]
   ↓
   [Agent processing - you don't see this]
   ↓
🐕 Robot Dog: [Response with personality]
```

### Personality Elements

- `[latido]` = Dog bark
- `*balança o rabo*` = Physical action (wags tail)
- Emotions adapted to visitor's mood

### Agent Delegation (Behind the scenes)

When you see a response, the coordinator:
1. ✅ Checked safety (safety_agent)
2. 🎯 Detected emotion (personality_tools)
3. 🗺️ Or 🧠 Used tour_agent OR knowledge_agent
4. 🐕 Added personality

## Pro Tips

### 1. **Start with demo mode** to understand capabilities
```bash
python app.py --mode demo
```

### 2. **Test edge cases**
```bash
# Try questions the robot shouldn't know
"Qual é a capital da França?"
→ Should redirect to Inteli-related topics

# Try inappropriate content
"[something harmful]"
→ Should be blocked by safety_agent
```

### 3. **Monitor state** (for debugging)

The system tracks:
- Current tour section
- Visitor emotions
- Questions asked
- Personality statistics

### 4. **Extend knowledge**

Add to `knowledge_agent.py`:
```python
general_knowledge = {
    "new_topic": {
        "keywords": ["keyword1", "keyword2"],
        "content": "Information..."
    }
}
```

## Common Questions

**Q: Can I use a different model?**
A: Yes! `python app.py --model gemini-1.5-pro`

**Q: How do I add more tour sections?**
A: Edit `../documents/script.md` and update `tour_agent.py`

**Q: Can I deploy this to production?**
A: Yes! See README.md for deployment options

**Q: How accurate is the Q&A?**
A: Currently uses keyword matching. For production, implement vector embeddings.

**Q: Can I integrate with a physical robot?**
A: Yes! The app.py is designed to be extended with hardware APIs

## Ready to Go! 🐕

You now have a fully functional robot dog tour guide!

Try it out:
```bash
python app.py --mode interactive
```

For more details, see [README.md](README.md)

---

**Happy touring! [latido]** 🐕
