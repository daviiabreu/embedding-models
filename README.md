# 🐕 Inteli Robot Dog Tour Guide

> Sistema de agentes AI usando Google ADK para guiar visitantes pelo campus do Inteli com personalidade de cachorro-robô

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-1.16%2B-green.svg)](https://github.com/google/adk-toolkit)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)]()

---

## 🚀 Quick Start

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Testar instalação
python test_agent_flow.py

# 3. Executar modo demo
python run_app.py --mode demo

# 4. Executar modo interativo
python run_app.py --mode interactive
```

---

## 📖 Sobre o Projeto

Sistema multi-agente com **Google ADK** e **RAG** para tours do Inteli com personalidade de cachorro-robô:

- 🎯 Tours personalizados do campus
- 💬 Q&A sobre processo seletivo, cursos, bolsas
- 🛡️ Validação de segurança multi-layer
- 🐕 Personalidade canina consistente

### Arquitetura:

```
Coordinator Agent (Robot Dog)
├── Safety Agent → Valida conteúdo
├── Tour Agent → Gerencia roteiro  
└── Knowledge Agent → RAG para Q&A
```

---

## 📂 Estrutura

```
embedding-models/
├── run_app.py              # 🎯 Entry point
├── test_agent_flow.py      # 🧪 Testes
├── ROADMAP_SPRINT.md       # 📋 Roadmap 10 dias
│
├── agent_flow/             # 📦 Package principal
│   ├── agents/             # 🤖 Coordinator, Safety, Tour, Knowledge
│   ├── tools/              # 🛠️ Personality, Safety, Document tools
│   ├── prompts/            # 📝 Guidelines (2.7k+ palavras)
│   └── docs/               # 📚 Documentação
│
└── documents/              # 📄 Script + chunks RAG
```

---

## 🎯 Funcionalidades

### ✅ Implementado:
- [x] Arquitetura multi-agente (Google ADK)
- [x] RAG básico (keyword-based)
- [x] Personality tools (emoção, barks)
- [x] Safety validation
- [x] Tour script management
- [x] Guidelines completas (~2.7k palavras)

### 🚧 Em Desenvolvimento:
- [ ] RAG semântico (embeddings)
- [ ] Safety robusto (100+ keywords)
- [ ] Testes unitários (20+)
- [ ] Logging estruturado
- [ ] Dashboard visualização

---

## 📊 Exemplo de Conversa

```
👤 Você: Como funciona o processo seletivo?

🤖 Processing:
  1. Safety Agent → ✅ Conteúdo seguro
  2. Emotion Detection → 😊 Curioso
  3. Knowledge Agent → 🔍 Busca "processo_seletivo"
  4. Add Personality → 🐕 Latidos + ações

🐕 Robot Dog: [latido curioso] Ótima pergunta! *inclina a cabeça*
   O processo tem 3 eixos: Prova, Perfil e Projeto.
   
   1. **Prova**: 24 questões de matemática e lógica...
   2. **Perfil**: Redações sobre você e tecnologia...
   3. **Projeto**: Dinâmica em grupo online...
   
   *balança o rabo* Quer saber mais detalhes?
```

---

## 🔧 Configuração

### `.env` (criar na raiz):
```env
GOOGLE_API_KEY=sua_chave_aqui
DEFAULT_MODEL=gemini-2.0-flash-exp
```

### Modelos suportados:
- `gemini-2.0-flash-exp` (recomendado - rápido)
- `gemini-1.5-pro` (mais capaz)
- `gemini-1.5-flash` (muito rápido)

---

## 📋 Roadmap Sprint (10 dias)

Ver **[ROADMAP_SPRINT.md](./ROADMAP_SPRINT.md)** para plano completo.

### Prioridades:
1. **RAG Semântico** ⭐⭐⭐ (embeddings + FAISS)
2. **Safety Robusto** ⭐⭐⭐ (100+ keywords, LLM)
3. **Testes** ⭐⭐ (unit + integration)
4. **Logging** ⭐⭐ (structured logs)

---

## 🧪 Testes

```bash
# Validação completa
python test_agent_flow.py

# Deve passar 6 testes:
# ✅ Environment variables
# ✅ Agent imports
# ✅ Tool imports  
# ✅ Required files
# ✅ Agent creation
# ✅ Personality tools
```

---

## 📚 Documentação Completa

- **[ROADMAP_SPRINT.md](./ROADMAP_SPRINT.md)** - Plano detalhado 10 dias
- **[agent_flow/prompts/base_personality.txt](./agent_flow/prompts/base_personality.txt)** - Guidelines personalidade
- **[agent_flow/prompts/safety_guidelines.txt](./agent_flow/prompts/safety_guidelines.txt)** - Diretrizes segurança
- **[agent_flow/docs/](./agent_flow/docs/)** - Arquitetura e diagramas

---

## ⚠️ Limitações Conhecidas

1. **RAG keyword-based** (não semântico) → Prioridade #1
2. **Safety simples** (10 keywords) → Prioridade #2  
3. **Sem testes unitários** → Prioridade #3
4. **Sem logging estruturado** → Prioridade #4

---

## 🤝 Contribuindo

Prioridades atuais em **[ROADMAP_SPRINT.md](./ROADMAP_SPRINT.md)**

```bash
# 1. Fork e branch
git checkout -b feature/minha-feature

# 2. Desenvolver + testar
python test_agent_flow.py

# 3. Commit e PR
git commit -m "feat: adiciona X"
```

---

## 📄 Recursos

- [Google ADK Docs](https://google.github.io/adk-toolkit/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) (Agent reasoning)
- [RAG Paper](https://arxiv.org/abs/2005.11401) (Retrieval-Augmented Generation)

---

**Status:** Em desenvolvimento ativo 🚀  
**Última atualização:** 7 de novembro de 2025  
**Branch:** feat/adk-agent