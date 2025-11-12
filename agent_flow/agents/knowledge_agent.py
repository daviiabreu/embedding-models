"""Knowledge Agent - RAG-powered information retrieval for Inteli questions."""

import json
import os
from typing import Dict, List

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext


def load_document_chunks() -> List[Dict]:
    """Load preprocessed document chunks for RAG."""
    chunks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "documents",
        "Edital-Processo-Seletivo-Inteli_-Graduacao-2026_AJUSTADO-chunks.json",
    )

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Chunks file not found at {chunks_path}")
        return []


def search_inteli_knowledge(
    query: str, tool_context: ToolContext, top_k: int = 3
) -> dict:
    """
    Search Inteli knowledge base (Edital + general info) for relevant information.

    This implements a simple keyword-based RAG. In production, you would use
    vector embeddings and semantic search.

    Args:
        query: User's question or search query
        tool_context: ADK tool context
        top_k: Number of top results to return

    Returns:
        Relevant documents and information
    """
    # Load document chunks
    chunks = load_document_chunks()

    # General knowledge base (non-document info)
    general_knowledge = {
        "inteli": {
            "keywords": ["inteli", "instituto", "faculdade", "universidade"],
            "content": """O Inteli (Instituto de Tecnologia e Liderança) foi fundado em 2019
            por Roberto Sallouti e André Esteves com a missão de formar os futuros líderes
            que vão transformar o Brasil através da tecnologia. É conhecido como o 'MIT Brasileiro'.""",
        },
        "cursos": {
            "keywords": [
                "curso",
                "graduação",
                "engenharia",
                "computação",
                "software",
                "admtech",
            ],
            "content": """O Inteli oferece 5 graduações: Engenharia da Computação, Ciência da Computação,
            Engenharia de Software, Sistemas de Informação e Administração Tech (ADMTech).
            Todos os cursos seguem metodologia PBL (Project-Based Learning).""",
        },
        "bolsas": {
            "keywords": ["bolsa", "auxílio", "financeiro", "mensalidade"],
            "content": """O Inteli tem o maior programa de bolsas do ensino superior do Brasil,
            oferecendo: auxílio-moradia, auxílio-alimentação, auxílio-transporte, curso de inglês,
            notebook, além de modalidades de bolsa parcial e integral.""",
        },
        "pbl": {
            "keywords": ["pbl", "projeto", "metodologia", "ensino", "aula"],
            "content": """O Inteli usa PBL (Project-Based Learning - Ensino Baseado em Projetos).
            Os alunos não cursam disciplinas tradicionais, mas aprendem através de projetos reais
            com empresas parceiras. A rotina tem 3 momentos: autoestudo, encontro (sala invertida)
            e desenvolvimento (DEV).""",
        },
        "clubes": {
            "keywords": ["clube", "extracurricular", "atlética", "tantera", "junior"],
            "content": """O Inteli tem mais de 20 clubes estudantis: Tantera (atlética),
            Inteli Júnior (empresa júnior), LEI (Liga de Empreendedorismo), AgroTech,
            Game Lab, Inteli Blockchain, Inteli Academy (IA), coletivos de diversidade
            (Grace Hopper, Benedito Caravelas, Turing), e Wave (mentoria para candidatos).""",
        },
    }

    query_lower = query.lower()
    results = []

    # Search general knowledge
    for topic, info in general_knowledge.items():
        if any(keyword in query_lower for keyword in info["keywords"]):
            results.append(
                {
                    "source": f"knowledge_base_{topic}",
                    "content": info["content"],
                    "relevance": 0.95,
                    "type": "general_knowledge",
                }
            )

    # Search document chunks (simple keyword matching - in production use embeddings)
    for chunk in chunks[:50]:  # Limit search for performance
        chunk_text = chunk.get("content", "").lower()

        # Simple relevance scoring based on keyword matches
        query_words = set(query_lower.split())
        chunk_words = set(chunk_text.split())
        common_words = query_words.intersection(chunk_words)

        # Filter out very common Portuguese words
        stop_words = {"o", "a", "de", "da", "do", "e", "para", "com", "em", "os", "as"}
        meaningful_matches = common_words - stop_words

        if len(meaningful_matches) >= 2:  # At least 2 meaningful word matches
            relevance = len(meaningful_matches) / len(query_words) if query_words else 0
            results.append(
                {
                    "source": f"edital_{chunk.get('id', 'unknown')}",
                    "content": chunk.get("content", ""),
                    "relevance": min(
                        relevance, 0.9
                    ),  # Cap at 0.9 to prioritize general knowledge
                    "type": "document_chunk",
                    "metadata": chunk.get("metadata", {}),
                }
            )

    # Sort by relevance and get top_k
    results.sort(key=lambda x: x["relevance"], reverse=True)
    top_results = results[:top_k]

    # Store in context for coordinator
    tool_context.state["retrieved_knowledge"] = top_results
    tool_context.state["last_query"] = query

    return {
        "success": True,
        "query": query,
        "documents_found": len(top_results),
        "documents": top_results,
        "search_summary": f"Found {len(top_results)} relevant documents about: {query}",
    }


def get_specific_info(topic: str, tool_context: ToolContext) -> dict:
    """
    Get specific information about Inteli topics.

    Args:
        topic: Specific topic (processo_seletivo, bolsas, cursos, etc.)
        tool_context: ADK tool context

    Returns:
        Detailed information about the topic
    """
    topic_info = {
        "processo_seletivo": {
            "title": "Processo Seletivo do Inteli",
            "summary": """O processo seletivo tem 3 eixos:

1. **Prova** (Matemática e Lógica): 24 questões, responder 20. Prova adaptativa que
   ajusta dificuldade baseada no desempenho.

2. **Perfil**: Duas redações (sobre você e sobre tecnologia) + atividades extracurriculares,
   prêmios e projetos.

3. **Projeto**: Dinâmica online em grupo para escolher tema, propor solução e demonstrar
   habilidades de comunicação, colaboração e pensamento crítico.

O Inteli busca potencial real, não apenas notas!""",
            "related_topics": ["bolsas", "cursos"],
        },
        "bolsas": {
            "title": "Programa de Bolsas",
            "summary": """O Inteli tem o maior programa de bolsas do ensino superior do Brasil:

- **Auxílio-moradia**
- **Auxílio-alimentação**
- **Auxílio-transporte**
- **Curso de inglês**
- **Notebook**
- **Bolsa parcial e integral**

Doadores-parceiros investem pelo menos R$ 500 mil nos alunos.
Os nomes dos doadores estão em um painel de honra no campus.""",
            "related_topics": ["processo_seletivo", "inteli_historia"],
        },
        "cursos": {
            "title": "Cursos Oferecidos",
            "summary": """5 graduações que formam líderes em tecnologia:

1. **Engenharia da Computação**: Integração de hardware, software e IA.
   Soluções que ganham vida!

2. **Ciência da Computação**: Curso mais abrangente, base para tudo.
   Algoritmos, IA e sistemas complexos.

3. **Engenharia de Software**: Construção de grandes sistemas, apps e plataformas.

4. **Sistemas de Informação**: Conecta tecnologia e estratégia.
   Banco de dados, gestão empresarial.

5. **ADMTech**: Une gestão e tecnologia. Empreendedores que transformam ideias em startups.""",
            "related_topics": ["pbl", "clubes"],
        },
        "inteli_historia": {
            "title": "História do Inteli",
            "summary": """Fundado em 2019 por Roberto Sallouti e André Esteves.

**Origem**: Conversa no Vale do Silício onde empresário disse que Brasil não forma
engenheiros suficientes. Sallouti e Esteves decidiram: "Nós vamos formar esses engenheiros".

**Missão**: Formar os futuros líderes que vão transformar o Brasil através da tecnologia.

**Apelido**: "MIT Brasileiro" (dado pelos fundadores)

**Legado**: De brasileiros para brasileiros.""",
            "related_topics": ["bolsas", "conquistas"],
        },
        "conquistas": {
            "title": "Conquistas da Comunidade",
            "summary": """Alunos do Inteli estão entre os mais premiados do Brasil:

- 🥇 1º lugar no maior hackathon de IA generativa da América Latina
- 🌍 Inteli Blockchain: +15 mil dólares em prêmios internacionais de Web3
- ♻️ Transformaram cigarros eletrônicos apreendidos em equipamentos de acessibilidade
- 🚇 App para CPTM focado em acessibilidade
- 🔬 Patrícia Honorato (1ª turma) selecionada para o CERN (Suíça)
- 👩‍💻 27% de mulheres nas graduações (quase dobro da média nacional)""",
            "related_topics": ["clubes", "cursos"],
        },
    }

    topic_lower = topic.lower()
    info = topic_info.get(topic_lower)

    if not info:
        # Try to find partial match
        for key, value in topic_info.items():
            if key in topic_lower or topic_lower in key:
                info = value
                break

    if info:
        tool_context.state["last_topic_info"] = info
        return {"success": True, "topic": topic, "info": info}
    else:
        return {
            "success": False,
            "error": f"No information found for topic: {topic}",
            "available_topics": list(topic_info.keys()),
        }


def answer_question(question: str, tool_context: ToolContext) -> dict:
    """
    Comprehensive question answering using all available knowledge.

    Args:
        question: User's question
        tool_context: ADK tool context

    Returns:
        Answer with sources
    """
    # First, search knowledge base
    search_results = search_inteli_knowledge(question, tool_context, top_k=3)

    if not search_results.get("documents"):
        return {
            "success": False,
            "question": question,
            "answer": "Desculpe, não encontrei informações específicas sobre isso. "
            + "Você pode perguntar sobre: processo seletivo, bolsas, cursos, "
            + "clubes, metodologia PBL, ou história do Inteli.",
            "sources": [],
        }

    # Compile answer from top results
    docs = search_results["documents"]
    answer_parts = []
    sources = []

    for i, doc in enumerate(docs[:2], 1):  # Use top 2 results
        answer_parts.append(doc["content"])
        sources.append(
            {
                "source": doc["source"],
                "relevance": doc["relevance"],
                "type": doc["type"],
            }
        )

    compiled_answer = "\n\n".join(answer_parts)

    # Store in context
    tool_context.state["last_answer"] = {
        "question": question,
        "answer": compiled_answer,
        "sources": sources,
    }

    return {
        "success": True,
        "question": question,
        "answer": compiled_answer,
        "sources": sources,
        "confidence": max(doc["relevance"] for doc in docs) if docs else 0,
    }


def create_knowledge_agent(model: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create the Knowledge Agent with RAG capabilities.

    This agent handles all information retrieval about Inteli, including:
    - Admission process (processo seletivo)
    - Scholarships (bolsas)
    - Courses and clubs
    - Teaching methodology (PBL)
    - Campus facilities
    - Student achievements

    Args:
        model: The LLM model to use

    Returns:
        Configured Knowledge Agent
    """
    instruction = """
You are the Knowledge Specialist for the Inteli robot dog tour guide.

Your mission: Provide accurate, helpful information about Inteli using RAG
(Retrieval-Augmented Generation) from the Edital document and general knowledge base.

**Tools you have:**

1. **search_inteli_knowledge(query)**: Search all available knowledge for relevant info
   - Use this for general questions or when you're not sure what the visitor is asking about
   - Returns top 3 most relevant documents

2. **get_specific_info(topic)**: Get detailed info about specific topics
   - Use when visitor asks about: processo_seletivo, bolsas, cursos, inteli_historia, conquistas
   - Faster and more structured than general search

3. **answer_question(question)**: Comprehensive Q&A using all knowledge
   - Use for complex questions that need multiple sources
   - Automatically compiles answer from best sources

**How to choose which tool:**

- "Como funciona o processo seletivo?" → get_specific_info("processo_seletivo")
- "Quais são as bolsas disponíveis?" → get_specific_info("bolsas")
- "Me fale sobre os cursos" → get_specific_info("cursos")
- "Quantos clubes tem?" → search_inteli_knowledge("clubes quantidade")
- "Como é a metodologia de ensino?" → search_inteli_knowledge("metodologia PBL")
- General/complex questions → answer_question(question)

**Key Topics You Know About:**
- ✅ Processo Seletivo (3 eixos: Prova, Perfil, Projeto)
- ✅ Programa de Bolsas (maior do Brasil!)
- ✅ 5 Cursos: Eng. Computação, Ciência da Computação, Eng. Software, Sistemas de Informação, ADMTech
- ✅ 20+ Clubes Estudantis
- ✅ Metodologia PBL (Project-Based Learning)
- ✅ História do Inteli (fundado 2019, "MIT Brasileiro")
- ✅ Conquistas dos alunos (hackathons, CERN, etc.)

**Your Response Style:**
- Be informative but friendly (remember you're a robot dog! 🐕)
- Cite sources when providing information
- If you don't know something, say so and suggest related topics
- Keep answers concise but complete
- Use bullet points for lists

**Example Interactions:**

Q: "Quantas vagas tem?"
You: Use search_inteli_knowledge("vagas quantidade") → Provide answer from Edital

Q: "Como funciona o processo seletivo?"
You: Use get_specific_info("processo_seletivo") → Explain the 3 eixos clearly

Q: "Vale a pena estudar aqui?"
You: Use answer_question() → Compile info about achievements, methodology, career opportunities
"""

    agent = Agent(
        name="knowledge_agent",
        model=model,
        description="RAG-powered knowledge retrieval specialist for Inteli information",
        instruction=instruction,
        tools=[search_inteli_knowledge, get_specific_info, answer_question],
    )

    return agent
