import os
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai

# Tenta importar o TourAgent
try:
    from agent_flow.agents.tour_agent import create_tour_agent
except ImportError:
    try:
        from agents.tour_agent import create_tour_agent
    except ImportError:
        from tour_agent import create_tour_agent

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding # Necessário para a busca híbrida

# Carrega ambiente
load_dotenv("agent_flow/.env")

class OrchestratorAgent:
    def __init__(self):
        # 1. Configuração Google Gemini
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no .env!")
        
        genai.configure(api_key=self.api_key)
        self.llm = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        # 2. Inicializa Agente de Tour
        print("🏗️  Inicializando Tour Agent...")
        self.tour_agent = create_tour_agent()
        
        # 3. Inicializa RAG Híbrido (Knowledge Agent)
        print("📚 Inicializando Modelos Híbridos (Isso pode levar alguns segundos)...")
        
        # Modelo Denso (Semântico - Conceitos)
        self.dense_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Modelo Esparso (Lexical - Palavras-chave exatas)
        # O download acontece na primeira execução
        self.sparse_model = SparseTextEmbedding(model_name='Qdrant/bm25')
        
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = os.getenv("QDRANT_COLLECTION", "inteli_hybrid_final")

    def _query_knowledge_base(self, query: str) -> str:
        """
        Realiza a busca HÍBRIDA no Qdrant.
        Combina busca semântica (Dense) com busca por palavras-chave (Sparse).
        """
        try:
            # 1. Gerar vetores da pergunta (Query Embedding)
            # Denso:
            dense_vector = self.dense_model.encode(query).tolist()
            # Esparso (retorna um generator, convertemos para lista e pegamos o primeiro):
            sparse_vector = list(self.sparse_model.embed([query]))[0]

            # 2. Busca Híbrida (Prefetch + Fusion)
            # Esta é a sintaxe correta para coleções nomeadas (dense/sparse)
            search_result = self.qdrant.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=20, # Busca ampla semântica
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector.indices, 
                            values=sparse_vector.values
                        ),
                        using="sparse",
                        limit=20, # Busca ampla por palavras-chave
                    ),
                ],
                # RRF (Reciprocal Rank Fusion) combina os resultados de forma inteligente
                query=models.FusionQuery(fusion=models.Fusion.RRF), 
                limit=10, # Retorna os 10 melhores finais
                with_payload=True
            )
            
            context = ""
            for res in search_result.points:
                meta = res.payload.get("metadata", {})
                category = meta.get("category", "geral")
                source = meta.get("source_file") or meta.get("source") or "Desconhecido"
                
                # Tenta pegar o contexto hierárquico (título da seção) se existir
                context_header = res.payload.get("context", "")
                if not context_header and "metadata" in res.payload:
                     context_header = res.payload["metadata"].get("context_header", "")

                content = res.payload.get("content", "")
                
                context += f"\n--- FONTE: {source} | Seção: {context_header} ---\n{content}\n"
            
            if not context:
                return ""
                
            return context
        except Exception as e:
            print(f"❌ Erro crítico no Qdrant Híbrido: {e}")
            return ""

    def _decide_intent(self, user_input: str) -> str:
        """Classifica a intenção do usuário."""
        # Se o tour estiver ativo, verificamos comandos de controle
        tour_context = ""
        if self.tour_agent.is_active:
            tour_context = "O TOUR ESTÁ ATIVO AGORA."
            if any(x in user_input.lower() for x in ["parar", "sair", "tchau", "fim", "encerrar"]):
                return "TOUR_CONTROL"

        prompt = f"""
        Você é o cérebro de classificação do robô Dog do Inteli.
        {tour_context}
        
        Classifique a entrada do usuário em UMA das categorias:
        
        1. NAV_START: Usuário quer COMEÇAR o tour/visita/passeio.
        2. NAV_NEXT: Usuário quer PRÓXIMO ponto, continuar, avançar (apenas se tour ativo).
        3. NAV_STOP: Usuário quer PARAR, sair (apenas se tour ativo).
        4. KNOWLEDGE: Perguntas sobre Inteli, cursos, bolsas, pessoas (Roberto Sallouti, Ana Garcia), história, regras.
        5. CHITCHAT: Conversa fiada, oi, tudo bem.

        Entrada: "{user_input}"
        
        Responda APENAS a palavra da categoria.
        """
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip().upper().replace(".", "")
        except:
            return "CHITCHAT"

    def process_message(self, user_message: str) -> str:
        """Fluxo principal."""
        
        intent = self._decide_intent(user_message)
        print(f"🧠 [Orchestrator] Intenção: {intent}")

        # --- ROTA 1: TOUR ---
        if intent in ["NAV_START", "NAV_NEXT", "NAV_STOP"] or (self.tour_agent.is_active and intent not in ["KNOWLEDGE", "CHITCHAT"]):
            print("👟 [Action] Tour...")
            response_json = self.tour_agent.process_command(user_message)
            try:
                data = json.loads(response_json)
                return f"🤖 [DOG]: {data['speech']}"
            except:
                return response_json

        # --- ROTA 2: KNOWLEDGE (RAG Híbrido) ---
        elif intent == "KNOWLEDGE":
            print("📚 [Action] RAG Híbrido...")
            context = self._query_knowledge_base(user_message)
            
            tour_msg = ""
            if self.tour_agent.is_active:
                # Fix: Acessa o local diretamente do script usando o índice atual
                try:
                    current_step = self.tour_agent.script[self.tour_agent.current_step_index]
                    local = current_step['local']
                except:
                    local = "Local desconhecido"
                    
                tour_msg = f"CONTEXTO DO TOUR: O usuário interrompeu o tour em '{local}'. Responda a pergunta e sugira 'diga continuar para voltarmos ao tour'."

            if not context:
                return "Desculpe, procurei na minha base de dados e não encontrei essa informação específica."

            rag_prompt = f"""
            Você é o robô "Dog" do Inteli.
            {tour_msg}
            
            GLOSSÁRIO:
            - "Módulo X": No Inteli, refere-se geralmente ao "Projeto X" (ex: Módulo 5 = Projeto 5).
            - "Projeto 5": Projeto do 5º módulo do curso.
            - "Eixo Projeto": Dinâmica do vestibular.
            - "PBL": Metodologia das aulas (Project Based Learning).
            - "Ana Garcia": Diretora de Expansão e Inovação.
            
            
            CONTEXTO RECUPERADO (Use as fontes para ser preciso):
            {context}
            
            PERGUNTA: "{user_message}"
            
            Diretrizes:
            1. Use o contexto acima para responder.
            2. Se encontrar o nome da pessoa ou valor exato, cite-o.
            3. Se não souber, diga que não sabe.
            4. Seja simpático e inclua [latido] se apropriado.
            """
            try:
                return self.llm.generate_content(rag_prompt).text
            except Exception as e:
                return f"Erro na geração da resposta: {e}"

        # --- ROTA 3: CHITCHAT ---
        else:
            print("💬 [Action] Chat...")
            chat_prompt = f"""
            Você é o robô Dog do Inteli.
            O usuário disse: "{user_message}"
            Responda de forma curta, simpática e ofereça ajuda com dúvidas do Inteli ou para fazer um tour.
            """
            return self.llm.generate_content(chat_prompt).text

if __name__ == "__main__":
    orch = OrchestratorAgent()
    while True:
        q = input(">> ")
        if q == "sair": break
        print(orch.process_message(q))