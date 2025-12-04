import logging
import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class LLMService:
    """Serviço para interação com o Gemini"""

    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        self.model = None

        # PRIMEIRO: Definir todas as configurações
        self.setup_configurations()

        # DEPOIS: Configurar o modelo
        self.setup_model()

    def setup_configurations(self):
        """Define todas as configurações antes de usar o modelo"""
        # Configuração de segurança MAIS permissiva
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Configuração de geração mais conservadora
        self.generation_config = {
            "temperature": 0.5,  # Reduzido para mais consistência
            "top_p": 0.9,
            "top_k": 20,
            "max_output_tokens": 500,  # Reduzido
        }

    def setup_model(self):
        """Configura o modelo tentando diferentes versões disponíveis"""
        models_to_try = ["gemini-2.0-flash"]

        for model_name in models_to_try:
            try:
                logging.info(f"🔄 Tentando modelo: {model_name}")
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name

                # Teste MUITO simples para evitar safety filters
                test_response = self.model.generate_content(
                    "Olá",
                    safety_settings=self.safety_settings,
                    generation_config={"max_output_tokens": 10},
                )

                # Verificar se há resposta válida
                if (
                    hasattr(test_response, "text")
                    and test_response.text
                    and test_response.text.strip()
                ):
                    logging.info(f"✅ Modelo {model_name} funcionando")
                    return
                elif hasattr(test_response, "candidates") and test_response.candidates:
                    # Verificar se foi bloqueado por safety
                    candidate = test_response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        logging.warning(
                            f"⚠️ Modelo {model_name} bloqueado - finish_reason: {candidate.finish_reason}"
                        )
                    else:
                        logging.warning(f"⚠️ Modelo {model_name} sem texto válido")
                else:
                    logging.warning(f"⚠️ Modelo {model_name} sem resposta")

            except Exception as e:
                logging.warning(f"⚠️ Modelo {model_name} falhou: {e}")
                continue

        logging.error("❌ Nenhum modelo Gemini funcionou")
        self.model = None

    def create_prompt(self, user_transcription: str, context: str = None) -> str:
        """Cria um prompt mais neutro para evitar safety filters"""

        # Prompt mais neutro e direto
        if context:
            prompt = f"""Contexto: {context}

Pergunta: {user_transcription}

Por favor, responda de forma amigável e direta, sem muito texto, em português brasileiro. A resposta deve ser sem links e considerando Instituto de Tecnologia e Liderança (Inteli) para se pesquisar sobre
Retorne apenas uma LISTA do que for solicitado na pergunta, sem explicações adicionais.

"""

        else:
            prompt = f"""Pergunta: {user_transcription}

Responda de forma amigável em português brasileiro."""

        return prompt

    def get_response(
        self, user_input: str, context: str = None, max_retries: int = 3
    ) -> Optional[str]:
        """Obtém resposta do Gemini com handling melhorado de safety filters"""

        if not self.model:
            logging.error("❌ Nenhum modelo disponível")
            return "Desculpe, o serviço de IA não está disponível no momento."

        # Criar prompt mais neutro
        prompt = self.create_prompt(user_input, context)

        for attempt in range(max_retries):
            try:
                logging.info(
                    f"🔄 Enviando para {self.model_name} (tentativa {attempt + 1}/{max_retries})..."
                )
                logging.debug(f"Prompt: {prompt[:100]}...")

                # Tentar diferentes configurações
                configs_to_try = [
                    # Configuração 1: Com safety settings
                    {
                        "safety_settings": self.safety_settings,
                        "generation_config": self.generation_config,
                    },
                    # Configuração 2: Apenas generation config
                    {"generation_config": self.generation_config},
                    # Configuração 3: Configuração mínima
                    {"generation_config": {"max_output_tokens": 300}},
                ]

                response = None
                for i, config in enumerate(configs_to_try):
                    try:
                        logging.debug(f"Tentando configuração {i + 1}...")
                        response = self.model.generate_content(prompt, **config)
                        break
                    except Exception as config_error:
                        logging.debug(f"Configuração {i + 1} falhou: {config_error}")
                        continue

                if not response:
                    raise Exception("Todas as configurações falharam")

                # Verificar se há texto na resposta
                if hasattr(response, "text") and response.text:
                    logging.info(f"✅ Resposta recebida: {response.text[:100]}...")
                    return response.text.strip()

                # Se não há texto, verificar o motivo
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        finish_reason = candidate.finish_reason
                        if finish_reason == 2:  # SAFETY
                            logging.warning("⚠️ Resposta bloqueada por safety filter")
                            # Tentar resposta alternativa mais genérica
                            if attempt < max_retries - 1:
                                prompt = (
                                    f"Como você responderia sobre: {user_input[:50]}..."
                                )
                                continue
                            else:
                                return "Desculpe, não posso responder a essa pergunta específica. Posso ajudar com outra coisa?"
                        elif finish_reason == 3:  # MAX_TOKENS
                            logging.warning("⚠️ Resposta truncada por limite de tokens")
                            return "Desculpe, minha resposta foi muito longa. Pode reformular sua pergunta?"
                        else:
                            logging.warning(f"⚠️ Finish reason: {finish_reason}")

                # Se chegou até aqui, não há resposta válida
                if attempt < max_retries - 1:
                    logging.info("🔄 Tentando com prompt mais simples...")
                    prompt = f"Responda brevemente: {user_input}"
                    continue
                else:
                    return "Desculpe, não consegui gerar uma resposta adequada."

            except Exception as e:
                logging.error(f"❌ Erro na tentativa {attempt + 1}: {e}")

                if attempt < max_retries - 1:
                    continue
                else:
                    return "Desculpe, ocorreu um erro ao processar sua mensagem."


# Instância global do serviço
llm_service = None


def initialize_llm_service():
    """Inicializa o serviço LLM de forma lazy"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service


def get_llm_response(user_input: str, context: str = None) -> Optional[str]:
    """Função utilitária para obter resposta da LLM"""
    service = initialize_llm_service()
    return service.get_response(user_input, context)


# Teste do módulo
if __name__ == "__main__":
    print("🧪 Testando LLM Service...")

    # Inicializar serviço
    service = initialize_llm_service()
    print(f"Modelo ativo: {service.model_name if service.model else 'Nenhum'}")

    # Teste básico
    if service.model:
        test_input = "Olá, como você está?"
        response = get_llm_response(test_input)

        if response:
            print("✅ Teste bem-sucedido!")
            print(f"Entrada: {test_input}")
            print(f"Resposta: {response}")
        else:
            print("❌ Teste falhou")
    else:
        print("❌ Nenhum modelo funcional encontrado")

    # Teste específico da pergunta problemática
    print("\n🧪 Testando pergunta específica...")
    problematic_input = "Quantos filhos tem o Mister Catra?"
    response = get_llm_response(problematic_input)
    print(f"Resposta para pergunta problemática: {response}")
