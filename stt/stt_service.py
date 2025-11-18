import whisper
import logging
import os
from typing import Optional

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "small"
model = None
context_prompt = "Esta é uma conversa amigável e informativa em português brasileiro durante um tour pelo INTELI (Instituto de Tecnologia e Liderança)"

try:
    logging.info(f"Carregando modelo Whisper '{MODEL_NAME}'...")
    model = whisper.load_model(MODEL_NAME)
    logging.info(f"Modelo Whisper '{MODEL_NAME}' carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro fatal ao carregar o modelo Whisper: {e}")

def transcribe_audio(
    audio_filepath: str,
    context_prompt: Optional[str] = None
) -> Optional[str]:
    """
    Transcreve um arquivo de áudio para texto usando o Whisper com otimizações.

    Args:
        audio_filepath: Caminho para o arquivo de áudio
        context_prompt: Prompt de contexto para melhorar a transcrição

    Returns:
        Texto transcrito ou None em caso de erro
    """
    if model is None:
        logging.error("O modelo Whisper não está carregado. Impossível transcrever.")
        return None

    if not os.path.exists(audio_filepath):
        logging.error(f"Arquivo de áudio não encontrado em: {audio_filepath}")
        return None

    try:
        logging.info(f"Iniciando transcrição para: {os.path.basename(audio_filepath)}")

        # Executar transcrição
        result = model.transcribe(audio_filepath, initial_prompt=context_prompt)

        transcribed_text = result["text"]
        detected_language = result.get("language", "unknown")

        logging.info(f"Transcrição concluída. Idioma detectado: {detected_language}")

        return transcribed_text.strip()

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a transcrição do áudio: {e}")
        return None

# --- Bloco de Teste ---
if __name__ == "__main__":
    print("Executando teste do módulo STT...")

    test_path = os.path.join("..", "input_audio", "audio1.ogg")

    if os.path.exists(test_path):
        text = transcribe_audio(test_path)
        if text:
            print("\n--- SUCESSO ---")
            print(f"Texto Transcrito: {text}")
        else:
            print("\n--- FALHA ---")
            print("Verifique os logs de erro.")
    else:
        print(f"\n--- AVISO ---")
        print(f"Arquivo de teste não encontrado em {test_path}")
