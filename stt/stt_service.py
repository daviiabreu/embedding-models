import whisper
import logging
import os

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "base"
model = None

try:
    logging.info(f"Carregando modelo Whisper '{MODEL_NAME}'...")
    model = whisper.load_model(MODEL_NAME)
    logging.info(f"Modelo Whisper '{MODEL_NAME}' carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro fatal ao carregar o modelo Whisper: {e}")

def transcribe_audio(audio_filepath: str) -> str | None:
    """
    Transcreve um arquivo de áudio para texto usando o Whisper.
    """
    if model is None:
        logging.error("O modelo Whisper não está carregado. Impossível transcrever.")
        return None

    if not os.path.exists(audio_filepath):
        logging.error(f"Arquivo de áudio não encontrado em: {audio_filepath}")
        return None

    try:
        logging.info(f"Iniciando transcrição para: {os.path.basename(audio_filepath)}")

        result = model.transcribe(audio_filepath)
        transcribed_text = result["text"]
        detected_language = result["language"]

        logging.info(f"Transcrição concluída. Idioma detectado: {detected_language}")

        return transcribed_text.strip()

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a transcrição do áudio: {e}")
        return None

# --- Bloco de Teste ---
if __name__ == "__main__":
    print("Executando teste do módulo STT...")
    test_path = os.path.join("input_audio", "audio1.mp3")

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
