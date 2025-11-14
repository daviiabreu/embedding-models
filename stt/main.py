import os
import logging
from stt_service import transcribe_audio

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuração de Caminhos ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AUDIO_INPUT_DIR = os.path.join(PROJECT_ROOT, 'input_audio')
TEXT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_transcribed_audio')

# Garante que as pastas existam
os.makedirs(AUDIO_INPUT_DIR, exist_ok=True)
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

def run_stt_pipeline(input_filename: str):
    """
    Orquestra a pipeline de STT: Transcreve um áudio e salva em .txt.
    """
    audio_file_path = os.path.join(AUDIO_INPUT_DIR, input_filename)

    transcribed_text = transcribe_audio(audio_file_path)

    if transcribed_text is None:
        logging.error(f"Falha na transcrição de {input_filename}. Pulando.")
        return

    base_filename = os.path.splitext(input_filename)[0]
    output_txt_filename = f"{base_filename}.txt"
    output_txt_path = os.path.join(TEXT_OUTPUT_DIR, output_txt_filename)

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcribed_text)
    except Exception as e:
        logging.error(f"Erro ao salvar arquivo .txt para {input_filename}: {e}")

if __name__ == "__main__":

    audio_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')

    try:
        all_files = os.listdir(AUDIO_INPUT_DIR)
        audio_files = [f for f in all_files if f.lower().endswith(audio_extensions)]

        if not audio_files:
            logging.warning(f"Nenhum arquivo de áudio encontrado em: {AUDIO_INPUT_DIR}")
        else:
            logging.info(f"Encontrados {len(audio_files)} arquivos de áudio. Iniciando processamento...")
            for audio_file in audio_files:
                run_stt_pipeline(audio_file)
            logging.info("Processamento de todos os arquivos concluído.")

    except FileNotFoundError:
        logging.error(f"ERRO: O diretório de entrada não foi encontrado em {AUDIO_INPUT_DIR}")
    except Exception as e:
        logging.error(f"Um erro inesperado ocorreu: {e}")
