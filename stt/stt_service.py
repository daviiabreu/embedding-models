import whisper
import logging
import os
import sys
import torch
from typing import Optional
from pathlib import Path

# Adicionar diretório pai ao path para importar utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar funções utilitárias compartilhadas
from utils.gpu_utils import (
    setup_device,
    get_gpu_info,
    clear_gpu_cache,
    validate_file_exists,
    log_gpu_memory,
    PerformanceMonitor
)

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "small"
model = None
device = None
context_prompt = "Esta é uma conversa amigável e informativa em português brasileiro durante um tour pelo INTELI (Instituto de Tecnologia e Liderança)"

# Configurar dispositivo antes de carregar o modelo
device = setup_device()

try:
    logging.info(f"Carregando modelo Whisper '{MODEL_NAME}' no {device.upper()}...")
    model = whisper.load_model(MODEL_NAME, device=device)
    logging.info(f"Modelo Whisper '{MODEL_NAME}' carregado com sucesso no {device.upper()}.")
        
except Exception as e:
    logging.error(f"Erro fatal ao carregar o modelo Whisper: {e}")

def transcribe_audio(
    audio_filepath: str,
    context_prompt: Optional[str] = None,
    fp16: bool = True
) -> Optional[str]:
    """
    Transcreve um arquivo de áudio para texto usando o Whisper com otimizações GPU.

    Args:
        audio_filepath: Caminho para o arquivo de áudio
        context_prompt: Prompt de contexto para melhorar a transcrição
        fp16: Usar FP16 (half precision) para acelerar na GPU (padrão: True)

    Returns:
        Texto transcrito ou None em caso de erro
    """
    if model is None:
        logging.error("O modelo Whisper não está carregado. Impossível transcrever.")
        return None

    # Validar se arquivo existe usando função reutilizável
    if not validate_file_exists(audio_filepath, "arquivo de áudio"):
        return None

    try:
        logging.info(f"Iniciando transcrição no {device.upper()}: {os.path.basename(audio_filepath)}")

        # Iniciar monitoramento de performance
        monitor = PerformanceMonitor(device)
        monitor.start()

        # Executar transcrição com otimizações
        # fp16 (half precision) acelera significativamente na GPU
        result = model.transcribe(
            audio_filepath,
            initial_prompt=context_prompt,
            fp16=(fp16 and device == "cuda"),  # Usar FP16 apenas se tiver GPU
            language="pt",  # Forçar português para melhor performance
            verbose=False  # Reduzir logs durante transcrição
        )

        transcribed_text = result["text"]
        detected_language = result.get("language", "unknown")

        # Mostrar estatísticas de performance
        stats = monitor.stop()
        
        logging.info(f"Transcrição concluída. Idioma: {detected_language}")
        logging.info(f"Tempo: {stats.get('elapsed_time_formatted', 'N/A')}")

        return transcribed_text.strip()

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a transcrição do áudio: {e}")
        return None

# --- Bloco de Teste ---
if __name__ == "__main__":
    from utils.gpu_utils import print_system_info, format_file_size
    
    print("STT Service - Teste com GPU")
    
    # Mostrar informações do sistema
    print_system_info(device)
    
    # Mostrar informações do dispositivo
    gpu_info = get_gpu_info(device)
    print("\nStatus atual:")
    for key, value in gpu_info.items():
        print(f"   • {key}: {value}")

    test_path = os.path.join("..", "input_audio", "audio1.ogg")

    if os.path.exists(test_path):
        file_size = os.path.getsize(test_path)
        print(f"\nTestando transcrição...")
        print(f"Arquivo: {test_path}")
        print(f"Tamanho: {format_file_size(file_size)}")
        
        text = transcribe_audio(test_path)
        
        if text:
            print("SUCESSO")
            print(f"Texto Transcrito: {text}")
            print(f"Palavras: {len(text.split())}")
                
            # Limpar cache
            clear_gpu_cache(device)
        else:
            print("FALHA")
    else:
        print("AVISO")
        print(f"Arquivo de teste não encontrado: {test_path}")
        print("\nDica: Coloque um arquivo de áudio em '../input_audio/audio1.ogg'")