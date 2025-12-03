import os
import sys
import logging
import torch
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Adicionar diretório pai ao path para importar utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar funções utilitárias compartilhadas
from utils.gpu_utils import (
    setup_device,
    get_gpu_info,
    clear_gpu_cache,
    setup_output_directory,
    normalize_output_path,
    validate_text_input,
    log_gpu_memory,
    PerformanceMonitor
)

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==================== CLASSE PRINCIPAL ====================

class TTSServiceGPU:
    """Serviço de Text-to-Speech usando modelos locais com suporte a GPU"""

    def __init__(self, model_type: str = "bark"):
        """
        Inicializa o serviço TTS com GPU
        
        Args:
            model_type: Tipo de modelo ("bark" ou "coqui")
        """
        self.default_output_dir = setup_output_directory()
        self.device = setup_device()
        self.model_type = model_type
        self.model = None
        self.processor = None
        
        logging.info(f"Inicializando TTS com modelo: {model_type}")
        
        self._load_model()

    def _load_model(self):
        """Carrega o modelo TTS escolhido"""
        try:
            if self.model_type == "bark":
                self._load_bark()
            elif self.model_type == "coqui":
                self._load_coqui()
            else:
                raise ValueError(f"Modelo não suportado: {self.model_type}")
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
            raise

    def _load_bark(self):
        """Carrega o modelo Bark com otimizações GPU"""
        try:
            from transformers import AutoProcessor, BarkModel
            
            logging.info("Carregando Bark...")
            
            # Carregar processador e modelo
            self.processor = AutoProcessor.from_pretrained("suno/bark-small")
            self.model = BarkModel.from_pretrained("suno/bark-small")
            
            # Mover para GPU se disponível
            self.model = self.model.to(self.device)
            
            # Otimizações para GPU (FP16)
            if self.device == "cuda":
                self.model = self.model.half()
            
            logging.info("Bark carregado com sucesso")
            
        except ImportError:
            logging.error("transformers não instalado. Execute: pip install transformers")
            raise

    def _load_coqui(self):
        """Carrega o modelo Coqui TTS"""
        try:
            from TTS.api import TTS
            
            logging.info("Carregando Coqui TTS...")
            
            # Modelo multilíngue de alta qualidade
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            self.model = TTS(model_name).to(self.device)
            
            logging.info("Coqui TTS carregado com sucesso")
            
        except ImportError:
            logging.error("TTS não instalado. Execute: pip install TTS")
            raise



    def synthesize_speech(self, text: str, output_path: str, speaker: Optional[str] = None) -> bool:
        """
        Converte texto em áudio usando GPU
        
        Args:
            text: Texto para converter
            output_path: Caminho de saída
            speaker: ID do speaker (específico para cada modelo)
        """
        # Validar entrada usando função reutilizável
        if not validate_text_input(text):
            return False

        try:
            # Usar função reutilizável para normalizar caminho
            output_path = normalize_output_path(output_path, self.default_output_dir)
            
            logging.info(f"Gerando áudio no {self.device.upper()}...")
            
            # Iniciar monitoramento de performance
            monitor = PerformanceMonitor(self.device)
            monitor.start()
            
            # Mostrar uso de memória antes da síntese
            log_gpu_memory(self.device, "Antes da síntese")
            
            # Executar síntese conforme modelo
            if self.model_type == "bark":
                success = self._synthesize_bark(text, output_path, speaker)
            elif self.model_type == "coqui":
                success = self._synthesize_coqui(text, output_path, speaker)
            else:
                logging.error(f"Modelo não suportado: {self.model_type}")
                return False
            
            # Mostrar estatísticas de performance
            if success:
                stats = monitor.stop()
                log_gpu_memory(self.device, "Após síntese")
                logging.info(f"Tempo total: {stats.get('elapsed_time_formatted', 'N/A')}")
            
            return success
            
        except Exception as e:
            logging.error(f"Erro na síntese: {e}")
            return False

    def _synthesize_bark(self, text: str, output_path: Path, speaker: Optional[str]) -> bool:
        """Síntese usando Bark"""
        import scipy.io.wavfile as wavfile
        
        # Voice presets para Bark
        # v2/en_speaker_0 a v2/en_speaker_9 (inglês)
        # v2/pt_speaker_0 a v2/pt_speaker_9 (português)
        voice_preset = speaker or "v2/pt_speaker_0"
        
        # Processar texto
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Gerar áudio
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        # Converter para CPU e numpy
        audio_array = audio_array.cpu().numpy().squeeze()
        
        # Salvar (Bark usa sample rate de 24kHz)
        sample_rate = self.model.generation_config.sample_rate
        wavfile.write(str(output_path), rate=sample_rate, data=audio_array)
        
        logging.info(f"Áudio gerado: {output_path}")
        return True

    def _synthesize_coqui(self, text: str, output_path: Path, speaker: Optional[str]) -> bool:
        """Síntese usando Coqui TTS"""
        
        # Para XTTS v2, você pode clonar voz de um arquivo de referência
        # ou usar speakers pré-definidos
        
        if speaker and Path(speaker).exists():
            # Clone de voz
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker,
                language="pt"
            )
        else:
            # Usar speaker padrão
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path),
                language="pt"
            )
        
        logging.info(f"Áudio gerado: {output_path}")
        return True

    def get_gpu_info(self) -> dict:
        """Wrapper para função reutilizável get_gpu_info"""
        return get_gpu_info(self.device)

    def clear_gpu_cache(self):
        """Wrapper para função reutilizável clear_gpu_cache"""
        clear_gpu_cache(self.device)


# Instância global do serviço
tts_service_gpu = None

def initialize_tts_gpu(model_type: str = "bark") -> TTSServiceGPU:
    """Inicializa o serviço TTS com GPU"""
    global tts_service_gpu
    tts_service_gpu = TTSServiceGPU(model_type=model_type)
    return tts_service_gpu

def sentence_to_speech_gpu(text: str, output_path: str, speaker: Optional[str] = None) -> bool:
    """Função utilitária para conversão text-to-speech usando GPU"""
    if tts_service_gpu is None:
        initialize_tts_gpu()
    return tts_service_gpu.synthesize_speech(text, output_path, speaker)


# Teste do módulo
if __name__ == "__main__":
    print("TTS Service GPU - Teste")
    
    # Verificar GPU
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memória disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Nenhuma GPU detectada, usando CPU")
    
    # Escolher modelo
    print("\nModelos disponíveis:")
    print("  1. bark (Suno Bark - rápido, boa qualidade)")
    print("  2. coqui (Coqui XTTS v2 - melhor qualidade, mais lento)")
    
    model_choice = input("\nEscolha o modelo (1 ou 2) [padrão: 1]: ").strip() or "1"
    model_type = "bark" if model_choice == "1" else "coqui"
    
    # Inicializar serviço
    print(f"\nInicializando TTS com {model_type}...")
    service = initialize_tts_gpu(model_type=model_type)
    
    # Mostrar info da GPU
    gpu_info = service.get_gpu_info()
    print("\nInformações do dispositivo:")
    for key, value in gpu_info.items():
        print(f"   • {key}: {value}")
    
    # Teste
    test_text = "Olá! Este é um teste de síntese de voz usando GPU. A qualidade deve ser excelente."
    test_output = f"{model_type}_gpu_test.wav"
    
    print(f"\nGerando áudio...")
    print(f"Texto: '{test_text}'")
    
    import time
    start_time = time.time()
    
    success = sentence_to_speech_gpu(test_text, test_output)
    
    elapsed_time = time.time() - start_time
    
    if success:
        output_file = service.default_output_dir / test_output
        size = output_file.stat().st_size / 1024  # KB
        
        print(f"\nTeste bem-sucedido!")
        print(f"Arquivo: {output_file}")
        print(f"Tamanho: {size:.2f} KB")
        print(f"Tempo: {elapsed_time:.2f}s")
        
        # Info final da GPU
        if gpu_info["available"]:
            final_info = service.get_gpu_info()
            print(f"\nMemória GPU usada: {final_info['memory_allocated']}")
    else:
        print("\nTeste falhou")