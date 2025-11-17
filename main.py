import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Adicionar diretórios ao path para importações - CAMINHOS CORRIGIDOS
project_root = Path(__file__).parent  # main.py está na raiz agora
sys.path.append(str(project_root / "pipeline"))  # Para llm_service
sys.path.append(str(project_root / "stt"))       # Para stt_service
sys.path.append(str(project_root / "tts"))       # Para tts_service

from stt_service import transcribe_audio
from llm_service import get_llm_response
from tts_service import text_to_speech

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

class AudioPipeline:
    """Pipeline completa: Áudio → Transcrição → LLM → TTS → Áudio"""

    def __init__(self):
        self.setup_directories()

    def setup_directories(self):
        """Cria diretórios necessários"""
        # Diretórios agora na raiz do projeto
        self.input_dir = Path(__file__).parent / "input_audio"
        self.output_dir = Path(__file__).parent / "output_audio"

        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        logging.info(f"📁 Diretório de entrada: {self.input_dir}")
        logging.info(f"📁 Diretório de saída: {self.output_dir}")

    def process_audio(self, audio_filename: str, conversation_context: str = None):
        """
        Processa um arquivo de áudio através da pipeline completa

        Args:
            audio_filename: Nome do arquivo na pasta input_audio
            conversation_context: Contexto adicional para a LLM

        Returns:
            tuple: (sucesso, caminho_audio_resposta, transcrição, resposta_llm)
        """
        audio_path = self.input_dir / audio_filename

        if not audio_path.exists():
            logging.error(f"❌ Arquivo não encontrado: {audio_path}")
            return False, None, None, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(audio_filename).stem

        try:
            # ETAPA 1: Transcrever áudio
            logging.info(f"🎤 Iniciando transcrição de: {audio_filename}")
            transcription = transcribe_audio(str(audio_path))

            if not transcription:
                logging.error("❌ Falha na transcrição")
                return False, None, None, None

            logging.info(f"✅ Transcrição concluída: {transcription[:100]}...")

            # ETAPA 2: Enviar para LLM
            logging.info("🤖 Enviando para LLM...")
            llm_response = get_llm_response(transcription, conversation_context)

            if not llm_response:
                logging.error("❌ Falha na resposta da LLM")
                return False, None, transcription, None

            logging.info(f"✅ Resposta da LLM: {llm_response[:100]}...")

            # ETAPA 3: Converter resposta para áudio
            logging.info("🔊 Convertendo resposta para áudio...")

            # Ajustar extensão baseada no TTS usado
            # Se usar gTTS: .mp3 | Se usar Bark/XTTS: .wav
            output_filename = f"{base_name}_response_{timestamp}.wav"  # Mudado para .wav
            output_path = self.output_dir / output_filename

            audio_success = text_to_speech(llm_response, str(output_path))

            if not audio_success:
                logging.error("❌ Falha na conversão para áudio")
                return False, None, transcription, llm_response

            logging.info(f"✅ Áudio gerado: {output_path}")

            return True, str(output_path), transcription, llm_response

        except Exception as e:
            logging.error(f"❌ Erro na pipeline: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False, None, None, None

    def process_all_audio_files(self):
        """Processa todos os arquivos de áudio na pasta de entrada"""
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}

        audio_files = [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            logging.warning(f"⚠️ Nenhum arquivo de áudio encontrado em {self.input_dir}")
            return

        logging.info(f"🎯 Encontrados {len(audio_files)} arquivos para processar")

        results = []
        for audio_file in audio_files:
            logging.info(f"\n{'='*50}")
            logging.info(f"🔄 Processando: {audio_file.name}")

            success, output_path, transcription, llm_response = self.process_audio(
                audio_file.name
            )

            results.append({
                'input_file': audio_file.name,
                'success': success,
                'output_file': output_path,
                'transcription': transcription,
                'llm_response': llm_response
            })

        # Relatório final
        logging.info(f"\n{'='*50}")
        logging.info("📊 RELATÓRIO FINAL")

        successful = sum(1 for r in results if r['success'])
        logging.info(f"✅ Sucessos: {successful}/{len(results)}")

        for result in results:
            status = "✅" if result['success'] else "❌"
            logging.info(f"{status} {result['input_file']}")

def main():
    """Função principal da pipeline"""
    logging.info("🚀 Iniciando Pipeline de Áudio")

    pipeline = AudioPipeline()

    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1:
        # Processar arquivo específico
        audio_filename = sys.argv[1]
        context = sys.argv[2] if len(sys.argv) > 2 else None

        logging.info(f"📁 Processando arquivo específico: {audio_filename}")
        success, output_path, transcription, llm_response = pipeline.process_audio(
            audio_filename, context
        )

        if success:
            logging.info(f"🎉 Pipeline concluída com sucesso!")
            logging.info(f"📄 Transcrição: {transcription}")
            logging.info(f"🤖 Resposta LLM: {llm_response}")
            logging.info(f"🔊 Áudio gerado: {output_path}")
        else:
            logging.error("❌ Pipeline falhou")
    else:
        # Processar todos os arquivos
        logging.info("📁 Processando todos os arquivos na pasta de entrada")
        pipeline.process_all_audio_files()

if __name__ == "__main__":
    main()
