import os
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import texttospeech


# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TTSService:
    """Serviço de Text-to-Speech usando Google TTS (gTTS)"""


    def __init__(self):
        self.default_output_dir = self.setup_output_directory()

    def setup_output_directory(self):
        """Cria o diretório de saída padrão"""
        current_dir = Path(__file__).parent
        output_dir = current_dir.parent / "output_audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"📁 Diretório de saída: {output_dir}")
        return output_dir

    def normalize_output_path(self, output_path: str) -> Path:
        """Normaliza o caminho de saída para MP3"""
        output_path = Path(output_path)

        # Se não é absoluto, usar diretório padrão
        if not output_path.is_absolute():
            output_path = self.default_output_dir / output_path.name

        # Garantir extensão MP3 (formato nativo do gTTS)
        if output_path.suffix.lower() not in ['.mp3']:
            output_path = output_path.with_suffix('.mp3')

        # Criar diretório
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def synthesize_speech(self, text: str, output_path: str, voice_speed: bool = False) -> bool:
        """Converte texto em áudio usando gTTS"""

        if not text or not text.strip():
            logging.error("❌ Texto vazio fornecido")
            return False

        client = texttospeech.TextToSpeechClient()

        voice = texttospeech.VoiceSelectionParams(
            language_code="pt-BR",
            name="Puck",
            model_name="gemini-2.5-pro-tts"
        )

        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        try:
            # Normalizar caminho (sempre MP3)
            output_path = self.normalize_output_path(output_path)


            response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=voice,
                audio_config=audio_config
            )

            with open(output_path, "wb") as out:
                out.write(response.audio_content)
                print("✅ TTS gerou o arquivo de áudio:", output_path)

            return True

        except Exception as e:
            logging.error(f"❌ Erro na síntese: {e}")
            return False


# Instância global do serviço
tts_service = TTSService()

def sentence_to_speech(text: str, output_path: str) -> bool:
    """Função utilitária para conversão text-to-speech"""
    return tts_service.synthesize_speech(text, output_path)

# Teste do módulo
if __name__ == "__main__":
    # Teste básico
    test_text = "Olá! Este é um teste do Google Text-to-Speech em português brasileiro. A qualidade é muito boa e funciona perfeitamente.[LATIDO]"
    test_output = "gemini_tts_test_*.mp3"

    print(f"\n🎯 Gerando áudio: '{test_text[:50]}...'")
    success = sentence_to_speech(test_text, test_output)

    if success:
        print(f"✅ Teste bem-sucedido! Arquivo MP3 gerado.")

        # Mostrar arquivos gerados
        output_dir = tts_service.default_output_dir
        audio_files = list(output_dir.glob("gemini_tts_test_*.mp3"))
        if audio_files:
            print(f"📁 Arquivos gerados:")
            for file in audio_files:
                size = file.stat().st_size
                print(f"   • {file.name} ({size} bytes)")
    else:
        print("❌ Teste falhou")
