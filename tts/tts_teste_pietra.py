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
            name="Leda",
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

def text_to_speech(text: str, output_path: str) -> bool:
    """Função utilitária para conversão text-to-speech"""
    return tts_service.synthesize_speech(text, output_path)

# Teste do módulo
if __name__ == "__main__":
    # Roteiro do tour — cada item é um trecho com entonação diferente
    tour_segments = [
        {
            "id": 0,
            "filename": "trecho_0.mp3",
            "style": None,  
            "text": (
                "[excited] Que alegria receber vocês aqui hoje. [strong emphasis] Qual é o nome de vocês?[short pause]"
            ),
        },
        {
            "id": 1,
            "filename": "trecho_1.mp3",
            "style": None,  
            "text": (
                "[hopeful] Sejam bem-vindos à minha casa: o Inteli [high pitch]. [short pause]"
                "Ou, como os nossos fundadores gostam de brincar... o “MIT brasileiro” [laugh] [short pause]."
            ),
        },
        {
            "id": 2,
            "filename": "trecho_2.mp3",
            "style": "Excited",
            "text": (
                "[storytelling] O Inteli foi fundado recentemente, em 2019, depois de uma conversa do Roberto Sallouti no Vale do Silício com um dos maiores empresários de Venture Capital do país. [short pause]"
                "A história verdadeira foi mais ou menos assim: [clearing throat]"
            ),
        },
        {
            "id": 3,
            "filename": "trecho_3.mp3",
            "style": "playful", 
            "text": (
                "Auau, disse o Roberto. [short pause]"
                "Auau, respondeu o empresário. [laughing] [medium pause]"
                "E então, nasceu o Inteli! [cheerful]"
            ),
        },
        {
            "id": 4,
            "filename": "trecho_4.mp3",
            "style": "laughing", 
            "text": (
                "Brincadeira. [short pause]"
            ),
        },
        {
            "id": 5,
            "filename": "trecho_5.mp3",
            "style": "seriuos", 
            "text": (
                "Na verdade, a conversa foi assim: [short pause]"
                "'Por que você investe tão pouco no Brasil?' - Perguntou o Roberto. [short pause]"
                "[disappointed] 'É porque o Brasil não forma engenheiros o suficiente. - Respondeu o empresário. [medium pause]"
            ),
        },

    ]

    # Escolha os trechos que deseja gerar
    SEGMENTS_TO_RUN = [1]  # <<< ALTERE AQUI
    # SEGMENTS_TO_RUN = "all"

    if SEGMENTS_TO_RUN == "all":
        segments_to_process = tour_segments
    else:
        segments_to_process = [s for s in tour_segments if s["id"] in SEGMENTS_TO_RUN]

    if not segments_to_process:
        print("⚠️ Nenhum trecho selecionado para gerar.")
        exit()

    for segment in segments_to_process:
        styled_text = f"[{segment['style']}] {segment['text']}"
        output_name = segment["filename"]

        print(f"\n🎯 Gerando trecho {segment['id']}: {output_name}")
        success = text_to_speech(styled_text, output_name)

        if success:
            print(f"✅ Trecho {segment['id']} gerado com sucesso!")
        else:
            print(f"❌ Erro ao gerar o trecho {segment['id']}.")

    print("\n📁 Processo concluído!")