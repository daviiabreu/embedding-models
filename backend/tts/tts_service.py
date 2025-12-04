import base64

from dotenv import load_dotenv
from google.cloud import texttospeech

# Carregar variáveis de ambiente
load_dotenv()


class TTSService:
    def __init__(self):
        self.client = None
        self.voice = None
        self.audio_config = None

    def setup_model(self):
        self.client = texttospeech.TextToSpeechClient()

        self.voice = texttospeech.VoiceSelectionParams(
            language_code="pt-BR", name="Puck", model_name="gemini-2.5-pro-tts"
        )

        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

    def sentence_to_speech(self, sentence):
        answer_in_mp3 = self.client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=sentence),
            voice=self.voice,
            audio_config=self.audio_config,
        )
        answer_in_base64 = base64.b64encode(answer_in_mp3.audio_content)
        return answer_in_base64

    def text_breaker(self, text: str):
        return text.split(".")
