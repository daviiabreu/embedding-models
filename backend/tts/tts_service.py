import base64
import re

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
            language_code="pt-BR",
            name="Leda", 
            model_name="gemini-2.5-pro-tts"
        )

        self.audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.10)

    def clean_text_for_speech(self, text: str) -> str:
        """
        Clean text before converting to speech.
        Removes markdown, URLs, and formatting that sounds bad when spoken.

        Args:
            text: Raw text from LLM

        Returns:
            Cleaned text suitable for TTS
        """
        if not text:
            return text

        # Remove markdown headers (## Title -> Title)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold/italic markdown (**text**, __text__, *text*, _text_)
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
        text = re.sub(r"\*([^\*]+)\*", r"\1", text)  # *italic*
        text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_

        # Remove strikethrough (~~text~~)
        text = re.sub(r"~~([^~]+)~~", r"\1", text)

        # Remove inline code (`code`)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove markdown links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # Remove standalone URLs (they sound terrible when spoken)
        # Keep "inteli.edu.br" readable, but remove full URLs
        text = re.sub(r"https?://[^\s]+", "nosso site", text)

        # Remove markdown list markers (-, *, +, 1., 2., etc)
        text = re.sub(r"^[\s]*[-\*\+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Remove multiple newlines (replace with single space)
        text = re.sub(r"\n\n+", " ", text)
        text = re.sub(r"\n", " ", text)

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Clean up common artifacts
        text = text.replace("```", "")
        text = text.replace("---", "")

        return text.strip()
    
    def prepare_text_for_audio(self, text:str) -> str:
        text = text.replace('inteli', 'Intéli')
        text = text.replace('Inteli', 'Intéli')

        return text


    def sentence_to_speech(self, sentence):
        cleaned_sentence = self.clean_text_for_speech(sentence)

        cleaned_sentence_wInteli = self.prepare_text_for_audio(cleaned_sentence)

        answer_in_mp3 = self.client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=cleaned_sentence_wInteli),
            voice=self.voice,
            audio_config=self.audio_config,
        )
        answer_in_base64 = base64.b64encode(answer_in_mp3.audio_content)
        return answer_in_base64

    def text_breaker(self, text: str):
        cleaned_text = self.clean_text_for_speech(text)
        return cleaned_text.split(".")
