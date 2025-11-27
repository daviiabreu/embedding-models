import whisper
from typing import Optional
import os


class STTService:

    def __init__(self):
        self.model_name = "small"
        self.context_prompt = "Esta é uma conversa amigável e informativa em português brasileiro durante um tour pelo INTELI (Instituto de Tecnologia e Liderança)"
        self.model = None



    def setup_model(self):
        self.model = whisper.load_model(self.model_name)

    def run_model_test(self):
        result = self.model.transcribe("stt/audio_test.ogg")
        return result["text"]

