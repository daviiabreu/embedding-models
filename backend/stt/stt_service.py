import whisper
import base64

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

    def convert_to_mp3 (self, base64_audio):
        mp3_bytes = base64.b64decode(base64_audio)
        with open("output.mp3", "wb") as f:
            f.write(mp3_bytes)

    def transcribe(self, base64_audio):
        self.convert_to_mp3(base64_audio)
        result = self.model.transcribe("output.mp3")
        return result["text"]

