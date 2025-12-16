import base64
import whisper

class STTService:
    def __init__(self):
        self.model_name = "large"
        self.context_prompt = "Esta é uma conversa amigável e informativa em português brasileiro durante um tour pelo INTELI (Instituto de Tecnologia e Liderança)"
        self.model = None

    def setup_model(self):
        self.model = whisper.load_model(self.model_name)

    def convert_to_mp3(self, base64_audio):
        mp3_bytes = base64.b64decode(base64_audio)
        with open("output.mp3", "wb") as f:
            f.write(mp3_bytes)

    def transcribe(self, base64_audio):
        self.convert_to_mp3(base64_audio)
        result = self.model.transcribe("output.mp3")
        result_verified = self.inteli_verifier(result["text"])
        return result_verified

    def inteli_verifier(self, text):
        cognates = ["intel", "e tele", "tele", "em tele", "interior", "intelio"]
        new_text = text
        for cognate in cognates:
            new_text = new_text.lower().replace(cognate, "Inteli")

        return new_text