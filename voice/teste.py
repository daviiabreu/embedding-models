import os
from dotenv import load_dotenv
from google.cloud import texttospeech

# carrega variáveis do arquivo .env
load_dotenv()

# agora o Python sabe onde está a credencial e o projeto
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

print("Credencial carregada de:", credential_path)
print("Projeto atual:", project_id)

client = texttospeech.TextToSpeechClient()

prompt = "You are a friendly kid speaking with warmth and humor. Whisper when instructed."

text = (
    "My master Pietra is teaching me to be more human. [short pause] "
    "[whispering] Don't tell her, but she's really nice. [short pause] "
    "[laughing] Hahaha, I'm so excited to become human. Kisses!"
)

voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="Puck",
    model_name="gemini-2.5-pro-tts"
)

audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

try:
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text, prompt=prompt),
        voice=voice,
        audio_config=audio_config
    )
    with open("pietra_gemini.mp3", "wb") as out:
        out.write(response.audio_content)
        print("✅ Gemini TTS gerou pietra_gemini.mp3")
except Exception as e:
    print("❌ Gemini TTS indisponível:", e)