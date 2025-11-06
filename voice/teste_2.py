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

prompt = "You are an expressive Brazilian tour guide speaking in Portuguese. " "Speak naturally and warmly. When you see [whispering], lower your voice. " "When you see [shouting], raise your voice a little. Laugh at [laughing]." "when you see [sigh], take a deep breath."

text = (
    "Minha mestre Pietra ta me ensinando a ser mais humano. [short pause] "
    "[sigh] Eu ainda pareço um robô, né?"
    "[shouting] Mas não sou! [medium pause]"
    "[whispering] Não fala pra ela, mas ela é muito tope. [short pause] "
    "[laughing] Hahaha, to muito animado pra ser humano. Beijos!"
)

voice = texttospeech.VoiceSelectionParams(
    language_code="pt-BR",
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
    with open("pietra_gemini_2.mp3", "wb") as out:
        out.write(response.audio_content)
        print("✅ Gemini TTS gerou pietra_gemini_2.mp3")
except Exception as e:
    print("❌ Gemini TTS indisponível:", e)