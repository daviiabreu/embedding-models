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

# Texto que será transformado em fala
text = "Olá tudo bem? Minha mestre Pietra está me testando hahaha!"

# Entrada de texto
synthesis_input = texttospeech.SynthesisInput(
    text="Olá tudo bem? Minha mestre Pietra está me testando hahaha!",
    prompt="Fale com um tom amigável e alegre."
)

# Escolha de voz (português, modelo Neural2)
voice = texttospeech.VoiceSelectionParams(
    language_code="pt-BR",
    name="Puck",
    model_name="gemini-2.5-pro-tts",
)

# Configurações do áudio de saída
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Faz a chamada à API
response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

# Salva o áudio localmente
with open("saida.mp3", "wb") as out:
    out.write(response.audio_content)
    print("Áudio salvo em: saida.mp3")