from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from backend.stt.stt_service import STTService
from backend.tts.tts_service import TTSService
from fastapi.middleware.cors import CORSMiddleware

# setup do STT
stt = STTService()
stt.setup_model()
teste_transcricao = stt.run_model_test()

# setup do TTS
tts = TTSService()
tts.setup_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/teste-transcricao")
async def root():
    if (teste_transcricao):
        return {"status": 200, "transcrição": teste_transcricao}
    else:
        return {"status": 500, "erro": "Modelo de Transcrição não carregou."}
    
@app.websocket("/tts")
async def falar_texto(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                broken_text = tts.text_breaker(data)
                for sentence in broken_text:
                    audio_bytes = tts.sentence_to_speech(sentence)
                    await websocket.send_bytes(audio_bytes)

    except Exception as e:
        print("WebSocket desconectado:", e)

@app.websocket("/stt")
async def falar_texto(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                transcription = stt.transcribe(data)
                await websocket.send_bytes(transcription)
    except Exception as e:
        print("WebSocket desconectado:", e)