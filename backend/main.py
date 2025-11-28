from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from backend.stt.stt_service import STTService
from backend.tts.tts_service import TTSService

# setup do STT
stt = STTService()
stt.setup_model()
teste_transcricao = stt.run_model_test()

# setup do TTS
tts = TTSService()
tts.setup_model()

app = FastAPI()

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
                audio_bytes = tts.sentence_to_speech(data)
                await websocket.send_bytes(audio_bytes)
    except Exception as e:
        print("WebSocket desconectado:", e)