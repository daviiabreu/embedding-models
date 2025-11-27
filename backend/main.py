from fastapi import FastAPI
from backend.stt.stt_service import STTService

stt = STTService()
stt.setup_model()
teste_transcricao = stt.run_model_test()

app = FastAPI()

@app.get("/")
async def root():
    if (teste_transcricao):
        return {"status": 200, "Transcrição obtida do Whisper": teste_transcricao}
    else:
        return {"status": 500}