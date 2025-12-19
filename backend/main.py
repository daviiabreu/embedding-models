from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent_flow.chat_service_v3 import ChatService
from backend.stt.stt_service import STTService
from backend.tts.tts_service import TTSService


class Prompt(BaseModel):
    message: str


# setup do STT
stt = STTService()
stt.setup_model()

# setup do TTS
tts = TTSService()
tts.setup_model()

# setup do Chat
chat = ChatService()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/teste-transcricao")
async def teste_transcricao():
    if teste_transcricao:
        return {"status": 200, "transcrição": teste_transcricao}
    else:
        return {"status": 500, "erro": "Modelo de Transcrição não carregou."}


@app.post("/chat")
async def chat_w_bot(prompt: Prompt):
    if prompt:
        prompt = chat.gives_context_to_llm(prompt.message)
        response_llm = chat.give_response(prompt)
        if response_llm:
            return {"status": 200, "response": str(response_llm)}
        return {
            "status": 200,
            "response": "Desculpe. Não consegui entender sua pergunta. Pode perguntar novamente",
        }
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
                    if sentence:
                        audio_bytes = tts.sentence_to_speech(sentence)
                        await websocket.send_bytes(audio_bytes)
                await websocket.send_bytes("END")
    except Exception as e:
        print("WebSocket desconectado:", e)


@app.websocket("/stt")
async def transcrever(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                transcription = stt.transcribe(data)
                await websocket.send_bytes(transcription)
    except Exception as e:
        print("WebSocket desconectado:", e)
