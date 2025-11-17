"""
Pipeline de Processamento de Áudio
==================================

Pipeline completa para processar áudio através de:
1. Speech-to-Text (Whisper)
2. Large Language Model (Gemini)
3. Text-to-Speech (Google Cloud TTS)

Uso:
    python main.py [arquivo_audio.mp3] [contexto_opcional]

    Ou para processar todos os arquivos:
    python main.py
"""

__version__ = "1.0.0"
__author__ = "Seu Nome"

from ..main import AudioPipeline
from .llm_service import get_llm_response, LLMService
from .tts_service import text_to_speech, TTSService

__all__ = [
    "AudioPipeline",
    "get_llm_response",
    "LLMService",
    "text_to_speech",
    "TTSService"
]
