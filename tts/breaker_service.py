import logging
from tts_service import sentence_to_speech

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BreakerService:
    """Serviço para quebrar o texto e gravar um aúdio"""

    def break_text(self, text: str):
        return text.split(".")
    
    def optimal_audio_synthesizer(self, text: str, output_path: str):

        try:
            broken_text = self.break_text(text)
            i = 0
            print(output_path)
            for sentence in broken_text:
                text_to_speech(sentence, str(i) + output_path)
                i = i + 1
        except Exception as e:
            logging.error(f"❌ Erro na síntese ótima: {e}")
            return False

breaker_service = BreakerService()

def text_to_speech(text: str, output_path: str) -> bool:
    """Função utilitária para conversão text-to-speech"""
    return breaker_service.optimal_audio_synthesizer(text, output_path)

if __name__ == "__main__":
    # Teste básico
    test_text = "Olá! Este é um teste do Google Text-to-Speech em português brasileiro. A qualidade é muito boa e funciona perfeitamente."
    test_output = "gemini_tts_test_*.mp3"

    print(f"\n🎯 Gerando áudio: '{test_text[:50]}...'")
    success = text_to_speech(test_text, test_output)

    if success:
        print(f"✅ Teste bem-sucedido! Arquivos MP3 gerado.")
    else:
        print("❌ Teste falhou")