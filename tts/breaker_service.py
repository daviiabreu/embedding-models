from tts_service import text_to_speech
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

class BreakerService:

    def text_breaker(self, text:str):
        return text.split(".")
    
    def optimal_audio_synthesizer(self, text: str, output_path: str):
        try:
            broken_text = self.text_breaker(text)
            i = 0
            print(output_path)
            for sentence in broken_text:
                text_to_speech(sentence, output_path + "-" + str(i) + ".wav")
                i = i + 1

        
            return True
        except Exception as e:
            logging.error(f"Erro na otimização do áudio: {e}")
            return False



# Instância global do serviço
breaker_service = BreakerService()

def optimal_tts_synthesizer(text: str, output_path: str) -> bool:
    """Função utilitária para conversão text-to-speech"""
    
    return breaker_service.optimal_audio_synthesizer(text, output_path)

# Teste do módulo
if __name__ == "__main__":
    print("Testando o Optimal TTS Synthesizer...")

    text = "O Daniel é um ótimo apresentador. Ele fará a apresentação da sprint 3"
    test_output = "gemini_tts_test_*.mp3"

    success = optimal_tts_synthesizer(text, test_output)

    if success:
        print("Deu certo!")

