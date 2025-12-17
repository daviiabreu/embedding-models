import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import texttospeech

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

class TTSService:
    def __init__(self):
        self.default_output_dir = self.setup_output_directory()

    def setup_output_directory(self):
        current_dir = Path(__file__).parent
        output_dir = current_dir.parent / "output_audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def synthesize_speech(self, text: str, output_path: str, speed: float = 1.0) -> bool:
        if not text: return False

        client = texttospeech.TextToSpeechClient()
        
        # Configuração da Voz (Leda + Gemini)
        voice = texttospeech.VoiceSelectionParams(
            language_code="pt-BR",
            name="Leda", 
            model_name="gemini-2.5-pro-tts"
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=speed)  # Controla a velocidade (1.0 = normal, 1.2 = 20% mais rápido)

        try:
            # Garante caminho correto
            path_obj = Path(output_path)
            if not path_obj.is_absolute():
                path_obj = self.default_output_dir / path_obj.name
            
            # Verifica cache (Se já existe, não gera de novo)
            if path_obj.exists():
                logging.info(f"⏭️  Já existe (pulando): {path_obj.name}")
                return True

            response = client.synthesize_speech(
                request={
                    "input": texttospeech.SynthesisInput(text=text),
                    "voice": voice,
                    "audio_config": audio_config
                }
            )

            with open(path_obj, "wb") as out:
                out.write(response.audio_content)
                logging.info(f"✅ Gerado (Velocidade {speed}): {path_obj.name}")
            return True

        except Exception as e:
            logging.error(f"Erro em {output_path}: {e}")
            return False

# ROTEIRO E EXECUÇÃO
if __name__ == "__main__":
    tts = TTSService()

    VELOCIDADE_PADRAO = 1.15

    # LISTA MESTRA DO ROTEIRO
    tour_script = [
        # BLOCO 1: BOAS VINDAS
        { "id": "01_boas_vindas", "emotion": "[excited, happy]", "text": "Que alegria receber vocês aqui hoje!" },
        { "id": "02_mit_brasileiro", "emotion": "[proud, warm]", "text": "Sejam bem-vindos à minha casa, o Inteli! Ou, para nossos fundadores, o MIT Brasileiro." },
        
        # BLOCO 2: HISTÓRIA 
        { "id": "03_intro_historia", "emotion": "[storytelling, happy]", "text": "O Inteli foi fundado há pouco tempo, em 2019, e foi o resultado de uma conversa do Roberto Saluti no Vale do Silício com um dos maiores empresários de Venture Capital do país." },
        { "id": "04_historia_importante", "emotion": "[joking, inspiring]", "text": "A conversa foi mais ou menos assim:", "speed": 1.1 },
        { "id": "05_piada_auau", "emotion": "[playful, joking, happy]", "text": "Auauu.., disse o Sallouti. Auau, respondeu o empresário. E então, surgiu o Inteli!", "speed": 1.1 },
        { "id": "06_correcao_seria", "emotion": "[laughing then serious]", "text": "Brincadeira! Na verdade foi assim. Por que você investe tão pouco no Brasil? Perguntou o Roberto." },
        { "id": "07_resposta_empresario", "emotion": "[deep voice, angry]", "text": "E o empresário respondeu: Porque o Brasil não forma engenheiros o suficiente." },
        { "id": "08_resolucao", "emotion": "[inspiring, determined, exciting]", "text": "E então, como tanto o Roberto quanto nosso outro fundador, o André Estêves, queriam deixar um legado para o Brasil, eles pensaram: Ele tá certo! E resolveram: Nós vamos formar esses engenheiros. Daí surgiu o Inteli. Um legado de brasileiros para brasileiros." },

        # BLOCO 3: BOLSAS 
        { "id": "09_missao", "emotion": "[passionate, clear]", "text": "Continuando. O Inteli tem a missão de formar os futuros líderes que vão transformar o Brasil através da tecnologia. E a gente acredita que esses líderes podem vir dos mais diversos contextos e é por isso que temos o maior programa de bolsas de estudo do ensino superior do Brasil.." },
        { "id": "10_lista_beneficios", "emotion": "[enumerating, clear]", "text": "A gente oferece: Auxílio-moradia; Auxílio-alimentação; Auxílio-transporte; curso de inglês e até notebook. Além das modalidades de bolsa parcial e integral." , "speed": 1.2 },
        { "id": "11_doadores", "emotion": "[grateful, soft]", "text": "Mas isso só foi possível, porque encontramos doadores-parceiros com o mesmo sonho de investir no desenvolvimento dos futuros líderes de tecnologia do país. Pessoas que investiram pelo menos 500 mil reais nesses alunos. Aqui em cima nesse painel, vocês podem ver os nomes deles..", "speed": 1.25 },
        { "id": "12_interacao_bolsas", "emotion": "[helpful, inviting]", "text": "Vocês têm alguma pergunta sobre a história ou o programa de bolsas do Inteli?.." , "speed": 1.18},
        { "id": "13_transicao", "emotion": "[upbeat]", "text": "Então bora pra próxima parada. Caso ainda tenham alguma dúvida, podem ir digitando no app. Vou responder na nossa próxima parada ou mandar uma mensagem pra vocês com as respostas de suas dúvidas." },

        # BLOCO 4: CURSOS
        { "id": "14_intro_cursos", "emotion": "[excited, high energy]", "text": "E agora, chegou a hora de falar sobre algo que vocês vão adorar: os cursos e clubes do Inteli!" },
        { "id": "17_cursos_tecnicos", "emotion": "[informative, professional]", "text": "Temos 5 cursos que formam os futuros líderes de tecnologia do país: Ciência da Computação é o curso-mãe, focado em algoritmos e inteligência artificial. Engenharia de Software constrói grandes sistemas." },       
        { "id": "18_cursos_business", "emotion": "[informative, professional]", "text": "Sistemas de Informação conecta tecnologia e estratégia, eles entendem de banco de dados e gestão. E ADM Tech une gestão e tecnologia, formando os próximos empreendedores." },        
        { "id": "16_piada_ec", "emotion": "[whispering, conspiratorial, laughing]", "text": "Engenharia de Computação cria soluções inovadoras. E adivinhem só... os alunos desse são os mais legais de todos! Até porque foram eles que me programaram, então não posso ser imparcial!" },

        # BLOCO 5: CLUBES
        { "id": "19_intro_clubes", "emotion": "[playful, storytelling]", "text": "Agora… se vocês acham que a vida de um intéler se resume a cálculos e derivadas... Errado! É nos clubes estudantis que a mágica acontece!", "speed": 1.23 },
        { "id": "20_quantidade_clubes", "emotion": "[energetic, dramatic]", "text": "Aqui no Inteli tem clube pra tudo — e quando eu digo tudo, é TU-DO mesmo. São mais de vinte grupos diferentes, todos criados e liderados pelos próprios alunos." },
        { "id": "21_lista_clubes", "emotion": "[energetic, fast-paced]", "text": "Tem a Tantéra, nossa atlética que faz o campus todo vibrar nos jogos. A Intéli Júnior, a empresa júnior que entrega projetos reais pra clientes de verdade. A LEI, liga de empreendedorismo, onde o pessoal respira inovação e sonha com o próximo unicórnio brasileiro. E se vocês acham que é só isso, segura aí...", "speed": 1.19 },
        { "id": "22_lista2_clubes", "emotion": "[energetic, passionate]", "text": "Temos ainda... a AgroTec, que leva tecnologia pro campo. O Game Lab, que desenvolve jogos incríveis. O Intéli Blockchain, que ganha hackathon atrás de hackathon com projetos de Web3. E a Inteli Academy, focada em IA: o pessoal que cria mentes tipo a minha a", "speed": 1.20 },
        { "id": "24_wave", "emotion": "[excited, remembering]", "text": "Ah, e não posso esquecer da Wave, a comunidade que ajuda candidatos a entrarem no Inteli com mentorias e simulados." },    
        { "id": "23_diversidade", "emotion": "[respectful, gentle]", "text": "Além dos clubes, temos grupos que tornam o Inteli um lugar diverso e acolhedor: O Coletivo Feminino Grace Hopper, o Coletivo Negro Benedito Caravelas e o Coletivo LGBTQIAPN+ Turing. Eles garantem representatividade e respeito." },
        { "id": "25_fim_clube", "emotion": "[super excited, fast-paced]", "text": "No fim, o que torna o Intéli tão especial é isso: Aqui, a aprendizagem vai muito além da sala de aula. Os clubes são pequenos laboratórios onde os alunos aprendem habilidades que vão levar pra vida toda — liderança, trabalho em equipe, comunicação e propósito." },
        { "id": "26_qa_clubes", "emotion": "[helpful]", "text": "e esse foi só um pouquinho sobre os cursos e clubes daqui. Se vocês quiserem saber mais sobre algum clube, fiquem à vontade pra perguntar agora!" },

        # BLOCO 6: PBL E ROTINA 
        { "id": "24_sem_materias", "emotion": "[shocked, dramatic]", "text": "E agora, preparem-se para um espanto: O Inteli não tem matérias!" },
        { "id": "25_explicacao_pbl", "emotion": "[explanatory, educational]", "text": "Os fundadores não queriam criar uma faculdade tradicional. Eles queriam também trazer a inovação por meio dela. E é daí que surge a ideia de implementar aqui no Brasil, um método de ensino famoso lá fora, mas pouco conhecido aqui dentro: o Ensino baseado em Projetos ou PBL para os íntimos." },
        { "id": "26_aprendendo_testando", "emotion": "[proud, reflective]", "text": "Nesse modelo, os alunos aprendem, tudo na prática. Então, em vez de cursar disciplinas isoladas, eles aprendem de um modo chamado de transdisciplinar. Isso quer dizer que no fim, eles acabam aprendendo os conteúdos de Cálculo I, mas isso acontece de forma dinâmica e não compete só ao professor de matemática, mas cruza com Negócios, Programação, Design, e principalmente, cruza com um projeto real." },
        { "id": "26_aprendendo_testando2", "emotion": "[proud, reflective]", "text": "Foi testando que eu aprendi a falar. E foi testando que eles aprenderam sobre LLMs, Redes Neurais e até sobre o mercado de robôs autônomos no Brasil. Entre teoria e prática, aqui escolhemos os dois." },
        { "id": "27_momento_dev", "emotion": "[engaging, question]", "text": "No primeiro ano, os alunos aprendem matemática e programação enquanto criam um jogo real. Esse projeto é encomendado por grandes parceiros do Inteli, como Meta, Google e Vivo" },
        { "id": "27_momento2_dev", "emotion": "[engaging, question]", "text": "Como o prazo é curto — apenas dez semanas — ninguém faz nada sozinho. O foco aqui é o trabalho em conjunto. Por isso eles têm acesso a esta 'casinha', onde colaboram durante o momento de DEV." },
        { "id": "28_explicacao_dev", "emotion": "[informative]", "text": "O DEV é um dos três momentos que a gente tem aqui no Inteli para o aluno aprender. Essa é a hora de colocar a mão na massa. É o momento em que eles desenvolvem o projeto efetivamente. Para o primeiro ano, isso acontece todos os dias, das duas às quatro da tarde." },
        { "id": "33_explicacao_autoestudo:", "emotion": "[warm, closing]", "text": "Mas eu disse pra vocês que existem três momentos, certo? Os outros dois são: o autoestudo e o encontro. O autoestudo começa pela manhã, onde os alunos acessam o material da aula e estudam sozinhos ou em grupo nas mesas que vimos ali atrás." },
        { "id": "34_autoestudo:", "emotion": "[warm, closing]", "text": "Isso garante que todos cheguem preparados para o terceiro momento: o Encontro. Mas sobre ele, nós vamos falar na nossa próxima parada." },
        { "id": "35_finalizacao", "emotion": "[warm, closing]", "text": "Vocês têm alguma pergunta sobre a metodologia de ensino ou a nossa rotina aqui na instituição?" }        
    ]

    # CONTROLE DE EXECUÇÃO
    
    # OPÇÃO 1: Gerar TODOS os áudios
    SEGMENTS_TO_RUN = "26_qa_clubes"

    # OPÇÃO 2: Gerar apenas áudios específicos (Descomente abaixo para usar)
    # Copie o "id" exato que está na lista acima.
    # SEGMENTS_TO_RUN = ["01_boas_vindas", "16_piada_ec"] 

    # LÓGICA DE FILTRAGEM 
    if SEGMENTS_TO_RUN == "all":
        queue = tour_script
        print(f"\n Modo selecionado: GERAR TUDO ({len(queue)} trechos)")
    else:
        # Filtra a lista principal procurando apenas os IDs solicitados
        queue = [item for item in tour_script if item["id"] in SEGMENTS_TO_RUN]
        print(f"\nModo selecionado: GERAR APENAS {len(queue)} TRECHOS")

    if not queue:
        print("Nenhum trecho encontrado. Verifique se os IDs estão corretos.")
        exit()

    # LOOP DE GERAÇÃO 
    for item in queue:
        # 1. Monta o texto com emoção: "[excited] Olá"
        texto_final = f"{item['emotion']} {item['text']}"
        
        # 2. Define nome do arquivo: "01_boas_vindas.mp3"
        nome_arquivo = f"{item['id']}.mp3"

        velocidade = item.get("speed", VELOCIDADE_PADRAO)

        print(f"Processando: {item['id']} (Velocidade: {velocidade})")
        
        # 3. Chama a API
        tts.synthesize_speech(texto_final, nome_arquivo, speed=velocidade)

    print("\nProcesso concluído! Verifique a pasta 'output_audio'.")