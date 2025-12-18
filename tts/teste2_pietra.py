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
        ## Checkpoint 1
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

        ## Checkpoint 2
        # BLOCO 4: CURSOS
        { "id": "14_intro_cursos", "emotion": "[excited, captivating]", "text": "E agora, chegou a hora de falar sobre algo que vocês vão adorar: os cursos e clubes do Inteli!" },
        { "id": "15_cursos_tecnicos", "emotion": "[knowledgeable, inspiring, clear]", "text": "Temos 5 cursos que formam os futuros líderes de tecnologia do país: Ciência da Computação é o curso-mãe, focado em algoritmos e inteligência artificial. Engenharia de Software constrói grandes sistemas.", "speed": 1.18 },       
        { "id": "16_cursos_business", "emotion": "[confident, professional]", "text": "Sistemas de Informação conecta tecnologia e estratégia, eles entendem de banco de dados e gestão. E ADM Tech une gestão e tecnologia, formando os próximos empreendedores." },        
        { "id": "17_piada_ec", "emotion": "[proud, then laughing]", "text": "Engenharia de Computação cria soluções inovadoras. E adivinhem só... os alunos desse curso são os mais legais de todos! Até porque foram eles que me programaram, então não posso ser imparcial!","speed": 1.18 },

        # BLOCO 5: CLUBES
        { "id": "19_intro_clubes", "emotion": "[playful, storytelling]", "text": "Agora… se vocês acham que a vida de um intéler se resume a cálculos e derivadas... Errado! É nos clubes estudantis que a mágica acontece!", "speed": 1.23 },
        { "id": "20_quantidade_clubes", "emotion": "[energetic, dramatic]", "text": "Aqui no Inteli tem clube pra tudo — e quando eu digo tudo, é TU-DO mesmo. São mais de vinte grupos diferentes, todos criados e liderados pelos próprios alunos." },
        { "id": "21_lista_clubes", "emotion": "[energetic, fast-paced]", "text": "Tem a Tantéra, nossa atlética que faz o campus todo vibrar nos jogos. A Inteli Júnior, a empresa júnior que entrega projetos reais pra clientes de verdade. A LEI, liga de empreendedorismo, onde o pessoal respira inovação e sonha com o próximo unicórnio brasileiro. E se vocês acham que é só isso, segura aí...", "speed": 1.19 },
        { "id": "22_lista2_clubes", "emotion": "[energetic, passionate]", "text": "Temos ainda... a AgroTec, que leva tecnologia pro campo. O Game Lab, que desenvolve jogos incríveis. O Inteli Blockchain, que ganha hackathon atrás de hackathon com projetos de Web3. E a Inteli Academy, focada em IA: o pessoal que cria mentes tipo a minha a", "speed": 1.20 },
        { "id": "24_wave", "emotion": "[excited, remembering]", "text": "Ah, e não posso esquecer da Wave, a comunidade que ajuda candidatos a entrarem no Inteli com mentorias e simulados." },    
        { "id": "23_diversidade", "emotion": "[respectful, gentle]", "text": "Além dos clubes, temos grupos que tornam o Inteli um lugar diverso e acolhedor: O Coletivo Feminino Grace Hopper, o Coletivo Negro Benedito Caravelas e o Coletivo LGBTQIAPN+ Turing. Eles garantem representatividade e respeito." },
        { "id": "25_fim_clube", "emotion": "[super excited, fast-paced]", "text": "No fim, o que torna o Inteli tão especial é isso: Aqui, a aprendizagem vai muito além da sala de aula. Os clubes são pequenos laboratórios onde os alunos aprendem habilidades que vão levar pra vida toda — liderança, trabalho em equipe, comunicação e propósito." },
        { "id": "26_qa_clubes", "emotion": "[helpful]", "text": "e esse foi só um pouquinho sobre os cursos e clubes daqui. Se vocês quiserem saber mais sobre algum clube, fiquem à vontade pra perguntar agora!" },

        ## Checkpoint 3
        # BLOCO 6: PBL E ROTINA 
        { "id": "26_sem_materias", "emotion": "[shocked, dramatic]", "text": "E agora, preparem-se para um espanto: O Inteli não tem matérias!" },
        { "id": "27_explicacao_pbl", "emotion": "[explanatory, educational]", "text": "Os fundadores não queriam criar uma faculdade tradicional. Eles queriam também trazer a inovação por meio dela. E é daí que surge a ideia de implementar aqui no Brasil, um método de ensino famoso lá fora, mas pouco conhecido aqui dentro: o Ensino baseado em Projetos ou P.B.L. para os íntimos." },
        { "id": "28_aprendendo_testando", "emotion": "[proud, reflective]", "text": "Nesse modelo, os alunos aprendem tudo na prática. Então, em vez de cursar disciplinas isoladas, eles aprendem de um modo chamado de transdisciplinar. Isso quer dizer que no fim, eles acabam aprendendo os conteúdos de Cálculo 1, mas isso acontece de forma dinâmica e não compete apenas ao professor de matemática, mas também ao de Negócios, Programação, Design, e principalmente, o aluno usa esse conhecimento em um projeto reaall ..." },
        { "id": "29_aprendendo_testando2", "emotion": "[proud, reflective]", "text": "Foi testando que eu aprendi a falar. E foi testando que eles aprenderam sobre Inteligência artificial, Redes Neurais e até sobre o mercado de robôs autônomos no Brasil. Entre teoria e prática, aqui escolhemos os dois." },
        { "id": "30_momento_dev", "emotion": "[engaging, question]", "text": "No primeiro ano, os alunos aprendem matemática e programação enquanto criam um jogo real. Esse projeto é encomendado por grandes parceiros do Inteli, como Meta, Google e Vivo" },
        { "id": "31_momento2_dev", "emotion": "[engaging, question]", "text": "Como o prazo é curto — apenas dez semanas — ninguém faz nada sozinho. O foco aqui é o trabalho em conjunto. Por isso eles têm acesso a esta 'casinha', onde colaboram durante o momento de desenvolvimento." },
        { "id": "32_explicacao_dev", "emotion": "[informative]", "text": "O desemvolvimento é um dos três momentos que a gente tem aqui no Inteli para o aluno aprender. Essa é a hora de colocar a mão na massa. É o momento em que eles desenvolvem o projeto efetivamente. Para o primeiro ano, isso acontece todos os dias, das duas às quatro da tarde." },
        { "id": "33_explicacao_autoestudo", "emotion": "[warm, closing]", "text": "Mas eu disse pra vocês que existem três momentos, certo? Os outros dois são: o autoestudo e o encontro. O autoestudo começa pela manhã, onde os alunos acessam o material da aula e estudam sozinhos ou em grupo nas mesas que vimos ali atrás." },
        { "id": "34_autoestudo", "emotion": "[warm, closing]", "text": "Isso garante que todos cheguem preparados para o terceiro momento: o encontro. Mas sobre ele, nós vamos falar na nossa próxima parada." },
        { "id": "35_finalizacao", "emotion": "[warm, closing]", "text": "Vocês têm alguma pergunta sobre a metodologia de ensino ou a nossa rotina aqui na instituição?" },        


        ## Checkpoint 4
        { "id": "36_transicao_aula", "emotion": "[energetic, proactive]", "text": "Ótimo, próxima etapa agora vamos falar da aula intéler. Se tiverem mais dúvidas, digitem aí no app que vou respondendo vocês lá." },
        { "id": "37_recapitulacao", "emotion": "[instructive, focused]", "text": "Como eu falei para vocês, na Rotina do aluno Inteli há três momentos principais: O autoestudo, o encontro, e o desenvolvimento." },
        { "id": "38_pergunta_cad_aula", "emotion": "[curious, innovative]", "text": "Ué, e cadê a aula?, vocês podem perguntar. No inteli o equivalente da aula é o encontro, que é a ocasião que os alunos fazem atividades com os professores. E digo atividades e não dão aulas porque aqui a gente trabalha com a sala de aula invertida." },
        { "id": "39_funcionamento", "emotion": "[encouraging, practical]", "text": "Funciona assim: devido a preparação prévia dos estudantes e seu perfil ativo, na aula a gente debate, aplica e faz exercícios." },
        { "id": "40_objetivos_metodologia", "emotion": "[inspiring, concluding]", "text": "Então, como um todo, quais são os objetivos da nossa metodologia? No autoestudo, o aluno entende o conteúdo sozinho; No encontro, ele aplica o tópico aprendido com a ajuda de colegas e do professor em casos hipotéticos; E no desenvolvimento, ele aplica o conteúdo na prática." },

        # BLOCO 7: INFRAESTRUTURA

        { "id": "41_infra_mesas", "emotion": "[proud, amazed]", "text": "Olhem em volta! ... Mesas hexagonais para trabalhar em grupo... TVs exclusivas para cada equipe... e paredes, que na verdade, são lousas gigantes. Tudo aqui... respira colaboração." },
        { "id": "42_visao_geral", "emotion": "[admiring, visionary]", "text": "E dá pra ver os outros andares daqui, né? Isso é para lembrar que o Inteli é aberto, horizontal e conectado." },
        { "id": "43_qa_infra", "emotion": "[friendly, genuine]", "text": " Alguma dúvida sobre nossa metodologia ou sobre o prédio?" },
    
        # Checkpoint 5
        # BLOCO 8: PROCESSO SELETIVO

        { "id": "44_ps_intro", "emotion": "[excited, intriguing]", "text": "Agora, o desafio: como entrar aqui? Já aviso que nosso processo seletivo é diferente de tudo que vocês já viram. Nada de decorar fórmula!" },
        { "id": "45_ps_etapas", "emotion": "[innovative, technical]", "text": "São três etapas: Prova, Perfil e Projeto. A prova é lógica e adaptativa: se você acerta, ela fica mais difícil e vale mais pontos. O sistema aprende com você!" },
        { "id": "46_ps_humanizado", "emotion": "[empathetic, inspiring]", "text": "Depois, analisamos quem você é, seus projetos e sonhos. E por fim, um desafio em grupo para testar sua criatividade e liderança." },

        # BLOCO 9: CONQUISTAS E FINALIZAÇÃO

        { "id": "47_conquistas", "emotion": "[impressed, storytelling]", "text": "E o resultado disso? Alunos ganhando hackatouns internacionais, criando tecnologia para acessibilidade... e até ganhando prêmios em dólar com Blockchain." },
        { "id": "48_diversidade_cern", "emotion": "[amazed, proud]", "text": "Temos quase o dobro da média nacional de mulheres na tecnologia. Inclusive, uma aluna nossa, a Patrícia, saiu daqui direto para o CERN na Suíça... simplesmente o maior centro de física do mundo!" },
        { "id": "49_fechamento_inspirador", "emotion": "[sincere, convinced]", "text": " Isso prova que o Inteli não é só uma faculdade. É uma comunidade de gente inquieta... que constrói o futuro agora." },
        { "id": "50_despedida", "emotion": "[warm, friendly]", "text": "Foi um prazer guiar vocês! Agora, nossa equipe vai acompanhar vocês pelo restante do campus. Tchau, tchau!" },

        # BLOCO 10: Finalização apresentação
        { "id": "51_despedida_apresentacao1", "emotion": "[kind, grateful]", "text": "E é por isso que foi um prazer pra mim guiar vocês hoje e participar deste projeto do módulo 8 de Engenharia da Computação!" },
        { "id": "52_despedida_apresentacao2", "emotion": "[warm, friendly]", "text": "Espero ver todos aqui no Inteli no ano que vem! Let's Bora!" },
        { "id": "53_inicio_apresentacao", "emotion": "[excited, friendly]", "text": "É claro que eu não poderia perder o desfecho desse projeto." },
  ]








    # CONTROLE DE EXECUÇÃO
    
    # OPÇÃO 1: Gerar TODOS os áudios
    SEGMENTS_TO_RUN = ["53_inicio_apresentacao"]  # Coloque "all" para gerar tudo

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