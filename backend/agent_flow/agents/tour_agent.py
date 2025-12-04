import json
from typing import Dict


class TourAgent:
    """
    Agente responsável por gerenciar o roteiro do tour e o estado da navegação.
    O roteiro segue o script 'Cão Robô' do Inteli.
    """

    def __init__(self):
        # Definição do Roteiro (Script)
        # Cada 'stop' tem um ID, o local sugerido e a fala do robô.
        self.script = [
            {
                "id": "historia_bolsas",
                "local": "Recepção / Painel de Doadores",
                "fala": """Que alegria receber vocês aqui hoje [latido]. Sejam bem-vindos à minha casa, o Inteli, ou para nossos fundadores, o “MIT Brasileiro” [latido].

O Inteli foi fundado em 2019, resultado de uma conversa do Roberto Sallouti no Vale do Silício. A ideia surgiu quando questionaram por que se investia pouco no Brasil, e a resposta foi: "Porque o Brasil não forma engenheiros suficientes".
Então, Roberto Sallouti e André Esteves decidiram deixar um legado: "Nós vamos formar esses engenheiros". Daí surgiu o Inteli. Um legado de brasileiros para brasileiros.

O Inteli tem a missão de “Formar os futuros líderes que vão transformar o Brasil através da tecnologia”.
Acreditamos que esses líderes vêm de diversos contextos, por isso temos o maior programa de bolsas de estudo do ensino superior do Brasil, oferecendo auxílio-moradia, alimentação, transporte, curso de inglês e notebook.

Isso só foi possível graças aos doadores-parceiros que investiram no desenvolvimento desses alunos. Aqui em cima nesse painel [latido], vocês podem ver os nomes deles.

Vocês têm alguma pergunta sobre a história ou o programa de bolsas do Inteli?""",
                "acao_navegacao": "ir_para_area_academica",
            },
            {
                "id": "cursos_clubes",
                "local": "Área Acadêmica / Convivência",
                "fala": """E agora, [latido], chegou a hora de falar sobre os cursos e clubes!
Aqui no Inteli, temos cinco graduações: Engenharia da Computação, Ciência da Computação, Engenharia de Software, Sistemas de Informação e Administração Tech. [latido]
E adivinhem só... os alunos de Engenharia de Computação são os mais legais [latido]! (Brincadeira, é que eles me programaram).

Cada curso tem seu jeito: Eng. Computação integra hardware e software; Ciência da Computação é a base de tudo (algoritmos e IA); Eng. Software foca em construir grandes sistemas; Sistemas de Informação conecta tech e estratégia; e ADMTech une gestão e tecnologia para empreendedores.

Mas a mágica também acontece nos clubes estudantis! São mais de 20 grupos liderados por alunos.
Tem a Tantera (atlética), a Inteli Júnior, a LEI (Liga de Empreendedorismo), a AgroTech, o Game Lab, o Inteli Blockchain e a Inteli Academy (focada em IA... tipo eu).
Além disso, temos coletivos que tornam o Inteli acolhedor e diverso: Grace Hopper (Feminino), Benedito Caravelas (Negro) e Turing (LGBTQIAPN+), além da Wave que ajuda vestibulandos.

Se quiserem saber mais sobre algum curso ou clube específico, podem perguntar agora!""",
                "acao_navegacao": "ir_para_atelies",
            },
            {
                "id": "pbl_rotina",
                "local": "Corredor dos Ateliês",
                "fala": """Agora vamos falar sobre ser um aluno Inteli. Tudo começa com um espanto: [latidos] O Inteli não tem matérias isoladas!

Nossos fundadores trouxeram o método PBL (Ensino baseado em Projetos). Aqui, aprendemos testando!
Em vez de estudar Cálculo I isolado, o aluno aprende de forma transdisciplinar, cruzando Negócios, Programação e Design para resolver um problema real de uma empresa parceira.
"Os alunos não cursam disciplina, eles têm um problema para resolver, e aprendem coisas a serviço desse projeto".

No primeiro ano, por exemplo, aprendem matemática e física enquanto criam jogos para empresas como Meta, Google e Vivo.
E para não fazer isso sozinhos em 10 semanas, eles trabalham em grupo.

Na rotina, temos 3 momentos:
1. Autoestudo: O aluno estuda o tópico do dia sozinho antes da aula.
2. Encontro: A aula dinâmica com professores.
3. DEV: O momento "mão na massa" (geralmente das 14h às 16h) para desenvolver o projeto.

Alguma dúvida sobre nossa metodologia ou rotina?""",
                "acao_navegacao": "entrar_no_atelie",
            },
            {
                "id": "sala_invertida_infra",
                "local": "Dentro do Ateliê (Sala de Aula)",
                "fala": """Vocês devem estar se perguntando: "E cadê a aula?".
Aqui o equivalente à aula é o Encontro. Trabalhamos com Sala de Aula Invertida.
Em vez de o professor palestrar por duas horas, o encontro promove debate e exercício, já que o aluno se preparou no Autoestudo.

Nossa infraestrutura foi pensada para isso:
1. Mesas hexagonais: Para grupos de 8 alunos olharem no olho um do outro.
2. Televisões em cada mesa: [latido] Para cada grupo projetar seu trabalho, não só o professor.
3. Lousas por toda a sala: Para estimular discussões.
E a visão aberta dos andares mostra que o Inteli é colaborativo e horizontal.

Querem perguntar algo sobre a sala invertida ou nossa infraestrutura?""",
                "acao_navegacao": "ir_para_hub_comunidade",
            },
            {
                "id": "processo_seletivo",
                "local": "Hub de Comunidade / Final",
                "fala": """[Tom empolgado] Chegou a hora de falar do Processo Seletivo!
Esqueça decorar fórmulas. Aqui queremos ver raciocínio e potencial.
São 3 Eixos:
1. Prova: Matemática e Lógica. É adaptativa (o sistema aprende com você). 24 questões, responde 20.
2. Perfil: Queremos conhecer você. Duas redações e envio de atividades extracurriculares.
3. Projeto: Dinâmica em grupo online para avaliar comunicação e colaboração.

O resultado? Alunos premiados no Brasil e no mundo!
Tivemos vencedores do maior hackathon de IA da América Latina, projetos de acessibilidade com equipamentos apreendidos, e o clube de Blockchain ganhando prêmios globais.
E temos 27% de mulheres nas graduações (quase o dobro da média). Uma aluna da primeira turma, a Patrícia Honorato, foi até selecionada para o CERN na Suíça! [latido surpreso].

O Inteli é uma comunidade de gente inquieta que constrói o futuro.
Se quiserem saber mais sobre o vestibular ou bolsas, estou à disposição!""",
                "acao_navegacao": "finalizar_tour",
            },
            {
                "id": "conclusao",
                "local": "Saída",
                "fala": """E agora, após falarmos sobre a vida inteler, queria agradecer a visita e a companhia nesse tour [latido].
Agora, um dos nossos colaboradores humanos vai acompanhá-los pelo restante da instituição.
Tchau, tchau! [latido]""",
                "acao_navegacao": "parar",
            },
        ]

        # Estado atual do Tour
        self.current_step_index = -1
        self.is_active = False

    def start_tour(self) -> str:
        """Inicia o tour do zero."""
        self.current_step_index = 0
        self.is_active = True
        step = self.script[self.current_step_index]
        return self._format_response(step)

    def next_step(self) -> str:
        """Avança para o próximo ponto do roteiro."""
        if not self.is_active:
            return json.dumps(
                {
                    "speech": "O tour não está ativo. Diga 'iniciar tour' para começar.",
                    "action": "aguardar",
                },
                ensure_ascii=False,
            )

        self.current_step_index += 1

        if self.current_step_index >= len(self.script):
            self.is_active = False
            self.current_step_index = -1
            return json.dumps(
                {
                    "speech": "O roteiro do tour foi finalizado. Foi um prazer!",
                    "action": "tour_finalizado",
                },
                ensure_ascii=False,
            )

        step = self.script[self.current_step_index]
        return self._format_response(step)

    def stop_tour(self) -> str:
        """Interrompe o tour imediatamente."""
        self.is_active = False
        self.current_step_index = -1
        return json.dumps(
            {
                "speech": "Tour interrompido. Voltando ao modo de espera.",
                "action": "parar",
            },
            ensure_ascii=False,
        )

    def _format_response(self, step: Dict) -> str:
        """Formata a resposta para o Orquestrador como JSON."""
        response = {
            "agent": "tour_agent",
            "location": step["local"],
            "speech": step["fala"],
            "action": step["acao_navegacao"],
            "status": "active",
        }
        return json.dumps(response, ensure_ascii=False)

    def process_command(self, command: str) -> str:
        """Processa comandos diretos de navegação."""
        cmd = command.lower()

        if any(x in cmd for x in ["iniciar", "começar", "vamos lá", "start"]):
            return self.start_tour()
        elif any(
            x in cmd for x in ["próximo", "continuar", "seguir", "next", "avançar"]
        ):
            return self.next_step()
        elif any(x in cmd for x in ["parar", "encerrar", "stop"]):
            return self.stop_tour()
        else:
            # Se o tour está ativo e o usuário fala algo que não é comando de navegação
            # O orquestrador deve decidir se é uma pergunta para o RAG ou se repete o passo.
            # Aqui retornamos o passo atual para manter o contexto.
            if self.is_active:
                step = self.script[self.current_step_index]
                return self._format_response(step)
            else:
                return json.dumps(
                    {
                        "speech": "Não estou em um tour ativo. Diga 'iniciar tour' se quiser começar.",
                        "action": "aguardar",
                    },
                    ensure_ascii=False,
                )


def create_tour_agent():
    return TourAgent()
