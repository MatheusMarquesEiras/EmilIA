from ollama import Client
import logging
import sys
from dataclasses import dataclass
import mss
from PIL import Image
from typing import Optional

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("VoiceAssistant")

# Funções auxiliares para manter o estilo visual
def log_info(msg): logger.info(f"{Colors.BLUE}[INFO]{Colors.RESET} {msg}")
def log_success(msg): logger.info(f"{Colors.GREEN}[OK]{Colors.RESET} {msg}")
def log_warning(msg): logger.warning(f"{Colors.YELLOW}[AVISO]{Colors.RESET} {msg}")
def log_error(msg): logger.error(f"{Colors.RED}[ERRO]{Colors.RESET} {msg}")
def log_user(msg): logger.info(f"\n{Colors.GREEN}{Colors.BOLD}[USUÁRIO] 🎤 {msg}{Colors.RESET}")
def log_assistant(msg): logger.info(f"\n{Colors.CYAN}{Colors.BOLD}[ASSISTENTE] 🤖 {msg}{Colors.RESET}")
def log_tool(name, res): logger.info(f"{Colors.PURPLE}[TOOL] 🛠️  Usando {name}... Resultado: {res[:50]}...{Colors.RESET}")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_main_url = "http://localhost:11434"
_aux_url = "http://localhost:11434"
_main_model = 'gemma3:4b'
_aux_model = 'func'

@dataclass
class UserMessage:
    message: str
    tag: str = None
    img: Optional[str] = None

    def format(self):
        data = {
            'role': 'user',
            'content': f"{self.message}"
        }
        if self.img is not None:
            data['image_url'] = self.img
        return data

@dataclass
class AIMessage:
    message: str

    def format(self):
        return {
            'role': 'assistant',
            'content': self.message
        }

@dataclass
class System:
    message: str

    def format(self):
        return {
            'role': 'system',
            'content': self.message
        }

@dataclass
class Tool:
    name: str
    result: str

    def format(self):
        return {
            'role': 'tool',
            'tool_name': self.name,
            'content': self.result
        }

    def is_empty(self):
        return self.name == ''

    def __repr__(self):
        return f'{{tool={self.name[:15]}...}}'

@dataclass
class Messages:
    sys: str
    chat: str


class AuxServer:
    def __init__(self, _url: str, _model, sys: str = ''):
        self.client = Client(host=_url, headers={'Authorization': 'Bearer 88a4d679872640ab9347b357354679b8.6wkfkl4cWeENTn0xq4V_uQee'})
        self.model = _model
        self.filename = "screenshot.jpg"

        # Tenta carregar system prompt do arquivo, se não existir usa padrão
        try:
            with open('sys_aux.txt', 'r', encoding='utf-8') as f:
                self.system = System(message=f.read())
        except FileNotFoundError:
            log_warning("Arquivo sys_aux.txt não encontrado. Usando system prompt padrão.")
            default_sys = """Você é um assistente inteligente que usa ferramentas para responder perguntas.

REGRAS IMPORTANTES:
1. Para pesquisas na internet, use a função web_search com a pergunta ORIGINAL do usuário
2. Para matemática, preste atenção nas palavras-chave:
   - "mais", "some", "adicione" → use add_numbers
   - "menos", "subtraia", "diminua" → use subtract_numbers  
   - "vezes", "multiplique", "produto" → use multiply_numbers
3. Use get_temperature apenas para perguntas sobre temperatura/clima
4. Use get_exchange_rate apenas para conversão de moedas
5. Use take_screenshot quando o usuário pedir para ver a tela

Seja preciso na escolha das ferramentas!"""
            self.system = System(message=default_sys)

        # Dicionário de funções executáveis (para chamar depois que o modelo escolher)
        self.tools_dict = {
            'add_numbers': self.add_numbers,
            'subtract_numbers': self.subtract_numbers,
            'multiply_numbers': self.multiply_numbers,
            'take_screenshot': self.take_screenshot,
            'web_search': self.web_search,
            'get_temperature': self.get_temperature,
            'get_exchange_rate': self.get_exchange_rate
        }

    def get_tool_list(self):
        """Retorna lista de funções Python (Ollama converte automaticamente)"""
        return [
            self.add_numbers,
            self.subtract_numbers,
            self.multiply_numbers,
            self.take_screenshot,
            self.web_search,
            self.get_temperature,
            self.get_exchange_rate
        ]

    def get_tool_dict(self):
        """Retorna o dicionário de funções executáveis"""
        return self.tools_dict

    def pull(self):
        """Baixa o modelo auxiliar explicitamente."""
        if self.model:
            try:
                log_info(f"Baixando/Verificando modelo auxiliar: {self.model} (Porta Aux)...")
                self.client.pull(model=self.model)
                self.client.pull(model=_aux_model)
                log_success(f"Modelo auxiliar {self.model} pronto.")
            except Exception as e:
                log_error(f"Falha ao baixar modelo auxiliar: {e}")
            return

    def generate_response(self, message: str):
        """Gera resposta usando o modelo e ferramentas."""
        user_message = UserMessage(message=message, tag='[test]')
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[self.system.format(), user_message.format()],
                tools=self.get_tool_list()  # Passa funções Python diretamente!
            )
        except Exception as e:
            log_error(f"Erro ao gerar resposta do modelo: {e}")
            return False, None

        msg = response.message

        if not msg.tool_calls:
            return False, None

        call = msg.tool_calls[0]
        tool_name = call.function.name
        args = call.function.arguments or {}

        # Debug: ver qual ferramenta foi escolhida
        log_info(f"🎯 Ferramenta escolhida: {tool_name} | Argumentos: {args}")

        if tool_name not in self.get_tool_dict():
            log_warning(f"⚠️ Ferramenta '{tool_name}' não encontrada no dicionário!")
            return False, None

        try:
            result = self.tools_dict[tool_name](**args)
        except Exception as e:
            log_error(f"Erro ao executar ferramenta {tool_name}: {e}")
            return False, None

        return True, {
            "tool_name": tool_name,
            "arguments": args,
            "result": result,
        }

    # ==================== TOOLS COM GOOGLE-STYLE DOCSTRINGS ====================

    def web_search(self, query: str) -> list[str]:
        """Realiza pesquisas na internet sobre informações atuais e em tempo real.

        Use esta ferramenta quando o usuário perguntar sobre eventos recentes,
        notícias, informações atualizadas ou qualquer coisa que não esteja no
        seu conhecimento base.

        Args:
            query: A pergunta ou termo a ser pesquisado na internet. Mantenha a
                pergunta original do usuário. Exemplos: "atual presidente dos
                Estados Unidos", "notícias tecnologia 2026", "cotação bitcoin hoje"

        Returns:
            Uma lista contendo uma string com o resultado sumarizado da pesquisa.
        """
        # try:
        log_info(f"🔍 Realizando busca web: {query}")
        response = self.client.web_search(query=query, max_results=5)
            
        log_info('pesquisando...')
        results = response.get('results', [])
        log_info('terminou')

        return 'terminou'
        #     if not results:
        #         log_warning("Nenhum resultado encontrado na busca")
        #         return ["Nenhum resultado encontrado para esta pesquisa."]

        #     # Extrai conteúdo dos resultados
        #     content_list = []
        #     for idx, item in enumerate(results, 1):
        #         title = item.get('title', 'Sem título')
        #         content = item.get('content', '')
        #         content_list.append(f"[Resultado {idx}] {title}: {content}")

        #     # Sumariza os resultados
        #     sys_msg = f'''Você é um assistente que resume resultados de pesquisa.

        #         Pergunta do usuário: {query}

        #         Resultados encontrados:
        #         {chr(10).join(content_list)}

        #         Instruções:
        #         - Responda a pergunta do usuário de forma direta e objetiva
        #         - Use informações dos resultados acima
        #         - Seja conciso e claro
        #         - Se houver informações conflitantes, mencione
        #     '''

        #     answer = self.client.chat(
        #         model=_aux_model,
        #         messages=[{'role': 'user', 'content': sys_msg}]
        #     )
            
        #     result_text = str(answer['message']['content']).strip()
        #     log_success(f"✅ Busca concluída: {result_text[:100]}...")
        #     return [result_text]
            
        # except Exception as e:
        #     log_error(f"Erro na busca web: {e}")
        #     return [f"Erro ao realizar busca: {str(e)}"]

    @staticmethod
    def add_numbers(a: int, b: int) -> int:
        """Soma dois números inteiros.

        Use esta ferramenta quando o usuário pedir para SOMAR, ADICIONAR,
        ou usar palavras como "mais", "some", "adicione".

        Args:
            a: Primeiro número a ser somado
            b: Second número a ser somado

        Returns:
            A soma de a e b
        """
        return int(a) + int(b)

    @staticmethod
    def subtract_numbers(a: int, b: int) -> int:
        """Subtrai o segundo número do primeiro (a - b).

        Use esta ferramenta quando o usuário pedir para SUBTRAIR, DIMINUIR,
        ou usar palavras como "menos", "tire", "remova", "subtraia".

        Args:
            a: Número do qual será subtraído (minuendo)
            b: Número a ser subtraído (subtraendo)

        Returns:
            A diferença entre a e b (a - b)
        """
        return int(a) - int(b)

    @staticmethod
    def multiply_numbers(a: int, b: int) -> int:
        """Multiplica dois números inteiros (a × b).

        Use esta ferramenta quando o usuário pedir para MULTIPLICAR ou usar
        palavras como "vezes", "multiplicado por", "produto de".

        Args:
            a: Primeiro número a ser multiplicado
            b: Segundo número a ser multiplicado

        Returns:
            O produto de a e b (a × b)
        """
        return int(a) * int(b)

    def take_screenshot(self) -> str:
        """Captura uma imagem da tela atual do usuário.

        Use esta ferramenta quando o usuário solicitar para VER, ANALISAR
        ou DESCREVER o que está na tela dele.

        Returns:
            Uma mensagem confirmando que a screenshot foi capturada ou uma
            mensagem de erro caso algo falhe.
        """
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                sct_img = sct.grab(monitor)
                img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)
                img.save(self.filename, quality=95)
            log_success("📸 Screenshot capturada com sucesso")
            return 'imagem tirada com sucesso'
        except Exception as e:
            log_error(f"Erro ao tirar print: {e}")
            return f"Error taking screenshot: {e}"

    @staticmethod
    def get_temperature(city: str) -> str:
        """Obtém a temperatura atual de uma cidade específica.

        Use esta ferramenta APENAS quando o usuário perguntar sobre
        TEMPERATURA ou CLIMA de uma cidade.

        Args:
            city: Nome da cidade em português ou inglês. Exemplos: "São Paulo",
                "London", "Tokyo", "Nova York"

        Returns:
            A temperatura da cidade como string (ex: "26°C") ou "Desconhecido"
            se a cidade não estiver no banco de dados.
        """
        temperatures = {
            "New York": "22°C",
            "Nova York": "22°C",
            "London": "15°C",
            "Londres": "15°C",
            "Tokyo": "18°C",
            "Tóquio": "18°C",
            "São Paulo": "26°C",
            "Sao Paulo": "26°C",
        }
        return temperatures.get(city, "Desconhecido")

    @staticmethod
    def get_exchange_rate(base: str, quote: str) -> str:
        """Obtém a taxa de câmbio entre duas moedas.

        Use esta ferramenta quando o usuário perguntar sobre CONVERSÃO DE
        MOEDAS ou VALOR de uma moeda em outra.

        Args:
            base: Código ISO 4217 de 3 letras da moeda base. Exemplos: "USD",
                "EUR", "BRL"
            quote: Código ISO 4217 de 3 letras da moeda de cotação. Exemplos:
                "BRL", "USD", "EUR"

        Returns:
            A taxa de câmbio como string ou "Desconhecido" se o par de moedas
            não estiver disponível.
        """
        rates = {
            ("USD", "BRL"): "5.10",
            ("EUR", "BRL"): "5.55",
            ("USD", "EUR"): "0.92",
            ("BRL", "USD"): "0.20",
            ("EUR", "USD"): "1.09",
        }
        return rates.get((base.upper(), quote.upper()), "Desconhecido")


if __name__ == '__main__':
    cli = AuxServer(_url=_aux_url, _model=_aux_model)
    # cli.pull()
    
    tests = [
        "Qual a temperatura em São Paulo?",
        "Quanto é 1 USD em BRL?",
        "Me explique o que é uma função hash.",
        "olá como vai?",
        'consegue olhar minha tela?',
        'pesquise para mim quem é o atual presidente dos Estados Unidos?',
        'quanto é 15 mais 27?',
        'multiplique 8 por 9',
        'subtraia 100 menos 35',
        'some 50 e 75'
    ]
    
    print(f'\n{Colors.PURPLE}{Colors.BOLD}{"="*70}{Colors.RESET}')
    print(f'{Colors.PURPLE}{Colors.BOLD}🤖  INICIANDO TESTES DO ASSISTENTE EMILIA{Colors.RESET}')
    print(f'{Colors.PURPLE}{Colors.BOLD}{"="*70}{Colors.RESET}\n')
    
    for i in tests:
        print(f'\n{Colors.CYAN}{"="*70}{Colors.RESET}')
        print(f'{Colors.YELLOW}❓ Pergunta: {i}{Colors.RESET}')
        print(f'{Colors.CYAN}{"="*70}{Colors.RESET}')
        result = cli.generate_response(message=i)
        print(f'{Colors.GREEN}✅ Resultado: {result}{Colors.RESET}')
    
    print(f'\n{Colors.PURPLE}{Colors.BOLD}{"="*70}{Colors.RESET}')
    print(f'{Colors.PURPLE}{Colors.BOLD}🏁  TESTES FINALIZADOS{Colors.RESET}')
    print(f'{Colors.PURPLE}{Colors.BOLD}{"="*70}{Colors.RESET}\n')