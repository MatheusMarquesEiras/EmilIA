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
    handlers=[logging.StreamHandler(sys.stdout)] # Apenas console, sem arquivo
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
_aux_model = 'functiongemma'
# _aux_aux_model = 'llama3.2:3b'

@dataclass
class UserMessage:
    message: str
    tag: str = None
    img: Optional[str] = None

    def format(self):
        data =  {
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

# futura implementação para as mensagens trocadas
@dataclass
class Messages:
    sys: str
    chat: str

class AuxServer:
    def __init__(self, _url: str, _model, sys: str = ''):
        self.client = Client(host=_url, headers={'Authorization': 'Bearer 88a4d679872640ab9347b357354679b8.6wkfkl4cWeENTn0xq4V_uQee'})
        self.model = _model
        self.filename = "screenshot.jpg"

        with open('sys_aux.txt', 'r', encoding='utf-8') as f:
            self.system = System(message=f.read())

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
        return list(self.tools_dict.values())
    
    def get_tool_dict(self):
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
        user_message = UserMessage(message=message, tag='[test]')
        response = self.client.chat(model=self.model, messages=[self.system.format(), user_message.format()], tools=self.get_tool_list())

        msg = response.message
    
        if not msg.tool_calls:
            return False, None

        call = msg.tool_calls[0]
        tool_name = call.function.name
        args = call.function.arguments or {}

        if tool_name not in self.get_tool_dict():
            return False, None

        result = self.tools_dict[tool_name](**args)

        return True, {
            "tool_name": tool_name,
            "arguments": args,
            "result": result,
        }

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
        try:
            response = self.client.web_search(query=query, max_results=2)
            content_list = [item['content'] for item in response.get('results', [])]
            sys_msg = f'Você vai receber a seguinte entrada "{content_list}" que deve ser sumarizado para responder a seguinte pergunta de forma mais direta possivel "{query}" mas sempre que for perguntada sobre algo atual você deve pesquisar na internet utilizando a função "web_search" e fornecer a ela a querry para a pesquisa'
            answer = self.client.chat(model=_aux_model, messages=[{'role': 'user', 'content': sys_msg}])
            return [str(answer['message']['content']).strip()]
        except Exception as e:
            return [f"Search error: {str(e)}"]

    @staticmethod
    def add_numbers(a: int, b: int) -> list[int]:
        """Soma dois números inteiros.

        Args:
            a: O primeiro número inteiro a ser somado.
            b: O segundo número inteiro a ser somado.

        Returns:
            A soma dos dois números inteiros.
        """
        return [int(a) + int(b)]

    @staticmethod
    def subtract_numbers(a: int, b: int) -> list[int]:
        """Subtrai o segundo número inteiro do primeiro.

        Args:
            a: O número inteiro do qual será subtraído.
            b: O número inteiro a ser subtraído.

        Returns:
            A diferença entre os dois números inteiros.
        """
        return [int(a) - int(b)]

    @staticmethod
    def multiply_numbers(a: int, b: int) -> list[int]:
        """Multiplica dois números inteiros.

        Args:
            a: O primeiro número inteiro a ser multiplicado.
            b: O segundo número inteiro a ser multiplicado.

        Returns:
            O produto dos dois números inteiros.
        """
        return [int(a) * int(b)]

    def take_screenshot(self) -> list[str]:
        """Permite olhar a tela do usuário quando solicitado

        Returns:
            A descrição detalhada da imagem gerada pela IA auxiliar, 
            ou uma mensagem de erro se a captura ou a chamada da IA falhar.
        """
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                sct_img = sct.grab(monitor)
                img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)
                img.save(self.filename, quality=95)
        except Exception as e:
            log_error(f"Erro ao tirar print: {e}")
            return f"Error taking screenshot: {e}"
        
        return 'imagem tirada com sucesso'
        
    def get_temperature(self, city: str) -> str:
        """Obtém a temperatura atual de uma cidade.

        Args:
        city: O nome da cidade (ex: "São Paulo")

        Returns:
        A temperatura atual da cidade como string.
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


    def get_exchange_rate(self, base: str, quote: str) -> str:
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
        }
        return rates.get((base.upper(), quote.upper()), "Desconhecido")
    

if __name__ == '__main__':
    cli = AuxServer(_url=_aux_url, _model=_aux_model)
    cli.pull()
    tests = [
        "Qual a temperatura em São Paulo?",
        "Quanto é 1 USD em BRL?",
        "Me explique o que é uma função hash.",
        "olá como vai?",
        'consegue olhar minha tela?',
        'pesquise para mim quem é o atual presedente dos Estados Unidos?'
    ]
    for i in tests:
        print('<=======>')
        print(i)
        print(cli.generate_response(message=i))