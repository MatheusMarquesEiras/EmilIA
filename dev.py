from ollama import Client
from dataclasses import dataclass
from typing import Optional

import mss
import logging

from PIL import Image
import sys

_main_url = "http://localhost:11434"
_aux_url = "http://localhost:11435"
_main_model = 'llama3.1:8b'
_aux_model = 'granite3.2-vision'

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

# Configuração do Logger Padrão do Python
# format='%(message)s' -> Removemos data/hora para ficar limpo no console como você gosta
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

@dataclass
class UserMessage:
    message: str
    tag: str
    img: Optional[str] = None

    def format(self):
        data =  {
            'role': 'user',
            'content': f"{self.tag} {self.message}"
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
    tool: str

    def format(self):
        return {
            'role': 'tool',
            'content': self.tool
        }

    def is_empty(self):
        return self.tool == ''
    
    def __repr__(self):
        return f'{{tool={self.tool[:15]}...}}'

class Aux:
    def __init__(self):
        self.client = Client(host=_aux_url)
        self. _aux_model = _aux_model
        self._sys_prompt = System(message="Você é uma inteligencia artificila que vai verificar se deve ou não chamar uma função e vai retornar um texto de acordo com o contexto que receber")

        self.tools_list = {
            'pass_turn': self.pass_turn,
            'add_numbers': self.add_numbers,
            'subtract_numbers': self.subtract_numbers,
            'multiply_numbers': self.multiply_numbers,
            'take_screenshot': self.take_screenshot,
        }

        self.tools = [
            self.pass_turn,
            self.add_numbers,
            self.subtract_numbers,
            self.multiply_numbers,
            self.take_screenshot,
        ]

    def get_tool_list(self):
        return self.tools_list
    
    def get_tools(self):
        return self.tools
    
    def pull(self):
        log_info('iniciando pull...')
        self.client.pull(self._aux_model)
        log_info('fim do pull...')
    
    def get_response(self, input: str, tag="[Professor]"):
        user_message = UserMessage(message=input, tag=tag)
        response = self.client.chat(model=self._aux_model, messages=[self._sys_prompt.format(), user_message.format()], tools=self.get_tools())

        if response.message.tool_calls:
            log_info(response.message.tool_calls)

        return response['message']['content']
    
    @staticmethod
    def pass_turn() -> str:
        """Indica ao modelo que ele deve pular a vez e não usar nenhuma ferramenta.

        Returns:
            Uma string vazia.
        """
        return ''

    @staticmethod
    def add_numbers(a: int, b: int) -> int:
        """Adiciona dois números inteiros.

        Args:
            a: O primeiro número inteiro a ser somado.
            b: O segundo número inteiro a ser somado.

        Returns:
            A soma dos dois números inteiros.
        """
        return int(a) + int(b)

    @staticmethod
    def subtract_numbers(a: int, b: int) -> int:
        """Subtrai o segundo número inteiro do primeiro.

        Args:
            a: O número inteiro do qual será subtraído.
            b: O número inteiro a ser subtraído.

        Returns:
            A diferença entre os dois números inteiros.
        """
        return int(a) - int(b)

    @staticmethod
    def multiply_numbers(a: int, b: int) -> int:
        """Multiplica dois números inteiros.

        Args:
            a: O primeiro número inteiro a ser multiplicado.
            b: O segundo número inteiro a ser multiplicado.

        Returns:
            O produto dos dois números inteiros.
        """
        return int(a) * int(b)

    def take_screenshot(self) -> str:
        """Tira uma captura de tela (screenshot) da tela principal, salva em um arquivo e, em seguida,
        envia a imagem para um modelo auxiliar de IA para obter uma descrição detalhada em português do Brasil.

        Returns:
            A descrição detalhada da imagem gerada pela IA auxiliar, ou uma mensagem de erro se a captura
            de tela ou a chamada à IA falhar.
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
        
        try:
            res = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Descreva esta imagem em um longo paragrafo em português do Brasil:', 
                        'images': [f'{self.filename}']
                    }
                ]
            )
            return res['message']['content']
        except Exception as e:
            log_error(f"Erro na IA Auxiliar: {e}")
            return f"Auxiliary AI Error: {e}"
    
if __name__ == "__main__":
    aux = Aux()
    aux.pull()
    response = aux.get_response(input=input('digite algo: '))

    print(response)
    