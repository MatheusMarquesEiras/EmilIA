import logging
import sys
import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional

# Bibliotecas de Áudio e IA
from ollama import Client
import wave
from piper import PiperVoice, SynthesisConfig
import soundfile as sf
import sounddevice as sd
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# Utilitários Visuais
import emoji
import mss
from PIL import Image

# ===== CONFIGURAÇÃO DE CORES E LOGGING =====

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

# ===== CONFIGURAÇÕES GERAIS =====
_main_url = "http://localhost:11434"
_aux_url = "http://localhost:11435"
_main_model = 'llama3.1:8b'
_aux_model = 'granite3.2-vision'

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
    
# ===== FERRAMENTAS =====

class AuxServer:
    def __init__(self, _url: str, _model, sys: str = ''):
        self.client = Client(host=_url, headers={'Authorization': 'Bearer 88a4d679872640ab9347b357354679b8.6wkfkl4cWeENTn0xq4V_uQee'})
        self.model = _model
        self.filename = "screenshot.jpg"

        self.tools_list = {
            'pass_turn': self.pass_turn,
            'add_numbers': self.add_numbers,
            'subtract_numbers': self.subtract_numbers,
            'multiply_numbers': self.multiply_numbers,
            'take_screenshot': self.take_screenshot,
            'web_search': self.web_search,
        }

        self.tools = [
            self.pass_turn,
            self.add_numbers,
            self.subtract_numbers,
            self.multiply_numbers,
            self.take_screenshot,
            self.web_search,
        ]

    def get_tool_list(self):
        return self.tools_list
    
    def get_tools(self):
        return self.tools
    
    def pull(self):
        """Baixa o modelo auxiliar explicitamente"""
        if self.model:
            try:
                log_info(f"Baixando/Verificando modelo auxiliar: {self.model} (Porta Aux)...")
                self.client.pull(model=self.model)
                log_success(f"Modelo auxiliar {self.model} pronto.")
            except Exception as e:
                log_error(f"Falha ao baixar modelo auxiliar: {e}")
            return

    def web_search(self, query: str) -> list[str]:
        try:
            response = self.client.web_search(query=query, max_results=2)
            return response.get('results', [])
        except Exception as e:
            return [f"Search error: {str(e)}"]

    @staticmethod
    def pass_turn() -> str:
        return ''

    @staticmethod
    def add_numbers(a: int, b: int) -> int:
        return int(a) + int(b)

    @staticmethod
    def subtract_numbers(a: int, b: int) -> int:
        return int(a) - int(b)

    @staticmethod
    def multiply_numbers(a: int, b: int) -> int:
        return int(a) * int(b)

    def take_screenshot(self) -> str:
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

class MainServer:
    def __init__(self, _main_url: str, _aux_url: str, _main_model: str, _aux_model: str, sys: str = ''):
        self.client = Client(host=_main_url)
        self.model = _main_model
        self.aux_server = AuxServer(_url=_aux_url, _model=_aux_model)

        try:
            if not sys:
                with open('sys.txt', 'r', encoding='utf-8') as file:
                    sys = file.read()
        except FileNotFoundError:
            log_warning("Arquivo sys.txt não encontrado. Usando system prompt padrão.")
            sys = "Você é um assistente útil."

        self.sysMessage = System(sys)
        self.messages: list[dict] = [self.sysMessage.format()]

    def stream(self, _messages_formated: str):
        return  self.client.chat(self.model, messages=_messages_formated, stream=True, tools=self.aux_server.get_tools())
    
    def pull(self, _model: str = None):
        target_model = _model if _model else self.model
        try:
            log_info(f"Baixando/Verificando modelo principal: {target_model}...")
            self.client.pull(model=target_model)
            log_success(f"Modelo principal {target_model} pronto.")
            self.aux_server.pull()
        except Exception as e:
            log_error(f"Erro ao baixar modelos: {e}")
    
    def generate_test(self):
        user_message = UserMessage(message='[Test] mensagem para testar se você esta funcionando responda se sim ou não')
        log_info(f'Aquecendo modelo com: "{user_message.message}"')
        list_tkns = []
        messages_to_send = [self.sysMessage.format(), user_message.format()]
        log_info(f'mensagem para aquecimento: "{messages_to_send}"')
        try:
            print(f"Iniciando: {Colors.CYAN}", end='')
            for part in self.stream(messages_to_send):
                log_info(part)
                content = part.message.content
                if content:
                    list_tkns.append(content)
                    print(content, end='', flush=True)
            print(f"Finalizando stream{Colors.RESET}")
        except Exception as e:
            log_error(f"Erro no teste: {e}")
        log_success('Aquecimento concluído.')
        return ''.join(list_tkns)

    def generate(self, message):
        user_message = UserMessage(message=message)
        self.messages.append(user_message.format())
        list_tkns = []
        
        # Flag para saber se o modelo decidiu ficar quieto
        used_pass_turn = False
        
        try:
            stream_generator = self.stream(self.messages)
            
            try:
                first_chunk = next(stream_generator)
            except StopIteration:
                log_warning("Modelo não retornou nada no primeiro chunk.")
                return "..."

            if first_chunk.message.tool_calls:
                for tool in first_chunk.message.tool_calls:
                    tool_name = tool['function']['name']
                    
                    # Verifica se a ferramenta é pass_turn
                    if tool_name == 'pass_turn':
                        used_pass_turn = True
                        log_tool(tool_name, "(Silêncio)")
                        tool_obj = Tool("") 
                        self.messages.append(tool_obj.format())
                        continue

                    # Executa outras ferramentas
                    tool_to_call: callable = self.aux_server.get_tool_list().get(tool_name)
                    if tool_to_call:
                        try: 
                            result = tool_to_call(**tool.function.arguments)
                            tool_obj = Tool(str(result))
                            log_tool(tool_name, str(result))
                            
                            if not tool_obj.is_empty():
                                self.messages.append(tool_obj.format())
                        except Exception as e:
                            log_error(f'Erro na tool {tool_name}: {e}')
                
                # Recria o stream (O modelo pode decidir falar algo depois da tool ou não)
                stream_generator = self.stream(self.messages)
            else:
                if first_chunk.message.content:
                    list_tkns.append(first_chunk.message.content)
                    print(f"{Colors.CYAN}{first_chunk.message.content}", end='', flush=True)

            # Consome o resto
            print(f"{Colors.CYAN}", end='') 
            for part in stream_generator:
                content = part.message.content
                if content:
                    list_tkns.append(content)
                    print(content, end='', flush=True)
            print(f"{Colors.RESET}")
            print() 
            
            full_text = emoji.replace_emoji(''.join(list_tkns), replace="").strip()
            
            m2 = AIMessage(full_text)
            self.messages.append(m2.format())
            
            # LÓGICA CORRIGIDA:
            if not full_text:
                if used_pass_turn:
                    log_info("IA decidiu passar a vez (Silêncio intencional).")
                    return None # Retorna None para avisar o VoiceAssistant para não usar TTS
                
                log_warning("IA retornou resposta vazia inesperada.")
                return "Não entendi."
                
            return full_text

        except Exception as e:
            log_error(f"Erro crítico na geração: {e}")
            return "Erro no sistema."

class TTS:
    def __init__(self):
        self.num = ''
        try:
            self.voice = PiperVoice.load(f"controle.onnx")
            self.syn_config = SynthesisConfig(volume=0.5, length_scale=1.3)
            try:
                self.device_id = sd.query_devices('CABLE Input (VB-Audio Virtual Cable), Windows WASAPI')['index']
            except:
                log_warning("Virtual Cable não achado. Usando saída padrão.")
                self.device_id = sd.default.device[1]

            self.output_file = f"test{self.num}.wav"
        except Exception as e:
            log_error(f"Erro TTS Init: {e}")
    
    def generate(self, text):
        if not text or not text.strip():
            log_warning("TTS recebeu texto vazio.")
            return

        try:
            with wave.open(self.output_file, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)
            
            audio_data, samplerate = sf.read(self.output_file)
            sd.play(audio_data, samplerate=samplerate, device=self.device_id)
            sd.wait()
        except Exception as e:
            log_error(f"Erro TTS Play: {e}")


class RealTimeSTT:
    def __init__(self, model_size="small", device="auto", language="pt", input_device_index=None, callback=None):
        self.CHUNK = 1024 
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.ENERGY_THRESHOLD = 300
        self.PAUSE_THRESHOLD = 0.8
        self.SILENCE_CHUNKS = int(self.PAUSE_THRESHOLD * self.RATE / self.CHUNK)
        
        self.audio_queue = queue.Queue()
        self.running = False
        self.producer_thread = None
        self.consumer_thread = None
        self.input_device_index = input_device_index
        self.callback = callback

        log_info(f"Carregando Whisper ({model_size})...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type="int8")
            self.language = language
            log_success("Modelo Whisper pronto.")
        except Exception as e:
            log_error(f"Falha ao carregar Whisper: {e}")

    def start(self):
        self.running = True
        self.producer_thread = threading.Thread(target=self._producer, daemon=True)
        self.consumer_thread = threading.Thread(target=self._consumer, daemon=True)
        self.producer_thread.start()
        self.consumer_thread.start()

    def stop(self):
        self.running = False
        if self.producer_thread:
            self.producer_thread.join(timeout=1)
        if self.consumer_thread:
            self.consumer_thread.join(timeout=1)

    def _producer(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                            input_device_index=self.input_device_index, frames_per_buffer=self.CHUNK)
            while self.running:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception:
                    break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _consumer(self):
        audio_buffer = [] 
        silence_counter = 0
        is_recording = False
        
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        log_info("SISTEMA DE OUVIDOS ATIVO (Fale algo...)")

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=1)
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                energy = np.sqrt(np.mean(audio_data.astype(np.float32)**2))

                if is_recording:
                    audio_buffer.append(chunk)
                    if energy < self.ENERGY_THRESHOLD:
                        silence_counter += 1
                    else:
                        silence_counter = 0 
                    
                    if silence_counter >= self.SILENCE_CHUNKS:
                        log_info("Silêncio detectado. Transcrevendo...")
                        self._transcribe_buffer(audio_buffer)
                        audio_buffer = []
                        silence_counter = 0
                        is_recording = False
                else:
                    if energy > self.ENERGY_THRESHOLD:
                        log_info("Voz detectada! Gravando...")
                        is_recording = True
                        audio_buffer.append(chunk)
                        silence_counter = 0

            except queue.Empty:
                continue
            except Exception as e:
                log_error(f"Erro no Consumer: {e}")

    def _transcribe_buffer(self, buffer_list):
        if not buffer_list: return
        full_audio_data = b"".join(buffer_list)
        audio_array = np.frombuffer(full_audio_data, np.int16).astype(np.float32) / 32768.0

        try:
            segments, _ = self.model.transcribe(
                audio_array, language=self.language, beam_size=5,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
            )
            text_segment = " ".join([s.text for s in segments]).strip()
            
            if text_segment:
                log_user(text_segment)
                if self.callback:
                    self.callback(text_segment)
                    
        except Exception as e:
            log_error(f"Erro Transcrição: {e}")

class VoiceAssistant:
    def __init__(self):
        log_info("Inicializando Voice Assistant...")
        self.main_server = MainServer(_main_url=_main_url, _aux_url=_aux_url, _main_model=_main_model, _aux_model=_aux_model)
        self.tts = TTS()
        
        self.stt = RealTimeSTT(
            model_size="small", 
            device="cuda", 
            language="pt", 
            callback=self.process_input
        )

    def process_input(self, text):
        cmd = text.lower().strip().replace('.', '')
        if cmd in ["sair", "exit", "desligar"]:
            log_warning("Comando de saída detectado. Encerrando...")
            self.stt.stop()
            exit(0)

        log_info("Gerando resposta da IA...")
        ai_response = self.main_server.generate(text)
        
        # SÓ FALA SE TIVER RESPOSTA (se for None, fica quieto)
        if ai_response:
            log_info("Sintetizando voz (TTS)...")
            self.tts.generate(text=ai_response)
        
        log_success("Ciclo concluído. Aguardando nova fala...")

    def run(self):
        # ... (pode manter o run igual estava) ...
        log_info('Pulling models (verificando atualizações)...')
        self.main_server.pull()
        log_info('Executando teste inicial de Áudio/LLM...')
        test_text = self.main_server.generate_test()
        log_info(f"Falando resposta de teste...")
        self.tts.generate(text=test_text)
        log_success('ASSISTENTE PRONTO.')
        self.stt.start()
        try:
            while self.stt.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            log_warning("Interrupção pelo teclado detectada.")
            self.stt.stop()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()