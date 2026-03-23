# Lista de Melhorias - EmilIA

Este documento lista as melhorias identificadas para o arquivo `main.py`, focadas em segurança, arquitetura, robustez e performance.

### 1. 🛡️ Segurança e Credenciais (Crítico)
- **Remover Hardcoded Tokens:** Existe um token `Bearer` exposto no `__init__` da classe `AuxServer`. Utilize variáveis de ambiente (`os.getenv`) ou um arquivo `.env`.
- **Tratamento de Arquivos Temporários:** Arquivos como `screenshot.jpg` e `test.wav` são salvos na raiz. Considere usar a biblioteca `tempfile` para evitar poluir o diretório do projeto e garantir que sejam deletados após o uso.

### 2. 🏗️ Arquitetura e Organização
- **Modularização:** O arquivo está ficando muito grande. Seria ideal separar em módulos:
    - `models.py`: Para as dataclasses de mensagens.
    - `audio_engine.py`: Para `TTS` e `RealTimeSTT`.
    - `llm_client.py`: Para `MainServer` e `AuxServer`.
    - `utils.py`: Para cores e logs.
- **Injeção de Dependência:** Em vez de usar globais como `_main_url`, passe essas configurações para os construtores das classes.

### 3. 🚀 Performance e Concorrência
- **Migração para Asyncio:** Como o projeto lida muito com I/O (requisições de rede para o Ollama e streaming de áudio), o uso de `asyncio` com a biblioteca `ollama-python` (que suporta async) seria muito mais eficiente do que gerenciar threads manualmente.
- **Buffer de Áudio:** No `RealTimeSTT`, o processamento de áudio é síncrono após detectar silêncio. Isso pode travar a captura da próxima frase. O ideal é que a transcrição ocorra em uma thread/task separada para não perder o "tempo real".

### 4. 🛠️ Robustez e Error Handling
- **Validação de Hardware:** O método `_get_cable_index` e `_get_mic_index` retornam `None` se não encontrarem o dispositivo, mas o código continua a execução, o que causará um crash posterior. Adicione verificações explícitas.
- **Retentativas (Retry Logic):** Requisições de rede para o Ollama podem falhar ocasionalmente. Adicionar um decorador de retry simples ajudaria na estabilidade.
- **Configurações via Arquivo:** Em vez de `sys.txt` e `sys_aux.txt`, considere um único `config.yaml` ou `config.json` para centralizar prompts, nomes de modelos e parâmetros de temperatura.

### 5. 🔊 Melhorias no Áudio (STT/TTS)
- **VAD (Voice Activity Detection):** O cálculo de energia manual pode ser substituído por bibliotecas dedicadas como `webrtcvad` ou o próprio VAD interno do Whisper para reduzir falsos positivos.
- **Sample Rate:** Capturar áudio diretamente na taxa esperada pelo modelo (geralmente 16kHz) em vez de fazer downsampling manual (`audio_np[::3]`).

### 6. 📝 Boas Práticas (Pythonic Way)
- **Docstrings:** Adicione documentação para explicar o que cada método faz.
- **Type Hinting:** Expandir o uso de tipos para cobrir todos os retornos de funções.
- **Context Managers:** Garantir que recursos como streams de áudio sejam fechados corretamente usando `with`.
