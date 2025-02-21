import os
import openai
from dotenv import load_dotenv
import logging

# Variáveis que definem os modelos para a OpenAI e para LLM local (para resumo)
model_openai = "gpt-4o"  # Modelo a ser utilizado pela OpenAI para gerar resumos
model_local_llm = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Exemplo para LLM local

# Variável de identificação do modelo de ASR
# model_id = "openai/whisper-small"
# model_id = "openai/whisper-medium"
model_id = "openai/whisper-large-v3-turbo"

# Variável global que define o tempo dos chunks (em segundos)
chunk_tempo = 10

# Variável para definir se o OpenVINO será utilizado (False por padrão)
USE_OPENVINO = False

# Carrega variáveis de ambiente
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Diretórios base
BASE_TRANSCRICOES_DIR = os.path.abspath(os.path.join(os.getcwd(), ".", "transcricoes"))

BASE_AUDIOS_DIR = os.path.abspath(os.path.join(os.getcwd(), ".", "audios"))
DOWNLOADS_DIR = os.path.abspath(os.path.join(os.getcwd(), ".", "downloads"))
os.makedirs(BASE_TRANSCRICOES_DIR, exist_ok=True)
os.makedirs(BASE_AUDIOS_DIR, exist_ok=True)
logging.info("Absolute path for transcriptions: %s", os.path.abspath(BASE_TRANSCRICOES_DIR))

