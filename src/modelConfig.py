# Variáveis que definem os modelos para a OpenAI e para LLM local (para resumo)
model_openai = "gpt-4o"  # Modelo a ser utilizado pela OpenAI para gerar resumos
model_local_llm = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Exemplo para LLM local

# Variável de identificação do modelo de ASR
# model_id = "openai/whisper-small"
# model_id = "openai/whisper-medium"
model_id = "openai/whisper-large-v3-turbo"

# Variável global que define o tempo dos chunks (em segundos)
chunk_tempo = 30

# Variável para definir se o OpenVINO será utilizado (False por padrão)
USE_OPENVINO = False