
import os

# Diretórios base para salvar transcrições, áudios e downloads
BASE_TRANSCRICOES_DIR = os.path.join(".", "transcricoes")
BASE_AUDIOS_DIR = os.path.join(".", "audios")
DOWNLOADS_DIR = os.path.join(".", "downloads")

# Tempo de chunk para divisão dos áudios (em segundos)
chunk_tempo = 20

# Modelo para transcrição local (caso utilize a função local de transcrição)
model_id_transcricao = "openai/whisper-large-v2"  # ou outro modelo adequado para transcrição local

# # Configurações para o modelo de sumarização local via OpenVINO
# #model_id_openvino = "AIFunOver/Qwen2.5-7B-Instruct-1M-openvino-fp16"
# model_id_openvino = "Qwen2.5-7B-Instruct-1M-openvino-fp16"
# model_dir_openvino = "C://Users//noldo\PycharmProjects\prontuAudio\models\models--AIFunOver--Qwen2.5-14B-Instruct-1M-openvino-fp16"
# #model_id_cpu = "AIFunOver/Qwen2.5-7B-Instruct-1M-openvino-fp16"
# model_id_cpu = "Qwen2.5-7B-Instruct-1M-openvino-fp16"
# model_dir_cpu = "C://Users//noldo\PycharmProjects\prontuAudio\models\models--AIFunOver--Qwen2.5-14B-Instruct-1M-openvino-fp16"


# Para GPU
model_id_openvino = "hsuwill000/DeepSeek-R1-Distill-Qwen-1.5B-openvino"
model_dir_openvino = os.path.join("models", "models--AIFunOver--DeepSeek-R1-Distill-Llama-8B-openvino-4bit")

# Para CPU
model_id_cpu = "hsuwill000/DeepSeek-R1-Distill-Qwen-1.5B-openvino"
model_dir_cpu = os.path.join("models", "models--AIFunOver--Qwen2.5-7B-Instruct-1M-openvino-fp16")

# AIFunOver/DeepSeek-R1-Distill-Llama-8B-openvino-8bit
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# hsuwill000/DeepSeek-R1-Distill-Qwen-1.5B-openvino


