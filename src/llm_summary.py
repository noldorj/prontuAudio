import os
import logging
import sys
from openai import OpenAI
import openvino_genai as ov_genai

# Prompt base global
prompt = (
    "Você é um especialista em medicina e análise de dados clínicos. Abaixo está a transcrição completa de uma consulta médica.\n"
    "Por favor, gere um resumo do prontuário seguindo estes tópicos:\n\n"
    "### Dados do Paciente\n"
    "- Nome do paciente, data da consulta (formato DD/MM/AAAA) e horário (formato HH:MM).\n"
    "- Nome do acompanhante (se identificado).\n\n"
    "### Medicação em Uso\n"
    "- Liste os nomes e as doses dos medicamentos que o paciente está utilizando.\n\n"
    "### Exames Recentes\n"
    "- Liste os exames realizados e apresentados durante a consulta.\n\n"
    "### Exames a Marcar\n"
    "- Liste os exames pedidos e, se informado, a data da próxima consulta sugerida.\n\n"
    "### Resumo da Consulta\n"
    "- Resuma os principais problemas relatados, possíveis diagnósticos e pontos de atenção.\n\n"
    "### Sugestão de Diagnóstico\n"
    "- Sugira um diagnóstico para apoio ao médico, indicando exames, procedimentos e pontos de atenção.\n\n"
    "Responda apenas conforme o template acima, em português do Brasil. Responsa apenas o resumo final, não responda nenhum "
    "texto adicional ou processo de pensamento.\n\n"
)


model_llama_8B_int4 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B-int4-ov"
#model_dir = "../models/DeepSeek-R1-Distill-Llama-8B-int4-ov"
model_dir_llama_8B_int4 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models/DeepSeek-R1-Distill-Llama-8B-int4-ov"))
model_dir_llama_8B_int8 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models/DeepSeek-R1-Distill-Llama-8B-int8-ov"))
model_dir_qwen_7B_fp16 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models/DeepSeek-R1-Distill-Qwen-7B-fp16-ov"))

model_dir = model_dir_llama_8B_int8

# Variável global para armazenar o pipeline pré-carregado via OpenVINO
openvino_pipeline_global = None
device = "GPU"
genai_chat_template_llama8B = "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def streamer(subword):
    print(subword, end="", flush=True)
    sys.stdout.flush()
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False

def initialize_openvino_pipeline():
    """
    Inicializa o pipeline de geração via OpenVINO GenAI utilizando a biblioteca openvino_genai.
    Usa o diretório local do modelo (model_dir_openvino) e o dispositivo (device) definidos em modelConfig.py.
    Configura o scheduler, realiza um warmup e retorna o pipeline pronto para uso.
    """
    logging.info("initialize_openvino_pipeline:: inicializando OpenVino ")
    pipe = ov_genai.LLMPipeline(model_dir, device)
    pipe.get_tokenizer().set_chat_template(genai_chat_template_llama8B)

    #warm-up
    logging.info("initialize_openvino_pipeline:: warm-up ")
    #input_prompt = "What is OpenVINO?"
    #print(f"Input text: {input_prompt}")
    #result = pipe.generate(input_prompt, generation_config, streamer)
    #print(f"Result: \n {result}")

    return pipe


def getInferenceOpenVino(prompt_completo: str):
    """
    Usa o pipeline pré-carregado via OpenVINO para gerar inferência.
    Se o pipeline não estiver carregado, tenta inicializá-lo.
    """
    global openvino_pipeline_global

    logging.info("getInferenceOpenVino:: realizando a inferencia")

    if openvino_pipeline_global is None:
        logging.info("getInferenceOpenVino:: inicializando Openvino")
        openvino_pipeline_global = initialize_openvino_pipeline()
    else:
        logging.info("OpenVino ja inicializado...")

    try:
        openvino_pipeline_global.get_tokenizer().set_chat_template(genai_chat_template_llama8B)
        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 128
        response = openvino_pipeline_global.generate(prompt_completo, generation_config, streamer)
        logging.info(f"]\n Response: \n {response}.")

        return response
    except Exception as e:
        logging.exception("Erro durante a geração do resumo com o pipeline OpenVINO: %s", e)
        raise e

def gerarResumoProntuario(transcricao, use_local=False):
    """
    Recebe a transcrição completa e gera um resumo do prontuário.
    Se use_local for True, utiliza o modelo local via OpenVINO (pipeline pré-carregado);
    caso contrário, utiliza a API da OpenAI.
    """
    from modelConfig import model_openai
    global prompt
    prompt_completo = prompt + f"Transcrição completa:\n{transcricao}\n"
    if use_local:
        logging.info("gerarResumoProntuario:: Utilizando modelo local via OpenVINO")
        try:
            response = getInferenceOpenVino(prompt_completo)
            return response
        except Exception as e:
            logging.exception("Erro ao gerar resumo com o pipeline OpenVINO: %s", e)
            return f"Erro ao gerar resumo com o pipeline OpenVINO: {e}"
    else:
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model_openai,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": prompt_completo}
                ],
                temperature=0.7,
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logging.exception("Erro ao gerar resumo do prontuário via OpenAI: %s", e)
            return f"Erro ao gerar resumo via OpenAI: {e}"

