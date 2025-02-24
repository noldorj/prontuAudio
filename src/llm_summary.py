import os
import logging
import openai
from openai import OpenAI
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Biblioteca Optimum Intel para OpenVINO
from openvino import Core
from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM

# Variável global para armazenar o pipeline pré-carregado via OpenVINO
openvino_pipeline_global = None


def initialize_openvino_pipeline():
    """
    Inicializa o pipeline de inferência usando OpenVINO, tentando usar a GPU com a maior memória total.
    Se o modelo IR não for encontrado localmente, tenta fazer o download do modelo via Hugging Face utilizando
    a chave API definida no .env.
    Se o dispositivo for CPU, utiliza o modelo "AIFunOver/Qwen2.5-7B-Instruct-1M-openvino-fp16";
    caso contrário (GPU), utiliza o modelo padrão (definido em modelConfig.py).
    Os diretórios dos modelos devem estar definidos em modelConfig.py.
    """
    global openvino_pipeline_global
    if openvino_pipeline_global is not None:
        logging.info("Pipeline OpenVINO já está carregado.")
        return openvino_pipeline_global

    try:
        from dotenv import load_dotenv
        load_dotenv()
        hf_api_key = os.environ.get("HF_API_KEY")
        if hf_api_key:
            logging.info("Hugging Face API key carregada com sucesso.")
        else:
            logging.warning("Hugging Face API key não encontrada no .env.")

        from openvino import Core
        from transformers import AutoTokenizer, pipeline
        from optimum.intel.openvino import OVModelForCausalLM
        from modelConfig import model_id_openvino, model_dir_openvino, model_id_cpu, model_dir_cpu

        core = Core()
        available_devices = core.available_devices
        logging.info("Dispositivos disponíveis: %s", available_devices)

        # Seleciona a GPU com maior memória total, se houver
        gpu_devices = [dev for dev in available_devices if dev.startswith("GPU")]
        if gpu_devices:
            selected_device = None
            max_total_mem = -1
            for dev in gpu_devices:
                try:
                    total_mem = core.get_property(dev, "GPU_DEVICE_TOTAL_MEM_SIZE")
                    if isinstance(total_mem, str) and "UNSUPPORTED" in total_mem.upper():
                        total_mem_val = 0
                    elif isinstance(total_mem, (int, float)):
                        total_mem_val = total_mem
                    else:
                        try:
                            total_mem_val = int(total_mem)
                        except Exception:
                            total_mem_val = 0
                    logging.info("Dispositivo %s: GPU_DEVICE_TOTAL_MEM_SIZE = %s", dev, total_mem_val)
                except Exception as e:
                    logging.exception("Erro ao obter GPU_DEVICE_TOTAL_MEM_SIZE para %s: %s", dev, e)
                    total_mem_val = 0
                if total_mem_val > max_total_mem:
                    max_total_mem = total_mem_val
                    selected_device = dev
            if selected_device is None:
                selected_device = gpu_devices[0]
                logging.info("Selecionando GPU %s (primeira disponível) por padrão.", selected_device)
            else:
                logging.info("Selecionando GPU %s com GPU_DEVICE_TOTAL_MEM_SIZE = %s", selected_device, max_total_mem)
            device = selected_device
        else:
            device = "CPU"
            logging.info("Nenhum dispositivo GPU encontrado; utilizando CPU.")

        # Escolhe o modelo e diretório com base no dispositivo
        if device.upper() == "CPU":
            logging.info("Dispositivo é CPU. Utilizando modelo 'AIFunOver/Qwen2.5-7B-Instruct-1M-openvino-fp16'.")
            model_id_to_use = model_id_cpu
            model_dir_to_use = model_dir_cpu
            ov_config_dict = {
                "compression": None,
                "compile": True,
                "inference_type": "default"
            }
        else:
            logging.info("Dispositivo é GPU. Utilizando modelo padrão.")
            model_id_to_use = model_id_openvino
            model_dir_to_use = model_dir_openvino
            ov_config_dict = {
                "compression": None,
                "compile": True,
                "inference_type": "default",
                "backend_config": {"device_name": device}
            }

        # Define os caminhos esperados para os arquivos IR
        ir_expected_xml = os.path.join(model_dir_to_use, "openvino_model.xml")
        ir_expected_bin = os.path.join(model_dir_to_use, "openvino_model.bin")
        ir_alternate_xml = os.path.join(model_dir_to_use, "model.xml")
        ir_alternate_bin = os.path.join(model_dir_to_use, "model.bin")

        # Verifica e, se necessário, copia os arquivos IR para os nomes esperados
        if not os.path.exists(ir_expected_xml) and os.path.exists(ir_alternate_xml):
            logging.info("Arquivo XML esperado não encontrado. Copiando '%s' para '%s'.", ir_alternate_xml,
                         ir_expected_xml)
            import shutil
            shutil.copy(ir_alternate_xml, ir_expected_xml)
        if not os.path.exists(ir_expected_bin) and os.path.exists(ir_alternate_bin):
            logging.info("Arquivo BIN esperado não encontrado. Copiando '%s' para '%s'.", ir_alternate_bin,
                         ir_expected_bin)
            import shutil
            shutil.copy(ir_alternate_bin, ir_expected_bin)

        # Se os arquivos IR ainda não existirem, tenta o download via Hugging Face
        if not (os.path.exists(ir_expected_xml) and os.path.exists(ir_expected_bin)):
            logging.warning(
                "Arquivos IR não encontrados no diretório: %s. Tentando fazer o download do modelo via Hugging Face.",
                model_dir_to_use)
            logging.info("Iniciando download do modelo via Hugging Face...")

            model_id = "AIFunOver/DeepSeek-R1-Distill-Llama-8B-openvino-4bit"
            ov_model = OVModelForCausalLM.from_pretrained(model_id)

            # ov_model = OVModelForCausalLM.from_pretrained(
            #     model_id_to_use,
            #     cache_dir=model_dir_to_use,
            #     local_files_only=False,  # Permite download do modelo
            #     export=True,
            #     compile=True,
            #     ov_config=ov_config_dict
            # )
            logging.info("Download do modelo concluído e modelo carregado com sucesso.")
        else:
            logging.info("Arquivos IR encontrados: XML=%s, BIN=%s. Carregando modelo localmente.", ir_expected_xml,
                         ir_expected_bin)
            ov_model = OVModelForCausalLM.from_pretrained(
                model_dir_to_use,
                local_files_only=True,
                export=False,
                compile=False,
                ov_config=ov_config_dict
            )
        logging.info("Modelo carregado com sucesso via OpenVINO no dispositivo %s.", device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir_to_use, local_files_only=True)
        except KeyError as ke:
            logging.warning(
                "Erro ao carregar tokenizer do diretório local (%s). Tentando carregar do modelo original (%s): %s",
                model_dir_to_use, model_id_to_use, ke)
            tokenizer = AutoTokenizer.from_pretrained(model_id_to_use, local_files_only=False)

        pipeline_instance = pipeline(
            "text-generation",
            model = ov_model,
            tokenizer = tokenizer,
            trust_remote_code = True,
            device=-1
        )
        logging.info("final:: Modelo carregado com sucesso via OpenVINO no dispositivo %s.", device)
        openvino_pipeline_global = pipeline_instance
        return openvino_pipeline_global
    except Exception as e:
        logging.exception("Erro ao inicializar o pipeline OpenVINO: %s", e)
        return None


def getInferenceOpenVino(transcricao, prompt_base: str):
    """
    Usa o pipeline pré-carregado via OpenVINO para gerar inferência.
    Se o pipeline não estiver carregado, tenta inicializá-lo.
    """
    global openvino_pipeline_global
    if openvino_pipeline_global is None:
        logging.info("Pipeline OpenVINO não carregado. Inicializando em tempo de inferência...")
        if initialize_openvino_pipeline() is None:
            raise RuntimeError("Falha ao carregar pipeline OpenVINO.")
    prompt_completo = prompt_base + f"Transcrição completa:\n{transcricao}\n"
    try:
        response = openvino_pipeline_global(prompt_completo, max_new_tokens=1024)
        logging.info("Geração de resposta concluída com sucesso via pipeline pré-carregado.")
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
            response = getInferenceOpenVino(transcricao, prompt)
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
                    {"role": "system", "content": "Você é um especialista em medicina e análise de dados clínicos."},
                    {"role": "user", "content": prompt_completo}
                ],
                temperature=0.7,
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logging.exception("Erro ao gerar resumo do prontuário via OpenAI: %s", e)
            return f"Erro ao gerar resumo via OpenAI: {e}"

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
    "texto adicional.\n\n"
)
