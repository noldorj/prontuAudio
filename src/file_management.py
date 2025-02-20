import os
import json
import logging
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

BASE_TRANSCRICOES_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "transcricoes"))

def listar_transcricoes():
    """
    Lê a pasta BASE_TRANSCRICOES_DIR e, se houver subpastas, navega por elas para encontrar arquivos .json.
    Retorna um dicionário onde as chaves são labels no formato "NomeDaPasta - NomeArquivo" e os valores são os caminhos completos.
    """
    mapping = {}
    logging.info("Iniciando a listagem de transcrições em: %s", BASE_TRANSCRICOES_DIR)
    try:
        # Verifica arquivos .json na raiz de BASE_TRANSCRICOES_DIR
        for file in os.listdir(BASE_TRANSCRICOES_DIR):
            full_path = os.path.join(BASE_TRANSCRICOES_DIR, file)
            if os.path.isfile(full_path) and file.endswith(".json"):
                logging.info("Arquivo encontrado na raiz: %s", full_path)
                mapping[file] = full_path

        # Verifica cada subpasta
        for subdir in os.listdir(BASE_TRANSCRICOES_DIR):
            subdir_path = os.path.join(BASE_TRANSCRICOES_DIR, subdir)
            if os.path.isdir(subdir_path):
                logging.info("Subpasta encontrada: %s", subdir_path)
                for file in os.listdir(subdir_path):
                    if file.endswith(".json"):
                        full_path = os.path.join(subdir_path, file)
                        logging.info("Arquivo encontrado na subpasta %s: %s", subdir, full_path)
                        label = f"{subdir} - {file}"
                        mapping[label] = full_path
    except Exception as e:
        logging.exception("Erro ao listar transcrições:")
    logging.info("Mapeamento final de transcrições: %s", mapping)
    return mapping


def atualizar_lista_transcricoes():
    """
    Returns an updated value for the Dropdown component containing the list of transcription labels.
    """
    logging.info("Atualizando lista de transcrições.")
    mapping = listar_transcricoes()
    choices = list(mapping.keys())
    logging.info("Lista de transcrições encontrada: %s", choices)
    # Use gr.update() to update the choices and reset the value to the first choice (or empty if no choices)
    return gr.update(choices=choices, value=choices[0] if choices else "")

def selecionar_transcricao(label):
    """
    Dado o label de uma transcrição, lê o arquivo JSON correspondente e retorna a transcrição.
    Se o label for uma lista, utiliza o primeiro elemento.
    """
    logging.info("Selecionando transcrição para o label: %s", label)
    if isinstance(label, list):
        label = label[0]
        logging.info("Label convertido de lista para string: %s", label)
    mapping = listar_transcricoes()
    if label in mapping:
        file_path = mapping[label]
        logging.info("Arquivo correspondente encontrado: %s", file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                transcription = data.get("transcription", "")
                logging.info("Transcrição carregada com sucesso.")
                return transcription
        except Exception as e:
            logging.exception("Erro ao ler a transcrição do arquivo:")
            return f"Erro: {e}"
    else:
        logging.error("Label '%s' não encontrado no mapeamento de transcrições.", label)
        return "Transcrição não encontrada."
