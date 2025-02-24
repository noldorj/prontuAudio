import datetime
import os
import json
import logging
import gradio as gr
from utils import get_metadata
from modelConfig import BASE_TRANSCRICOES_DIR, BASE_AUDIOS_DIR, DOWNLOADS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def save_transcription_to_file(patient_name, transcription_data, current_transcription_file):
    """
    Atualiza (ou cria) o arquivo de transcrição com os dados e metadata.
    """
    metadata = get_metadata(patient_name)
    data_to_save = {"metadata": metadata, "transcription": transcription_data}

    print(f"save_transcription_to_file:: current_transcription_file: {current_transcription_file}")

    try:
        # Garante que o diretório onde o arquivo será salvo exista
        os.makedirs(os.path.dirname(current_transcription_file), exist_ok=True)
        with open(current_transcription_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logging.info("Transcription updated successfully at %s", current_transcription_file)
    except Exception as e:
        logging.exception("Error saving transcription:")

def listar_transcricoes():
    """
    Lê a pasta BASE_TRANSCRICOES_DIR (i.e., "data/transcricoes") e suas subpastas para encontrar arquivos .json.
    Retorna um dicionário onde as chaves são labels no formato "subpasta - nomeArquivo.json" (ou apenas "nomeArquivo.json"
    para arquivos na raiz) e os valores são os caminhos completos.
    """
    mapping = {}
    logging.info("Iniciando a listagem de transcrições em: %s", BASE_TRANSCRICOES_DIR)
    try:
        # Garante que o diretório BASE_TRANSCRICOES_DIR exista
        if not os.path.exists(BASE_TRANSCRICOES_DIR):
            os.makedirs(BASE_TRANSCRICOES_DIR, exist_ok=True)
            logging.info("Diretório de transcrições criado: %s", BASE_TRANSCRICOES_DIR)

        # Usa os.walk para percorrer a raiz e as subpastas
        for root, dirs, files in os.walk(BASE_TRANSCRICOES_DIR):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    # Cria um label relativo: se estiver na raiz, label é o próprio nome do arquivo; caso contrário,
                    # utiliza o caminho relativo separado por " - "
                    rel_path = os.path.relpath(full_path, BASE_TRANSCRICOES_DIR)
                    label = rel_path.replace(os.sep, " - ")
                    mapping[label] = full_path
                    logging.info("Arquivo encontrado: Label: %s, Caminho: %s", label, full_path)
    except Exception as e:
        logging.exception("Erro ao listar transcrições:")
    logging.info("Mapeamento final de transcrições: %s", mapping)
    return mapping

def atualizar_lista_transcricoes():
    """
    Retorna uma atualização para o componente Dropdown com a lista de labels de transcrições.
    """
    logging.info("Atualizando lista de transcrições.")
    mapping = listar_transcricoes()
    choices = list(mapping.keys())
    logging.info("Lista de transcrições encontrada: %s", choices)
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
