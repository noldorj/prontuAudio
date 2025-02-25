import os
import time
import datetime
import threading
import queue
import json
import logging
import base64

import sounddevice as sd
import numpy as np
import soundfile as sf
import gradio as gr
import librosa

import openai
from openai import OpenAI
from llm_summary import gerarResumoProntuario, initialize_openvino_pipeline, openvino_pipeline_global
from file_management import listar_transcricoes, atualizar_lista_transcricoes, selecionar_transcricao, \
    save_transcription_to_file
from modelConfig import (BASE_TRANSCRICOES_DIR, BASE_AUDIOS_DIR, DOWNLOADS_DIR, chunk_tempo, model_id_transcricao)
from utils import get_metadata, convert_to_wav
from transcription import transcribe_openai, transcribe_local

from fpdf import FPDF

# Variáveis globais para transcrição em tempo real (consulta em tempo real)
transcription_data = []  # Lista dos segmentos transcritos

# Variáveis globais para informações do paciente
patient_name_global = ""
current_transcription_file = ""
patient_audio_folder = ""
patient_trans_folder = ""

# Variável global para armazenar o pipeline ASR local (consulta em tempo real)
local_asr_pipeline = None

audio_thread = None
transcription_thread = None

# Variável global para o arquivo de transcrição processada (para arquivos)
processed_transcription_file = ""

audio_queue = queue.Queue()

# Flags para controle das threads de gravação e transcrição (consulta em tempo real)
recording_running = threading.Event()
transcription_running = threading.Event()
transcription_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def is_openai_key_configured():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(api_key)


def audio_recorder():
    chunk_duration = chunk_tempo
    samplerate = 16000
    channels = 1
    while recording_running.is_set():
        try:
            logging.info("Recording audio for %d seconds...", chunk_duration)
            audio_chunk = sd.rec(int(chunk_duration * samplerate), samplerate=samplerate, channels=channels)
            sd.wait()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            audio_filename = os.path.join(patient_audio_folder, f"audio_{timestamp}.wav")
            sf.write(audio_filename, audio_chunk, samplerate)
            logging.info("Audio saved to: %s", audio_filename)
            audio_queue.put(audio_filename)
        except Exception as e:
            logging.exception("Error capturing audio:")
            time.sleep(1)


def transcription_worker(method="openai"):
    asr_pipeline = None
    if method == "local":
        if local_asr_pipeline is None:
            logging.error("Local model not loaded. Skipping transcription.")
            return
        asr_pipeline = local_asr_pipeline
    while transcription_running.is_set():
        try:
            audio_file = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        if method == "openai":
            transcription = transcribe_openai(audio_file)
        else:
            transcription = transcribe_local(audio_file, asr_pipeline)
        segment = {"timestamp": datetime.datetime.now().strftime("%H:%M:%S"), "text": transcription}
        with transcription_lock:
            transcription_data.append(segment)
        save_transcription_to_file(patient_name_global, transcription_data, current_transcription_file)
        audio_queue.task_done()


def start_process(patient_name, method):
    global patient_name_global, current_transcription_file, transcription_data
    global patient_audio_folder, patient_trans_folder, audio_thread, transcription_thread
    patient_name_global = patient_name.strip() if patient_name.strip() else "paciente"
    transcription_data = []
    patient_audio_folder = os.path.join(BASE_AUDIOS_DIR, patient_name_global)
    os.makedirs(patient_audio_folder, exist_ok=True)
    patient_trans_folder = os.path.join(BASE_TRANSCRICOES_DIR, patient_name_global)
    os.makedirs(patient_trans_folder, exist_ok=True)
    current_transcription_file = os.path.join(
        patient_trans_folder,
        f"transcricao_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    )
    save_transcription_to_file(patient_name, transcription_data, current_transcription_file)
    recording_running.set()
    transcription_running.set()
    audio_thread = threading.Thread(target=audio_recorder, daemon=True)
    audio_thread.start()
    transcription_thread = threading.Thread(target=transcription_worker, args=(method,), daemon=True)
    transcription_thread.start()
    logging.info("Process started.")
    return "Consulta iniciada. Transcrição em andamento!"


def stop_process(patient_name):
    global audio_thread, transcription_thread

    logging.info(f"stop_process::  patient_name: {patient_name.strip()}")

    if patient_name == "":
        return "Consulta ainda não foi iniciada."

    logging.info("Process stopped.")
    final_transcription = get_transcription(patient_name_global)
    recording_running.clear()
    transcription_running.clear()
    if audio_thread is not None:
        audio_thread.join(timeout=5)
    if transcription_thread is not None:
        transcription_thread.join(timeout=5)
    return "\n\nTranscrição Final:\n" + final_transcription


def get_transcription(patient_name):
    if patient_name == "":
        return "Consulta ainda não foi iniciada."
    with transcription_lock:
        return "\n".join([f"[{seg['timestamp']}] {seg['text']}" for seg in transcription_data])


# Função de transcrição de arquivo (síncrona) que processa o áudio em chunks e, a cada chunk,
# atualiza um arquivo JSON com a transcrição parcial e a barra de progresso.
def transcricaoArquivo(file_path, patient_name, method):
    import tempfile
    import json
    if method == "openai" and not is_openai_key_configured():
        return "A chave da OpenAI não foi configurada. Insira sua chave na aba 'Configurações' ou selecione o método local."
    patient_name_local = patient_name.strip() if patient_name.strip() else "paciente"
    patient_audio_folder_local = os.path.join(BASE_AUDIOS_DIR, patient_name_local)
    os.makedirs(patient_audio_folder_local, exist_ok=True)
    patient_trans_folder_local = os.path.join(BASE_TRANSCRICOES_DIR, patient_name_local)
    os.makedirs(patient_trans_folder_local, exist_ok=True)
    # Atualiza a variável global para que a função de leitura encontre o arquivo
    global patient_trans_folder
    patient_trans_folder = patient_trans_folder_local
    global processed_transcription_file
    processed_transcription_file = os.path.join(
        patient_trans_folder_local,
        f"transcricao_completa_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    )
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".wav":
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            file_path = convert_to_wav(file_path, temp_wav)
        except Exception as e:
            return f"Erro na conversão do arquivo: {e}"
    try:
        speech, sr = sf.read(file_path)
    except Exception as e:
        return f"Erro ao ler o arquivo de áudio: {e}"
    if sr != 16000:
        try:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception as e:
            return f"Erro durante a reamostragem: {e}"
    chunk_samples = int(chunk_tempo * sr)
    total_samples = len(speech)
    total_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    all_transcriptions = []

    def progress_bar(percent, bar_length=20):
        filled = int(percent * bar_length / 100)
        return "[" + "#" * filled + "-" * (bar_length - filled) + f"] {percent}%"

    for idx in range(total_chunks):
        start_idx = idx * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk = speech[start_idx:end_idx]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_chunk_path = tmp.name
            sf.write(temp_chunk_path, chunk, sr)
        except Exception as e:
            return f"Erro ao criar arquivo temporário para o chunk {idx + 1}: {e}"
        try:
            if method == "openai":
                text = transcribe_openai(temp_chunk_path)
            else:
                global local_asr_pipeline
                if local_asr_pipeline is None:
                    from transformers import WhisperProcessor, WhisperForConditionalGeneration
                    processor = WhisperProcessor.from_pretrained(model_id_transcricao)
                    model = WhisperForConditionalGeneration.from_pretrained(model_id_transcricao)
                    local_asr_pipeline = {"model": model, "processor": processor, "tokenizer": processor.tokenizer}
                text = transcribe_local(temp_chunk_path, local_asr_pipeline)
        except Exception as e:
            text = f"[Erro no chunk {idx + 1}: {e}]"
        finally:
            try:
                os.remove(temp_chunk_path)
            except Exception as rm_err:
                logging.exception("Erro ao remover arquivo temporário: %s", rm_err)
        all_transcriptions.append(text)
        percent = int(((idx + 1) / total_chunks) * 100)
        bar = progress_bar(percent)
        partial_result = {
            "progress": bar,
            "transcription": "\n".join(all_transcriptions)
        }
        try:
            with open(processed_transcription_file, "w", encoding="utf-8") as f:
                json.dump(partial_result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.exception("Erro ao atualizar arquivo de transcrição parcial: %s", e)
        yield f"Processado {idx + 1}/{total_chunks} ({bar}):\n" + "\n".join(all_transcriptions)

    full_transcription = "\n".join(all_transcriptions)
    metadata = get_metadata(patient_name_local)
    data_to_save = {"metadata": metadata, "transcription": full_transcription}
    try:
        with open(processed_transcription_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.exception("Erro ao salvar transcrição completa:")
        return f"Erro ao salvar transcrição completa: {e}"
    return f"Transcrição concluída. Arquivo salvo em: {processed_transcription_file}\n\n{full_transcription}"


# Função para ler o arquivo JSON da transcrição processada, retornando progresso e transcrição
def ler_transcricao_arquivo():
    global patient_trans_folder
    if not patient_trans_folder or not os.path.exists(patient_trans_folder):
        return "Nenhuma transcrição encontrada.", ""
    files = [f for f in os.listdir(patient_trans_folder) if f.endswith(".json")]
    if not files:
        return "Nenhuma transcrição encontrada.", ""
    files.sort()  # Ordena (assumindo timestamp no nome)
    latest_file = os.path.join(patient_trans_folder, files[-1])
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            progress = data.get("progress", "")
            transcription = data.get("transcription", "Nenhuma transcrição encontrada.")
            return progress, transcription
    except Exception as e:
        logging.exception("Erro ao ler a transcrição: %s", e)
        return f"Erro ao ler a transcrição: {e}", ""


def gerarResumoDoArquivo(label, method):
    transcricao = selecionar_transcricao(label)
    if method == "openai":
        if not is_openai_key_configured():
            return "A chave da OpenAI não foi configurada. Insira sua chave na aba 'Configurações' ou selecione o método local."
        resumo = gerarResumoProntuario(transcricao, use_local=False)
    else:
        from llm_summary import openvino_pipeline_global
        if openvino_pipeline_global is None:
            return "Modelo OpenVINO ainda está carregando. Por favor, aguarde um instante e tente novamente."
        resumo = gerarResumoProntuario(transcricao, use_local=True)
    return resumo


def choose_directory():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Selecione a pasta para salvar o PDF")
    root.destroy()
    return folder if folder else ""


def save_summary_as_pdf_in_dir(summary, chosen_dir):
    if not summary.strip():
        return "Nenhum resumo para salvar."
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary)
        output_file = os.path.join(chosen_dir,
                                   f"resumo_prontuario_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
        pdf.output(output_file)
        return os.path.abspath(output_file)
    except Exception as e:
        logging.exception("Erro ao salvar resumo em PDF:")
        return f"Erro ao salvar resumo em PDF: {e}"


def generate_pdf_with_directory(resumo, chosen_dir):
    if not chosen_dir.strip():
        return "<p style='color:red;'>Por favor, escolha um diretório primeiro.</p>"
    if not resumo.strip():
        return "<p style='color:red;'>Nenhum resumo para salvar em PDF.</p>"
    result = save_summary_as_pdf_in_dir(resumo, chosen_dir)
    if not os.path.exists(result):
        return f"<p style='color:red;'>Erro: arquivo PDF não encontrado em {chosen_dir}.</p>"
    try:
        with open(result, "rb") as f:
            pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            html_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{os.path.basename(result)}">Baixar PDF</a>'
        return f"<p style='color:green;'>Resumo gerado com sucesso!</p>{html_link}"
    except Exception as e:
        logging.exception("Erro ao gerar link de download:")
        return f"<p style='color:red;'>Erro ao gerar link de download: {e}</p>"


def set_openai_api_key(api_key):
    openai.api_key = api_key
    env_path = ".env"
    env_lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_lines = f.readlines()
    key_line = f"OPENAI_API_KEY={api_key}\n"
    found = False
    for i, line in enumerate(env_lines):
        if line.startswith("OPENAI_API_KEY="):
            env_lines[i] = key_line
            found = True
            break
    if not found:
        env_lines.append(key_line)
    with open(env_path, "w") as f:
        f.writelines(env_lines)
    return "API key configurada com sucesso e gravada no .env!"


def start_consulta_interface(patient_name, method):
    if patient_name == "":
        return "Preencha o campo 'Nome do Paciente' antes de iniciar a consulta."
    if method == "openai" and not is_openai_key_configured():
        return "A chave da OpenAI não foi configurada. Insira sua chave na aba 'Configurações' ou selecione o método local."
    return start_process(patient_name, method)


def stop_and_show_transcription(patient_name):
    if patient_name_global == "" :
        return "A consulta não iniciou, preencha o Nome do Paciente e clique em iniciar a consulta."
    stop_msg = stop_process(patient_name)
    final_transcription = get_transcription(patient_name)
    return stop_msg + "\n\nTranscrição Final:\n" + final_transcription


def start_transcricao_interface_wrapper(file_obj, patient_name, method):
    return start_transcricao_interface(file_obj, patient_name, method)


# Função de transcrição de áudio (síncrona) em background usando thread.
# Esta função inicia uma thread que processa o áudio em chunks e atualiza um arquivo JSON com o progresso.
# O botão "Atualizar Transcrição" lerá este arquivo para atualizar o campo.
def start_transcricao_interface(file_obj, patient_name, method):
    if patient_name == "":
        return ("Preencha o campo 'Nome do Paciente' antes de iniciar a transcrição.", "Progresso: 0%")
    if method == "openai" and not is_openai_key_configured():
        return ("A chave da OpenAI não foi configurada. Insira sua chave na aba 'Configurações' ou selecione o método local.", "Progresso: 0%")
    file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    global transcription_progress
    transcription_progress = ""  # Limpa o progresso anterior

    def run_transcription():
        global transcription_progress
        try:
            # Chama a função geradora em background e, a cada chunk, atualiza a variável global
            for update in transcricaoArquivo(file_path, patient_name, method):
                transcription_progress = update  # Atualiza com o último update
                logging.info("Transcrição parcial atualizada: %s", update)
        except Exception as e:
            logging.exception("Erro na thread de transcrição: %s", e)
            transcription_progress += f"\nErro na transcrição: {e}"

    threading.Thread(target=run_transcription, daemon=True).start()
    # Retorna uma tupla: mensagem inicial e progresso inicial
    return ("Transcrição iniciada. Clique em 'Atualizar Transcrição' para ver o progresso.", "Progresso: 0%")


# Função para atualizar a transcrição lendo o arquivo JSON processado e retornar progresso e transcrição.
def atualizar_transcricao():
    progress, transcription = ler_transcricao_arquivo()
    # Retorna uma string que mostra a barra de progresso e a transcrição.
    return f"{progress}\n\n{transcription}"


def ler_transcricao_arquivo():
    global patient_trans_folder
    if not patient_trans_folder or not os.path.exists(patient_trans_folder):
        return "Nenhum progresso encontrado.", "Nenhuma transcrição encontrada."
    files = [f for f in os.listdir(patient_trans_folder) if f.endswith(".json")]
    if not files:
        return "Nenhum progresso encontrado.", "Nenhuma transcrição encontrada."
    files.sort()  # Ordena (assumindo timestamp no nome)
    latest_file = os.path.join(patient_trans_folder, files[-1])
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            progress = data.get("progress", "")
            transcription = data.get("transcription", "Nenhuma transcrição encontrada.")
            return progress, transcription
    except Exception as e:
        logging.exception("Erro ao ler a transcrição: %s", e)
        return f"Erro ao ler a transcrição: {e}", ""


def gerarResumoDoArquivo(label, method):
    logging.info("gerarResumoDoArquivo:: inicnado gerarResumo")

    transcricao = selecionar_transcricao(label)

    if method == "openai":
        if not is_openai_key_configured():
            return "A chave da OpenAI não foi configurada. Insira sua chave na aba 'Configurações' ou selecione o método local."
        resumo = gerarResumoProntuario(transcricao, use_local=False)
    else:
        resumo = gerarResumoProntuario(transcricao, use_local=True)

    return resumo


def choose_directory():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Selecione a pasta para salvar o PDF")
    root.destroy()
    return folder if folder else ""


def save_summary_as_pdf_in_dir(summary, chosen_dir):
    if summary == "":
        return "Nenhum resumo para salvar."
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary)
        output_file = os.path.join(chosen_dir,
                                   f"resumo_prontuario_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
        pdf.output(output_file)
        return os.path.abspath(output_file)
    except Exception as e:
        logging.exception("Erro ao salvar resumo em PDF:")
        return f"Erro ao salvar resumo em PDF: {e}"


def generate_pdf_with_directory(resumo, chosen_dir):
    if not chosen_dir.strip():
        return "<p style='color:red;'>Por favor, escolha um diretório primeiro.</p>"
    if not resumo.strip():
        return "<p style='color:red;'>Nenhum resumo para salvar em PDF.</p>"
    result = save_summary_as_pdf_in_dir(resumo, chosen_dir)
    if not os.path.exists(result):
        return f"<p style='color:red;'>Erro: arquivo PDF não encontrado em {chosen_dir}.</p>"
    try:
        with open(result, "rb") as f:
            pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            html_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{os.path.basename(result)}">Baixar PDF</a>'
        return f"<p style='color:green;'>Resumo gerado com sucesso!</p>{html_link}"
    except Exception as e:
        logging.exception("Erro ao gerar link de download:")
        return f"<p style='color:red;'>Erro ao gerar link de download: {e}</p>"


def set_openai_api_key(api_key):
    openai.api_key = api_key
    env_path = ".env"
    env_lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_lines = f.readlines()
    key_line = f"OPENAI_API_KEY={api_key}\n"
    found = False
    for i, line in enumerate(env_lines):
        if line.startswith("OPENAI_API_KEY="):
            env_lines[i] = key_line
            found = True
            break
    if not found:
        env_lines.append(key_line)
    with open(env_path, "w") as f:
        f.writelines(env_lines)
    return "API key configurada com sucesso e gravada no .env!"


def start_consulta_interface(patient_name, method):
    if patient_name == "":
        return "Preencha o campo 'Nome do Paciente' antes de iniciar a consulta."
    if method == "openai" and not is_openai_key_configured():
        return "A chave da OpenAI não foi configurada. Insira sua chave na aba 'Configurações' ou selecione o método local."
    return start_process(patient_name, method)


def stop_and_show_transcription(patient_name):
    if patient_name_global == "":
        return "A consulta não iniciou, preencha o Nome do Paciente e clique em iniciar a consulta."
    stop_msg = stop_process(patient_name)
    final_transcription = get_transcription(patient_name)
    return stop_msg + "\n\nTranscrição Final:\n" + final_transcription


def start_transcricao_interface_wrapper(file_obj, patient_name, method):
    return start_transcricao_interface(file_obj, patient_name, method)


# Inicia a thread para pré-carregar o modelo OpenVINO em background
import threading
from llm_summary import initialize_openvino_pipeline

# def preload_openvino():
#     logging.info("Iniciando pré-carregamento do modelo OpenVINO em background...")
#     pipeline_instance = initialize_openvino_pipeline()
#     if pipeline_instance:
#         logging.info("Modelo OpenVINO pré-carregado com sucesso.")
#     else:
#         logging.error("Falha no pré-carregamento do modelo OpenVINO.")
#
#
# threading.Thread(target=preload_openvino, daemon=True).start()

# Interface Gradio

with gr.Blocks() as demo:
    gr.Markdown("# Sistema de Consulta e Transcrição")
    with gr.Row():
        patient_name_input = gr.Textbox(label="Nome do Paciente", placeholder="Digite o nome do paciente")
        method_choice = gr.Radio(["openai", "local"], label="Método", value="openai")

    with gr.Tab("Consulta em Tempo Real"):
        consulta_btn = gr.Button("Iniciar Consulta")
        finalizar_consulta_btn = gr.Button("Finalizar Consulta")
        consulta_output = gr.Textbox(label="Saída da Consulta", lines=10)
        atualizar_consulta_btn = gr.Button("🔄 Atualizar Transcrição")

    with gr.Tab("Transcrição de Áudio"):
        file_input = gr.File(label="Arquivo de Áudio")
        transcricao_btn = gr.Button("Iniciar Transcrição")
        interromper_transcricao_btn = gr.Button("Interromper Transcrição")
        atualizar_transcricao_btn = gr.Button("🔄 Atualizar Transcrição")
        transcricao_output = gr.Textbox(label="Transcrição Completa", lines=10, show_copy_button=True)
        progress_output = gr.Textbox(label="Progresso", lines=1, interactive=False)

    with gr.Tab("Transcrições Efetuadas"):
        transcricoes_dropdown = gr.Dropdown(label="Transcrições Efetuadas", choices=[], multiselect=False)
        refresh_transcricoes_btn = gr.Button("Atualizar Lista")
        transcricao_display = gr.Textbox(label="Conteúdo da Transcrição", lines=10, show_copy_button=True)

    with gr.Tab("Resumo do Prontuário"):
        resumo_dropdown = gr.Dropdown(label="Selecione a Transcrição", choices=[], multiselect=False)
        refresh_resumo_btn = gr.Button("Atualizar Lista")
        gerar_resumo_btn = gr.Button("Gerar Resumo")

        resumo_display = gr.Textbox(label="Resumo do Prontuário", lines=10, show_copy_button=True)

        with gr.Row():
            dir_button = gr.Button("📁 Escolher Diretório", elem_classes=["small-button"])
            dir_text = gr.Textbox(label="Diretório Selecionado", interactive=False, show_copy_button=True)
        with gr.Row():
            gerar_pdf_btn = gr.Button("💾 Gerar PDF", variant="secondary", elem_classes=["small-button"])
            pdf_link = gr.HTML(visible=False)
    with gr.Tab("Configurações"):
        gr.Markdown("### Configurações")
        openai_key_label = gr.Label("Chave OpenAI")
        openai_key_input = gr.Textbox(label="Insira sua chave OpenAI", placeholder="Sua API key aqui")
        save_key_button = gr.Button("Salvar 💾")
        save_key_output = gr.Textbox(label="Status", interactive=False)
        instructions_label = gr.HTML("""
        <p>Para obter sua chave API do OpenAI, acesse: <a href="https://platform.openai.com/account/api-keys" target="_blank">https://platform.openai.com/account/api-keys</a></p>
        <p>Se ainda não possui uma conta, crie uma, gere sua chave e insira-a no campo acima para usar os recursos de transcrição.</p>
        """)
        save_key_button.click(fn=set_openai_api_key, inputs=openai_key_input, outputs=save_key_output)
    gr.Markdown("""
    <style>
    .small-button {
      padding: 2px 4px !important;
      font-size: 12px !important;
      min-width: auto !important;
    }
    </style>
    """)
    consulta_btn.click(start_consulta_interface, inputs=[patient_name_input, method_choice], outputs=consulta_output)
    finalizar_consulta_btn.click(stop_and_show_transcription, inputs=[patient_name_input], outputs=consulta_output)
    atualizar_consulta_btn.click(get_transcription, inputs=[patient_name_input], outputs=consulta_output)
    transcricao_btn.click(
        start_transcricao_interface_wrapper,
        inputs=[file_input, patient_name_input, method_choice],
        outputs=[transcricao_output, progress_output],
        show_progress="full"
    )

    interromper_transcricao_btn.click(stop_process, inputs=[patient_name_input], outputs=transcricao_output)
    atualizar_transcricao_btn.click(ler_transcricao_arquivo, inputs=[], outputs=[progress_output, transcricao_output])
    refresh_transcricoes_btn.click(atualizar_lista_transcricoes, outputs=transcricoes_dropdown)
    transcricoes_dropdown.change(selecionar_transcricao, inputs=transcricoes_dropdown, outputs=transcricao_display)
    refresh_resumo_btn.click(atualizar_lista_transcricoes, outputs=resumo_dropdown)
    gerar_resumo_btn.click(gerarResumoDoArquivo, inputs=[resumo_dropdown, method_choice], outputs=resumo_display)
    dir_button.click(lambda: choose_directory(), inputs=[], outputs=dir_text)
    gerar_pdf_btn.click(generate_pdf_with_directory, inputs=[resumo_display, dir_text], outputs=pdf_link)

if __name__ == "__main__":
    def start_gradio():
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inline=False)


    import threading

    gradio_thread = threading.Thread(target=start_gradio, daemon=True)
    gradio_thread.start()
    time.sleep(3)
    import webview

    webview.create_window("Sistema de Consulta e Transcrição", "http://127.0.0.1:7860", width=1024, height=768)
    webview.start()
