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
import librosa  # Para reamostragem

from openai import OpenAI
from llm_summary import gerarResumoProntuario
from file_management import listar_transcricoes, atualizar_lista_transcricoes, selecionar_transcricao, \
    save_transcription_to_file
from modelConfig import *  # Deve definir CHUNK_DURATION, chunk_tempo, model_id, etc.
from utils import get_metadata, convert_to_wav
from transcription import transcribe_openai, transcribe_local

# Biblioteca para gerar PDF
from fpdf import FPDF

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Diretório para downloads (se usado para salvar PDFs temporariamente)
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

audio_queue = queue.Queue()

# Flags para controle das threads de gravação e transcrição
recording_running = threading.Event()
transcription_running = threading.Event()
transcription_lock = threading.Lock()

# Variáveis globais para informações do paciente
patient_name_global = ""
current_transcription_file = ""
patient_audio_folder = ""  # Pasta para salvar arquivos de áudio do paciente
patient_trans_folder = ""  # Pasta para salvar arquivos JSON de transcrições

# Variável global para armazenar o pipeline ASR local
local_asr_pipeline = None

transcription_data = []


# ========= FUNÇÕES DE TRANSCRIÇÃO =========

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
    global local_asr_pipeline
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
        segment = {"timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                   "text": transcription}
        with transcription_lock:
            transcription_data.append(segment)
        save_transcription_to_file()
        audio_queue.task_done()


def start_process(patient_name, method):
    logging.info("Setting up directories for patient '%s'", patient_name)
    global patient_name_global, current_transcription_file, transcription_data, audio_thread, transcription_thread
    global patient_audio_folder, patient_trans_folder
    patient_name_global = patient_name.strip() if patient_name.strip() else "paciente"
    transcription_data = []
    patient_audio_folder = os.path.join(BASE_AUDIOS_DIR, patient_name_global)
    os.makedirs(patient_audio_folder, exist_ok=True)
    patient_trans_folder = os.path.join(BASE_TRANSCRICOES_DIR, patient_name_global)
    os.makedirs(patient_trans_folder, exist_ok=True)
    current_transcription_file = os.path.join(
        patient_trans_folder,
        f"transcricao_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    save_transcription_to_file()
    recording_running.set()
    transcription_running.set()
    global audio_thread
    audio_thread = threading.Thread(target=audio_recorder, daemon=True)
    audio_thread.start()
    global transcription_thread
    transcription_thread = threading.Thread(target=transcription_worker, args=(method,), daemon=True)
    transcription_thread.start()
    logging.info("Process started.")


def stop_process():
    recording_running.clear()
    transcription_running.clear()
    if audio_thread is not None:
        audio_thread.join(timeout=5)
    if transcription_thread is not None:
        transcription_thread.join(timeout=5)
    logging.info("Process stopped.")
    return "Consulta interrompida."


def stop_and_show_transcription():
    stop_msg = stop_process()
    final_transcription = get_transcription()
    return stop_msg + "\n\nTranscrição Final:\n" + final_transcription


def get_transcription():
    with transcription_lock:
        return "\n".join([f"[{seg['timestamp']}] {seg['text']}" for seg in transcription_data])


def load_local_model_generator():
    global local_asr_pipeline
    try:
        yield "Modelo de IA sendo carregado... 0%"
        time.sleep(0.5)
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="portuguese", task="transcribe")
        local_asr_pipeline = {"model": model, "processor": processor, "tokenizer": processor.tokenizer}
        yield "Modelo de IA carregado com sucesso (fallback)!"
        yield "Modelo de IA carregado com sucesso!"
    except Exception as e:
        yield f"Erro ao carregar modelo local: {e}"
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            processor = WhisperProcessor.from_pretrained(model_id)
            model = WhisperForConditionalGeneration.from_pretrained(model_id)
            forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="portuguese", task="transcribe")
            local_asr_pipeline = {"model": model, "processor": processor, "tokenizer": processor.tokenizer}
            yield "Modelo de IA carregado com sucesso na CPU (fallback)!"
        except Exception as fallback_error:
            yield f"Erro no fallback ao carregar modelo: {fallback_error}"


def start_consulta(patient_name, method):
    if method == "local":
        results = list(load_local_model_generator())
        start_process(patient_name, method)
        return "\n".join(results) + "\nConsulta iniciada. Transcrição em andamento!"
    else:
        start_process(patient_name, method)
        return "Consulta iniciada. Transcrição em andamento!"


def transcricaoArquivo(file_path, patient_name, method):
    global patient_name_global, current_transcription_file
    patient_name_global = patient_name.strip() if patient_name.strip() else "paciente"
    patient_audio_folder_local = os.path.join(BASE_AUDIOS_DIR, patient_name_global)
    os.makedirs(patient_audio_folder_local, exist_ok=True)
    patient_trans_folder_local = os.path.join(BASE_TRANSCRICOES_DIR, patient_name_global)
    os.makedirs(patient_trans_folder_local, exist_ok=True)
    current_transcription_file = os.path.join(
        patient_trans_folder_local, f"transcricao_completa_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".wav":
        converted_path = os.path.join(patient_audio_folder_local,
                                      f"converted_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.wav")
        try:
            file_path = convert_to_wav(file_path, converted_path)
        except Exception as e:
            return f"Erro na conversão do arquivo: {e}"
    all_transcriptions = []
    try:
        speech, sr = sf.read(file_path)
    except Exception as e:
        return f"Erro ao ler arquivo de áudio: {e}"
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    chunk_length_samples = int(CHUNK_DURATION * sr)
    total_samples = len(speech)
    segments = []
    for i in range(0, total_samples, chunk_length_samples):
        chunk = speech[i:i + chunk_length_samples]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        chunk_filename = os.path.join(patient_audio_folder_local, f"chunk_{timestamp}_{i}.wav")
        sf.write(chunk_filename, chunk, sr)
        segments.append(chunk_filename)
    total_segments = len(segments)
    for idx, seg in enumerate(segments):
        if method == "openai":
            text = transcribe_openai(seg)
        else:
            global local_asr_pipeline
            if local_asr_pipeline is None:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                processor = WhisperProcessor.from_pretrained(model_id)
                model = WhisperForConditionalGeneration.from_pretrained(model_id)
                local_asr_pipeline = {"model": model, "processor": processor, "tokenizer": processor.tokenizer}
            text = transcribe_local(seg, local_asr_pipeline)
        all_transcriptions.append(text)
        progress = int(((idx + 1) / total_segments) * 100)
        current_progress = f"Transcrevendo segmento {idx + 1}/{total_segments} ({progress}% concluído)\n" + "\n".join(
            all_transcriptions)
        yield current_progress
    full_transcription = "\n".join(all_transcriptions)
    metadata = get_metadata(patient_name_global)
    data_to_save = {"metadata": metadata, "transcription": full_transcription}
    try:
        with open(current_transcription_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.exception("Erro ao salvar transcrição completa:")
    yield f"Transcrição concluída. Arquivo salvo em: {current_transcription_file}"


def gerarResumoDoArquivo(label):
    transcricao = selecionar_transcricao(label)
    resumo = gerarResumoProntuario(transcricao)
    return resumo


# Função para permitir que o usuário escolha o diretório (usando Tkinter)
def choose_directory():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()  # Oculta a janela principal
    folder = filedialog.askdirectory(title="Selecione a pasta para salvar o PDF")
    root.destroy()
    return folder if folder else ""


# Nova função que gera o PDF no diretório escolhido
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


# Função que será chamada ao clicar no botão de "Gerar PDF"
def generate_pdf_with_directory(resumo, chosen_dir):
    if not chosen_dir.strip():
        return "<p style='color:red;'>Por favor, escolha um diretório primeiro.</p>"
    result = save_summary_as_pdf_in_dir(resumo, chosen_dir)
    if not os.path.exists(result):
        return f"<p style='color:red;'>Erro: arquivo PDF não encontrado em {chosen_dir}.</p>"
    try:
        with open(result, "rb") as f:
            pdf_bytes = f.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        html_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{os.path.basename(result)}">Baixar PDF</a>'
        return html_link
    except Exception as e:
        logging.exception("Erro ao gerar link de download:")
        return f"<p style='color:red;'>Erro ao gerar link de download: {e}</p>"


# ==================== INTERFACE GRADIO ====================

def start_consulta_interface(patient_name, method):
    if not patient_name.strip():
        return "Preencha o campo 'Nome do Paciente' antes de iniciar a consulta."
    return start_consulta(patient_name, method)


def start_transcricao_interface(file_obj, patient_name, method):
    if not patient_name.strip():
        return "Preencha o campo 'Nome do Paciente' antes de iniciar a transcrição."
    file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    return transcricaoArquivo(file_path, patient_name, method)


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
        atualizar_transcricao_btn = gr.Button("Atualizar Transcrição")
        transcricao_output = gr.Textbox(label="Saída da Transcrição", lines=10)
    with gr.Tab("Transcrições Efetuadas"):
        transcricoes_dropdown = gr.Dropdown(label="Transcrições Efetuadas", choices=[], multiselect=False)
        refresh_transcricoes_btn = gr.Button("Atualizar Lista")
        transcricao_display = gr.Textbox(label="Conteúdo da Transcrição", lines=10)
    with gr.Tab("Resumo do Prontuário"):
        resumo_dropdown = gr.Dropdown(label="Selecione a Transcrição", choices=[], multiselect=False)
        refresh_resumo_btn = gr.Button("Atualizar Lista")
        gerar_resumo_btn = gr.Button("Gerar Resumo")
        resumo_display = gr.Textbox(label="Resumo do Prontuário", lines=10)
        with gr.Row():
            dir_button = gr.Button("📁 Escolher Diretório", elem_classes=["small-button"])
            dir_text = gr.Textbox(label="Diretório Selecionado", interactive=False)
        with gr.Row():
            gerar_pdf_btn = gr.Button("💾 Gerar PDF", variant="secondary", elem_classes=["small-button"])
            pdf_link = gr.HTML(visible=False)
    # Configuração das interações
    consulta_btn.click(start_consulta_interface, inputs=[patient_name_input, method_choice], outputs=consulta_output)
    finalizar_consulta_btn.click(stop_and_show_transcription, inputs=[], outputs=consulta_output)
    atualizar_consulta_btn.click(get_transcription, inputs=[], outputs=consulta_output)
    transcricao_btn.click(start_transcricao_interface, inputs=[file_input, patient_name_input, method_choice],
                          outputs=transcricao_output)
    interromper_transcricao_btn.click(stop_process, inputs=[], outputs=transcricao_output)
    atualizar_transcricao_btn.click(get_transcription, inputs=[], outputs=transcricao_output)
    refresh_transcricoes_btn.click(atualizar_lista_transcricoes, outputs=transcricoes_dropdown)
    transcricoes_dropdown.change(selecionar_transcricao, inputs=transcricoes_dropdown, outputs=transcricao_display)
    refresh_resumo_btn.click(atualizar_lista_transcricoes, outputs=resumo_dropdown)
    gerar_resumo_btn.click(gerarResumoDoArquivo, inputs=[resumo_dropdown], outputs=resumo_display)
    # Botão para escolher diretório
    dir_button.click(lambda: choose_directory(), inputs=[], outputs=dir_text)
    # Botão para gerar PDF usando o diretório escolhido
    gerar_pdf_btn.click(generate_pdf_with_directory, inputs=[resumo_display, dir_text], outputs=pdf_link)

    gr.Markdown("""
    <style>
    .small-button {
      padding: 2px 4px !important;
      font-size: 12px !important;
      min-width: auto !important;
    }
    </style>
    """)

# ========== Integração com PyWebView para janela nativa ==========
if __name__ == "__main__":
    def start_gradio():
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inline=False)


    gradio_thread = threading.Thread(target=start_gradio, daemon=True)
    gradio_thread.start()
    time.sleep(3)
    import webview

    webview.create_window("Sistema de Consulta e Transcrição", "http://127.0.0.1:7860", width=1024, height=768)
    webview.start()
