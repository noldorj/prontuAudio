import os
import time
import datetime
import threading
import queue
import json
import logging

import sounddevice as sd
import numpy as np
import soundfile as sf
import openai
import gradio as gr
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment  # Para conversão de áudio
import librosa  # Para reamostragem

from llm_summary import gerarResumoProntuario

from file_management import listar_transcricoes, atualizar_lista_transcricoes, selecionar_transcricao

from modelConfig import *


# Configuração do logging para incluir nome do arquivo e número da linha
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)



# Suprime DeprecationWarnings, se desejar
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Tenta importar OpenVINO
try:
    import openvino as ov

    OPENVINO_AVAILABLE = True
    logging.info("OpenVINO importado com sucesso.")
except ImportError:
    OPENVINO_AVAILABLE = False
    logging.info("OpenVINO não disponível.")

# Tenta importar a integração do Optimum para OpenVINO
try:
    from optimum.intel.openvino import OVModelForCTC, OVFeatureExtractor, OVTokenizer

    OPTIMUM_INTEL_AVAILABLE = True
    logging.info("Optimum para OpenVINO importado com sucesso.")
except ImportError:
    OPTIMUM_INTEL_AVAILABLE = False
    logging.info("Optimum para OpenVINO não disponível.")

# Força o uso do OpenVINO somente se USE_OPENVINO for True.
USE_OPENVINO = False

# Diretórios base para salvar os arquivos
BASE_TRANSCRICOES_DIR = "../transcricoes"
BASE_AUDIOS_DIR = "../audios"
os.makedirs(BASE_TRANSCRICOES_DIR, exist_ok=True)
os.makedirs(BASE_AUDIOS_DIR, exist_ok=True)
logging.info("Caminho absoluto para transcricoes: %s", os.path.abspath(BASE_TRANSCRICOES_DIR))

# Variáveis globais para a transcrição em tempo real
transcription_data = []  # Lista com os segmentos transcritos (para exibição em tempo real)
transcription_lock = threading.Lock()  # Para acesso seguro à lista
audio_queue = queue.Queue()  # Fila para armazenar os arquivos de áudio a serem transcritos

# Flags para controle dos processos de gravação e transcrição
recording_running = threading.Event()
transcription_running = threading.Event()

# Variáveis globais para o paciente
patient_name_global = ""
current_transcription_file = ""
patient_audio_folder = ""  # Pasta para salvar os áudios do paciente
patient_trans_folder = ""  # Pasta para salvar o JSON da transcrição

# Variável global para armazenar o pipeline do modelo local
local_asr_pipeline = None


# ==================== Funções Auxiliares Gerais ====================

def get_metadata(patient_name):
    """Retorna um dicionário com os metadados: data, horário e nome do paciente."""
    now = datetime.datetime.now()
    return {
        "data": now.strftime("%d/%m/%Y"),
        "horario": now.strftime("%H:%M"),
        "nome_paciente": patient_name
    }


def save_transcription_to_file():
    """Salva a transcrição completa (com metadados) em um arquivo JSON na pasta do paciente."""
    global current_transcription_file
    metadata = get_metadata(patient_name_global)
    data_to_save = {
        "metadata": metadata,
        "transcription": transcription_data
    }
    try:
        with open(current_transcription_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logging.info("Transcrição salva com sucesso em %s", current_transcription_file)
    except Exception as e:
        logging.exception("Erro ao salvar transcrição:")


# ==================== Funções para Transcrição em Tempo Real ====================
def audio_recorder():
    """Grava áudio em tempo real em blocos de chunk_tempo segundos e coloca os arquivos na fila."""
    chunk_duration = chunk_tempo  # segundos
    samplerate = 16000
    channels = 1
    while recording_running.is_set():
        try:
            logging.info("Gravando áudio por %d segundos...", chunk_duration)
            audio_chunk = sd.rec(int(chunk_duration * samplerate), samplerate=samplerate, channels=channels)
            sd.wait()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            audio_filename = os.path.join(patient_audio_folder, f"audio_{timestamp}.wav")
            sf.write(audio_filename, audio_chunk, samplerate)
            logging.info("Áudio salvo em: %s", audio_filename)
            audio_queue.put(audio_filename)
        except Exception as e:
            logging.exception("Erro na captura de áudio:")
            time.sleep(1)


def transcribe_openai(audio_file):
    """Transcreve o áudio utilizando a API da OpenAI (modelo Whisper) com um prompt."""
    client = OpenAI()
    prompt_text = (
        "Esta transcrição refere-se a uma consulta médica em tempo real, onde um médico conversa com seu paciente "
        "e possivelmente com um acompanhante. Cada segmento de áudio faz parte de uma consulta contínua que pode ultrapassar 40 minutos. "
        "Transcreva o áudio mantendo pontuação, capitalização e termos médicos importantes."
    )
    try:
        logging.info("Transcrevendo arquivo %s via OpenAI", audio_file)
        with open(audio_file, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=prompt_text,
                response_format="json",
                language="pt"
            )
        logging.info("Transcrição via OpenAI concluída para: %s", audio_file)
        return result.text
    except Exception as e:
        logging.exception("Erro na transcrição via OpenAI:")
        return ""


def transcribe_local(audio_file, asr_pipeline):
    """
    Transcreve o áudio utilizando o pipeline do Hugging Face (modo local) com processamento em batch e timestamps.
    Se asr_pipeline for um pipeline, utiliza-o com batch_size e return_timestamps.
    Caso contrário (fallback), divide o áudio manualmente em chunks de chunk_tempo.
    """
    try:
        if not isinstance(asr_pipeline, dict):
            logging.info("Transcrevendo arquivo %s via pipeline local com batch e timestamps", audio_file)
            result = asr_pipeline(audio_file, batch_size=8, return_timestamps=True)
            chunks = result.get("chunks", [])
            transcribed_text = ""
            for chunk in chunks:
                ts = chunk.get("timestamp", (0.0, 0.0))
                text = chunk.get("text", "")
                transcribed_text += f"[{ts[0]:.2f}-{ts[1]:.2f}] {text} "
            return transcribed_text
        else:
            logging.info("Transcrevendo arquivo %s via fallback custom local", audio_file)
            processor = asr_pipeline["processor"]
            model = asr_pipeline["model"]
            tokenizer = asr_pipeline["tokenizer"]
            speech, sr = sf.read(audio_file)
            logging.info("Arquivo %s lido com sampling_rate=%d", audio_file, sr)
            if sr != 16000:
                logging.info("Reamostrando áudio de %d Hz para 16000 Hz.", sr)
                speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
                sr = 16000
            chunk_length_samples = int(chunk_tempo * sr)
            chunks_text = []
            for i in range(0, len(speech), chunk_length_samples):
                chunk = speech[i:i + chunk_length_samples]
                inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
                forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="pt", task="transcribe")
                outputs = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
                text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                start_time = i / sr
                end_time = min((i + chunk_length_samples) / sr, len(speech) / sr)
                chunks_text.append(f"[{start_time:.2f}-{end_time:.2f}] {text}")
            return " ".join(chunks_text)
    except Exception as e:
        logging.exception("Erro na transcrição local:")
        return ""


def transcription_worker(method="openai"):
    """Processa a fila de áudio e atualiza a transcrição em tempo real."""
    global local_asr_pipeline
    asr_pipeline = None
    if method == "local":
        if local_asr_pipeline is None:
            logging.error("transcription_worker: Modelo local não carregado. Pulando transcrição.")
            return
        asr_pipeline = local_asr_pipeline
    while transcription_running.is_set():
        try:
            audio_file = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        if method == "openai":
            logging.info("transcription_worker: Chamando transcrição OpenAI para %s", audio_file)
            transcription = transcribe_openai(audio_file)
        else:
            logging.info("transcription_worker: Chamando transcrição local para %s", audio_file)
            transcription = transcribe_local(audio_file, asr_pipeline)
        segment = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "text": transcription
        }
        with transcription_lock:
            transcription_data.append(segment)
        save_transcription_to_file()
        audio_queue.task_done()


def start_process(patient_name, method):
    """Configura as pastas do paciente e inicia as threads de gravação e transcrição em tempo real."""
    logging.info("start_process: Configurando pastas para o paciente '%s'", patient_name)
    global patient_name_global, current_transcription_file, transcription_data, audio_thread, transcription_thread
    global patient_audio_folder, patient_trans_folder
    patient_name_global = patient_name.strip() if patient_name.strip() else "paciente"
    transcription_data = []
    patient_audio_folder = os.path.join(BASE_AUDIOS_DIR, patient_name_global)
    os.makedirs(patient_audio_folder, exist_ok=True)
    patient_trans_folder = os.path.join(BASE_TRANSCRICOES_DIR, patient_name_global)
    os.makedirs(patient_trans_folder, exist_ok=True)
    current_transcription_file = os.path.join(
        patient_trans_folder, f"transcricao_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )
    save_transcription_to_file()
    recording_running.set()
    transcription_running.set()
    audio_thread = threading.Thread(target=audio_recorder, daemon=True)
    audio_thread.start()
    logging.info("start_process: Thread de gravação iniciada.")
    transcription_thread = threading.Thread(target=transcription_worker, args=(method,), daemon=True)
    transcription_thread.start()
    logging.info("start_process: Thread de transcrição iniciada.")


def stop_process():
    """Interrompe as threads de gravação e transcrição em tempo real."""
    recording_running.clear()
    transcription_running.clear()
    if audio_thread is not None:
        audio_thread.join(timeout=5)
    if transcription_thread is not None:
        transcription_thread.join(timeout=5)
    logging.info("stop_process: Consulta interrompida.")
    return "Consulta interrompida."


def stop_and_show_transcription():
    """Interrompe a consulta e retorna a transcrição final completa."""
    stop_msg = stop_process()
    final_transcription = get_transcription()
    return stop_msg + "\n\nTranscrição Final:\n" + final_transcription


def load_local_model_generator():
    """
    Carrega o modelo local e fornece mensagens de progresso.
    Tenta carregar o modelo em dispositivos GPU; se não funcionar, usa CPU.
    """
    global local_asr_pipeline
    try:
        yield "Modelo de IA sendo carregado... 0%"
        time.sleep(0.5)
        if USE_OPENVINO and (OPENVINO_AVAILABLE or OPTIMUM_INTEL_AVAILABLE):
            from openvino import Core
            core = Core()
            devices = core.available_devices
            logging.info("Dispositivos disponíveis: %s", devices)
            gpu_devices = [dev for dev in devices if "GPU" in dev.upper()]
            device_to_use = None
            if gpu_devices:
                for dev in gpu_devices:
                    logging.info("Tentando carregar o modelo no dispositivo: %s", dev)
                    yield f"Modelo de IA sendo carregado... Tentando {dev}"
                    try:
                        cache_dir = "../openvino_cache"
                        os.makedirs(cache_dir, exist_ok=True)
                        model = OVModelForCTC.from_pretrained(model_id, export=True, library="transformers",
                                                              cache_dir=cache_dir)
                        feature_extractor = OVFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
                        tokenizer = OVTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
                        local_asr_pipeline = pipeline(
                            "automatic-speech-recognition",
                            model=model,
                            feature_extractor=feature_extractor,
                            tokenizer=tokenizer,
                            device=dev
                        )
                        device_to_use = dev
                        logging.info("Modelo carregado com sucesso no dispositivo %s.", dev)
                        yield f"Modelo de IA carregado com sucesso no dispositivo {dev}!"
                        break
                    except Exception as gpu_error:
                        logging.exception("Erro ao carregar modelo no dispositivo %s:", dev)
                        yield f"Erro ao carregar no dispositivo {dev}: {gpu_error}"
                if device_to_use is None:
                    logging.info("Nenhum dispositivo GPU funcionou, tentando carregar na CPU.")
                    yield "Nenhum dispositivo GPU funcionou. Carregando modelo na CPU..."
                    cache_dir = "../openvino_cache"
                    os.makedirs(cache_dir, exist_ok=True)
                    model = OVModelForCTC.from_pretrained(model_id, export=True, library="transformers",
                                                          cache_dir=cache_dir)
                    feature_extractor = OVFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
                    tokenizer = OVTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
                    local_asr_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model=model,
                        feature_extractor=feature_extractor,
                        tokenizer=tokenizer,
                        device="CPU"
                    )
                    logging.info("Modelo carregado com sucesso na CPU.")
                    yield "Modelo de IA carregado com sucesso na CPU!"
            else:
                logging.info("Nenhum dispositivo GPU encontrado. Carregando modelo na CPU.")
                yield "Nenhum dispositivo GPU encontrado. Carregando modelo na CPU..."
                cache_dir = "../openvino_cache"
                os.makedirs(cache_dir, exist_ok=True)
                model = OVModelForCTC.from_pretrained(model_id, export=True, library="transformers",
                                                      cache_dir=cache_dir)
                feature_extractor = OVFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
                tokenizer = OVTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
                local_asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    device="CPU"
                )
                logging.info("Modelo carregado com sucesso na CPU.")
                yield "Modelo de IA carregado com sucesso na CPU!"
        else:
            logging.info("Utilizando fallback do Transformers.")
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            processor = WhisperProcessor.from_pretrained(model_id)
            logging.info("Processor carregado com sucesso (fallback).")
            model = WhisperForConditionalGeneration.from_pretrained(model_id)
            logging.info("Modelo carregado com sucesso (fallback).")
            forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="portuguese", task="transcribe")
            local_asr_pipeline = {"model": model, "processor": processor, "tokenizer": processor.tokenizer}
            logging.info("Pipeline custom (fallback) inicializado com forced_decoder_ids.")
            yield "Modelo de IA carregado com sucesso (fallback)!"
        yield "Modelo de IA carregado com sucesso!"
        logging.info("load_local_model_generator: Modelo local carregado com sucesso.")
    except Exception as e:
        yield f"Erro ao carregar modelo local: {e}"
        logging.exception("load_local_model_generator: Erro ao carregar modelo local:")
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            processor = WhisperProcessor.from_pretrained(model_id)
            logging.info("Fallback: Processor carregado com sucesso.")
            model = WhisperForConditionalGeneration.from_pretrained(model_id)
            logging.info("Fallback: Modelo carregado com sucesso.")
            forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="portuguese", task="transcribe")
            local_asr_pipeline = {"model": model, "processor": processor, "tokenizer": processor.tokenizer}
            yield "Modelo de IA carregado com sucesso na CPU (fallback)!"
            logging.info("load_local_model_generator: Fallback carregado com sucesso.")
        except Exception as fallback_error:
            yield f"Erro no fallback ao carregar modelo: {fallback_error}"
            logging.exception("load_local_model_generator: Erro no fallback ao carregar modelo:")
            return


def start_consulta(patient_name, method):
    """
    Inicia a consulta em tempo real.
    Se o método for 'local', aguarda o carregamento completo do modelo antes de iniciar a captura de áudio.
    """
    if method == "local":
        for progress in load_local_model_generator():
            yield progress
        start_process(patient_name, method)
        yield "Consulta iniciada. Transcrição em andamento!"
    else:
        start_process(patient_name, method)
        yield "Consulta iniciada. Transcrição em andamento!"


def get_transcription():
    """
    Retorna a transcrição completa dos segmentos em tempo real.
    """
    with transcription_lock:
        return "\n".join([f"[{seg['timestamp']}] {seg['text']}" for seg in transcription_data])


def convert_to_wav(input_path, output_path):
    """
    Converte um arquivo de áudio para WAV usando pydub e salva em output_path.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        logging.info("Arquivo convertido para WAV com sucesso: %s", output_path)
        return output_path
    except Exception as e:
        logging.exception("Erro ao converter arquivo para WAV:")
        raise e


def transcricaoArquivoFunASR(file_path, patient_name, method):
    """
    Realiza a transcrição de um arquivo de áudio utilizando a biblioteca FunASR:
      - Converte para WAV se necessário.
      - Divide o áudio em blocos de chunk_tempo (em segundos) para processamento.
      - Transcreve cada bloco utilizando FunASR (com timestamps e, se configurado, diarização).
      - Atualiza a transcrição em tempo real com a transcrição acumulada e uma barra de progresso.
      - Ao final, gera um arquivo JSON com a transcrição completa.
    """
    try:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError as ie:
        logging.exception("FunASR não está instalado ou ocorreu erro ao importar FunASR:")
        yield f"Erro: FunASR não está instalado. Detalhes: {ie}"
        return

    global patient_name_global, current_transcription_file
    patient_name_global = patient_name.strip() if patient_name.strip() else "paciente"

    patient_audio_folder_local = os.path.join(BASE_AUDIOS_DIR, patient_name_global)
    os.makedirs(patient_audio_folder_local, exist_ok=True)
    patient_trans_folder_local = os.path.join(BASE_TRANSCRICOES_DIR, patient_name_global)
    os.makedirs(patient_trans_folder_local, exist_ok=True)

    current_transcription_file = os.path.join(
        patient_trans_folder_local, f"transcricao_funasr_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )

    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".wav":
        logging.info("Arquivo com extensão %s não reconhecido. Convertendo para WAV...", ext)
        converted_path = os.path.join(patient_audio_folder_local,
                                      f"converted_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.wav")
        try:
            file_path = convert_to_wav(file_path, converted_path)
        except Exception as e:
            yield f"Erro na conversão do arquivo: {e}"
            return

    all_transcriptions = []
    try:
        speech, sr = sf.read(file_path)
        logging.info("Arquivo %s lido com sampling_rate=%d", file_path, sr)
    except Exception as e:
        logging.exception("Erro ao ler arquivo de áudio:")
        yield f"Erro ao ler arquivo de áudio: {e}"
        return

    if sr != 16000:
        logging.info("Reamostrando áudio de %d Hz para 16000 Hz.", sr)
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000

    chunk_length_samples = int(chunk_tempo * sr)
    total_samples = len(speech)
    segments = []
    for i in range(0, total_samples, chunk_length_samples):
        chunk = speech[i:i + chunk_length_samples]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        chunk_filename = os.path.join(patient_audio_folder_local, f"chunk_{timestamp}_{i}.wav")
        sf.write(chunk_filename, chunk, sr)
        segments.append(chunk_filename)
        logging.info("transcricaoArquivoFunASR: Segmento salvo em %s", chunk_filename)

    total_segments = len(segments)
    model_fun = "iic/SenseVoiceSmall"
    try:
        device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None else "cpu"
        funasr_model = AutoModel(
            model=model_fun,
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            device=device,
        )
        logging.info("FunASR: Modelo carregado com sucesso no dispositivo %s.", device)
    except Exception as e:
        logging.exception("Erro ao carregar FunASR:")
        yield f"Erro ao carregar FunASR: {e}"
        return

    for idx, seg in enumerate(segments):
        logging.info("transcricaoArquivoFunASR: Transcrevendo segmento %s", seg)
        try:
            result = funasr_model.generate(
                input=seg,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=chunk_tempo,
                merge_vad=True,
                merge_length_s=15,
            )
            if "timestamp" in result[0]:
                ts = result[0]["timestamp"]
                text = rich_transcription_postprocess(result[0]["text"])
                transcribed_chunk = f"[{ts[0]:.2f}-{ts[1]:.2f}] {text}"
            else:
                transcribed_chunk = rich_transcription_postprocess(result[0]["text"])
        except Exception as e:
            logging.exception("Erro na transcrição do segmento %s:", seg)
            transcribed_chunk = "[Erro ao transcrever segmento]"
        all_transcriptions.append(transcribed_chunk)
        progress = int(((idx + 1) / total_segments) * 100)
        current_progress = f"Transcrevendo segmento {idx + 1}/{total_segments} ({progress}% concluído)\n" + "\n".join(
            all_transcriptions)
        yield current_progress

    full_transcription = "\n".join(all_transcriptions)
    metadata = get_metadata(patient_name_global)
    data_to_save = {
        "metadata": metadata,
        "transcription": full_transcription,
        "funasr_result": result
    }
    try:
        with open(current_transcription_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logging.info("transcricaoArquivoFunASR: Transcrição completa salva em %s", current_transcription_file)
    except Exception as e:
        logging.exception("transcricaoArquivoFunASR: Erro ao salvar transcrição completa:")
        yield f"Erro ao salvar transcrição completa: {e}"
        return

    yield f"Transcrição concluída. Arquivo salvo em: {current_transcription_file}"


def transcricaoArquivo(file_path, patient_name, method):
    """
    Realiza a transcrição de um arquivo de áudio:
      - Converte para WAV se necessário.
      - Divide o áudio em blocos de chunk_tempo (em segundos) para processamento.
      - Transcreve cada bloco usando o método escolhido (OpenAI ou local).
      - Atualiza a transcrição em tempo real com a transcrição acumulada e uma barra de progresso.
      - Ao final, gera um arquivo JSON com a transcrição completa.
    """
    global patient_name_global, current_transcription_file
    patient_name_global = patient_name.strip() if patient_name.strip() else "paciente"
    patient_audio_folder_local = os.path.join(BASE_AUDIOS_DIR, patient_name_global)
    os.makedirs(patient_audio_folder_local, exist_ok=True)
    patient_trans_folder_local = os.path.join(BASE_TRANSCRICOES_DIR, patient_name_global)
    os.makedirs(patient_trans_folder_local, exist_ok=True)
    current_transcription_file = os.path.join(
        patient_trans_folder_local, f"transcricao_completa_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".wav":
        logging.info("Arquivo com extensão %s não reconhecido. Convertendo para WAV...", ext)
        converted_path = os.path.join(patient_audio_folder_local,
                                      f"converted_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.wav")
        try:
            file_path = convert_to_wav(file_path, converted_path)
        except Exception as e:
            return f"Erro na conversão do arquivo: {e}"
    all_transcriptions = []
    try:
        speech, sr = sf.read(file_path)
        logging.info("Arquivo %s lido com sampling_rate=%d", file_path, sr)
    except Exception as e:
        logging.exception("Erro ao ler arquivo de áudio:")
        return f"Erro ao ler arquivo de áudio: {e}"
    if sr != 16000:
        logging.info("Reamostrando áudio de %d Hz para 16000 Hz.", sr)
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    chunk_length_samples = int(chunk_tempo * sr)
    total_samples = len(speech)
    segments = []
    for i in range(0, total_samples, chunk_length_samples):
        chunk = speech[i:i + chunk_length_samples]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        chunk_filename = os.path.join(patient_audio_folder_local, f"chunk_{timestamp}_{i}.wav")
        sf.write(chunk_filename, chunk, sr)
        segments.append(chunk_filename)
        logging.info("transcricaoArquivo: Segmento salvo em %s", chunk_filename)
    total_segments = len(segments)
    for idx, seg in enumerate(segments):
        logging.info("transcricaoArquivo: Transcrevendo segmento %s", seg)
        if method == "openai":
            text = transcribe_openai(seg)
        else:
            global local_asr_pipeline
            if local_asr_pipeline is None:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                processor = WhisperProcessor.from_pretrained(model_id)
                model = WhisperForConditionalGeneration.from_pretrained(model_id)
                local_asr_pipeline = {"model": model, "processor": processor, "tokenizer": processor.tokenizer}
                logging.info("transcricaoArquivo: Pipeline local custom inicializada.")
            text = transcribe_local(seg, local_asr_pipeline)
        all_transcriptions.append(text)
        progress = int(((idx + 1) / total_segments) * 100)
        current_progress = f"Transcrevendo segmento {idx + 1}/{total_segments} ({progress}% concluído)\n" + "\n".join(
            all_transcriptions)
        yield current_progress
    full_transcription = "\n".join(all_transcriptions)
    metadata = get_metadata(patient_name_global)
    data_to_save = {
        "metadata": metadata,
        "transcription": full_transcription
    }
    try:
        with open(current_transcription_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logging.info("transcricaoArquivo: Transcrição completa salva em %s", current_transcription_file)
    except Exception as e:
        logging.exception("transcricaoArquivo: Erro ao salvar transcrição completa:")
    yield f"Transcrição concluída. Arquivo salvo em: {current_transcription_file}"


from openai import ChatCompletion




def gerarResumoDoArquivo(label):
    """
    Recebe o label selecionado do dropdown de transcrições, lê a transcrição correspondente e gera o resumo do prontuário.
    """
    transcricao = selecionar_transcricao(label)
    resumo = gerarResumoProntuario(transcricao)
    return resumo


def gerarResumoProntuario(transcricao):
    """
    Recebe a transcrição completa e gera um resumo do prontuário com os seguintes tópicos:

    **Dados do Paciente**
    - Nome do paciente, data da consulta (formato DD/MM/AAAA) e horário (formato HH:MM).
    - Nome do acompanhante (se identificado).

    **Medicação em Uso**
    - Liste os nomes e as doses dos medicamentos que o paciente está utilizando.

    **Exames Recentes**
    - Liste os exames realizados e apresentados durante a consulta.

    **Exames a Marcar**
    - Liste os exames pedidos e, se informado, a data da próxima consulta sugerida.

    **Resumo da Consulta**
    - Resuma os principais problemas relatados, possíveis diagnósticos e pontos de atenção.

    **Sugestão de Diagnóstico**
    - Sugira um diagnóstico para apoio ao médico, indicando exames, procedimentos e pontos de atenção.

    Esta função utiliza a API de ChatCompletion da OpenAI com o modelo definido em model_openai para gerar o resumo.
    """
    prompt = f"""Você é um especialista em medicina e análise de dados clínicos. Abaixo está a transcrição completa de uma consulta médica.
Por favor, gere um resumo do prontuário seguindo estes tópicos:

### Dados do Paciente
- Nome do paciente, data da consulta (formato DD/MM/AAAA) e horário (formato HH:MM).
- Nome do acompanhante (se identificado).

### Medicação em Uso
- Liste os nomes e as doses dos medicamentos que o paciente está utilizando.

### Exames Recentes
- Liste os exames realizados e apresentados durante a consulta.

### Exames a Marcar
- Liste os exames pedidos e, se informado, a data da próxima consulta sugerida.

### Resumo da Consulta
- Resuma os principais problemas relatados, possíveis diagnósticos e pontos de atenção.

### Sugestão de Diagnóstico
- Sugira um diagnóstico para apoio ao médico, indicando exames, procedimentos e pontos de atenção.

Transcrição completa:
{transcricao}
"""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_openai,
            messages=[
                {"role": "system", "content": "Você é um especialista em medicina e análise de dados clínicos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logging.exception("Erro ao gerar resumo do prontuário:")
        return f"Erro ao gerar resumo: {e}"

# ==================== Interface Gráfica com Gradio ====================

css = """
.card-green {
    background-color: #e0f2f1;
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
}
"""

with gr.Blocks(css=css) as demo:
    # Card 1: Interface de consulta em tempo real
    with gr.Column(elem_classes=["card-green"]):
        gr.Markdown("### Transcrição em Tempo Real da Consulta Médica")
        with gr.Row():
            patient_name_input = gr.Textbox(label="Nome do Paciente", placeholder="Digite o nome do paciente")
            method_choice = gr.Radio(label="Método de Transcrição", choices=["openai", "local"], value="openai")
        with gr.Row():
            start_button = gr.Button("Iniciar Consulta")
            stop_button = gr.Button("Interromper Consulta")
        status_output = gr.Textbox(label="Status (Tempo Real)")
        transcription_output = gr.Textbox(label="Transcrição (Tempo Real)", lines=15)
        refresh_button = gr.Button("Atualizar Transcrição", elem_id="refresh_button")

    # Card 2: Transcrição de Arquivo
    with gr.Column(elem_classes=["card-green"]):
        gr.Markdown("### Transcrição de Arquivo")
        file_input = gr.File(label="Enviar arquivo", type="filepath")
        file_path_output = gr.Textbox(label="Caminho do arquivo enviado")
        transcricao_file_status = gr.Textbox(label="Status da Transcrição de Arquivo")
        transcricao_file_button = gr.Button("Iniciar Transcrição de Arquivo")

    # Card 3: Lista de Transcrições e Resumo do Prontuário
    with gr.Column(elem_classes=["card-green"]):
        gr.Markdown("### Transcrições Efetuadas")
        dropdown_transcricoes = gr.Dropdown(label="Selecione uma Transcrição", choices=[], allow_custom_value=True)
        atualizar_lista_button = gr.Button("Atualizar Lista")
        gr.Markdown("### Resumo do Prontuário")
        resumo_output = gr.Markdown("Resumo será exibido aqui")
        gerar_resumo_button = gr.Button("Gerar Resumo do Prontuário")

    # Configura os cliques dos botões
    start_button.click(fn=start_consulta, inputs=[patient_name_input, method_choice], outputs=status_output)
    stop_button.click(fn=stop_and_show_transcription, inputs=[], outputs=status_output)
    refresh_button.click(fn=get_transcription, inputs=[], outputs=transcription_output)
    file_input.change(fn=lambda x: x.name if x is not None else "", inputs=file_input, outputs=file_path_output)
    transcricao_file_button.click(
        fn=transcricaoArquivo,
        inputs=[file_input, patient_name_input, method_choice],
        outputs=transcricao_file_status
    )
    atualizar_lista_button.click(fn=atualizar_lista_transcricoes, inputs=[], outputs=dropdown_transcricoes)
    gerar_resumo_button.click(fn=gerarResumoDoArquivo, inputs=[dropdown_transcricoes], outputs=resumo_output)

    auto_refresh = gr.HTML(
        "<script>setInterval(function(){document.getElementById('refresh_button').click();}, 3000);</script>")
    gr.Markdown("A transcrição em tempo real será atualizada automaticamente a cada 3 segundos.")

demo.launch(share=True)
