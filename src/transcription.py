import logging
import librosa
from openai import OpenAI
import soundfile as sf

from src.modelConfig import chunk_tempo


def transcribe_openai(audio_file):
    client = OpenAI()
    prompt_text = (
        "Esta transcrição refere-se a uma consulta médica em tempo real, onde um médico conversa com seu paciente "
        "e possivelmente com um acompanhante. Cada segmento de áudio faz parte de uma consulta contínua que pode ultrapassar 40 minutos. "
        "Transcreva o áudio mantendo pontuação, capitalização e termos médicos importantes."
    )
    try:
        logging.info("Transcribing file %s via OpenAI", audio_file)
        with open(audio_file, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=prompt_text,
                response_format="json",
                language="pt"
            )
        logging.info("OpenAI transcription completed for: %s", audio_file)
        return result.text
    except Exception as e:
        logging.exception("Error during OpenAI transcription:")
        return ""

def transcribe_local(audio_file, asr_pipeline):
    try:
        if not isinstance(asr_pipeline, dict):
            logging.info("Transcribing file %s using local pipeline", audio_file)
            result = asr_pipeline(audio_file, batch_size=8, return_timestamps=True)
            chunks = result.get("chunks", [])
            transcribed_text = ""
            for chunk in chunks:
                ts = chunk.get("timestamp", (0.0, 0.0))
                text = chunk.get("text", "")
                transcribed_text += f"[{ts[0]:.2f}-{ts[1]:.2f}] {text} "
            return transcribed_text
        else:
            logging.info("Transcribing file %s using fallback local method", audio_file)
            processor = asr_pipeline["processor"]
            model = asr_pipeline["model"]
            tokenizer = asr_pipeline["tokenizer"]
            speech, sr = sf.read(audio_file)
            logging.info("File %s read with sampling_rate=%d", audio_file, sr)
            if sr != 16000:
                logging.info("Resampling audio from %d Hz to 16000 Hz.", sr)
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
        logging.exception("Error during local transcription:")
        return ""