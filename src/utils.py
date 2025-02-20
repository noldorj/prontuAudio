import datetime
import logging
from pydub import AudioSegment



def get_metadata(patient_name):
    now = datetime.datetime.now()
    return {
        "data": now.strftime("%d/%m/%Y"),
        "horario": now.strftime("%H:%M"),
        "nome_paciente": patient_name
    }

def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        logging.exception("Error converting file to WAV:")
        raise e