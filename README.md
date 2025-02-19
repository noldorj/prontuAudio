# Medical Consultation Transcription and Summary System

This project is a real-time transcription and summarization system designed for medical consultations. It captures and transcribes audio from live consultations (or from pre-recorded audio files) and then uses a language model to generate a structured summary of the consultation (i.e., a medical record summary). The system supports both cloud-based transcription (using OpenAI's Whisper API) and local transcription (using Hugging Face pipelines and FunASR).

## Objective

The goal of this project is to facilitate the creation of digital medical records by automatically transcribing spoken consultations and generating a summarized report. This summary includes key details such as patient information, current medications, recent exams, scheduled exams, and a diagnostic suggestion to support the physician's decision-making.

## How It Works

1. **Real-Time Transcription:**  
   The system captures audio in real time, splitting it into fixed-length chunks (configurable via a global variable). It then transcribes each chunk using either OpenAI’s Whisper API or a local transcription model.

2. **File Transcription:**  
   Users can also upload an audio file. The system converts the file to WAV format if needed, splits it into chunks, transcribes each chunk, and combines the results into a complete transcription.

3. **Transcription Summary:**  
   After transcription is complete, the system can generate a summary of the consultation. It reads the transcription file (stored as JSON), then sends the complete text along with a prompt to an LLM (via OpenAI’s ChatCompletion API) to produce a structured summary including:
   - Patient data (name, consultation date, time, and accompanying person if identified)
   - Current medications
   - Recent exams
   - Exams to be scheduled
   - Consultation summary (key problems, potential diagnoses, points of attention)
   - Diagnostic suggestions

4. **User Interface:**  
   A web interface (built with Gradio) provides three main sections:
   - **Real-Time Consultation:** Start and stop live transcription.
   - **File Transcription:** Upload an audio file and receive its transcription.
   - **Transcription Management & Summary:** List previously generated transcription files and generate a summary for a selected transcription.

## Modules and Technologies Used

- **Python 3.8+**
- **Gradio:** For building the web interface.
- **SoundDevice & SoundFile:** For capturing and processing real-time audio.
- **Pydub:** For audio file conversion.
- **Librosa:** For audio resampling.
- **OpenAI API:** For transcription (Whisper API) and summary generation via ChatCompletion.
- **Transformers:** For local transcription pipelines (e.g., using Whisper models) and potential fallback models.
- **FunASR:** For file transcription with advanced features such as timestamps and speaker diarization.
- **dotenv:** For managing environment variables (e.g., API keys).
- **Logging:** For detailed debugging and traceability.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/medical-consultation-transcription.git
   cd medical-consultation-transcription
