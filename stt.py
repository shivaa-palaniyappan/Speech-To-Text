import sounddevice as sd
import numpy as np
from groq import Groq

client = Groq(api_key="api key")

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.01

def capture_audio():
    print("\n🎤 Real-Time STT Started")
    print("Start speaking...\n")

    buffer = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            audio_chunk, _ = stream.read(CHUNK_SIZE)

            volume_level = np.linalg.norm(audio_chunk)

            if volume_level > SILENCE_THRESHOLD:
                buffer.append(audio_chunk)
            else:
                if len(buffer) > 0:
                    audio_data = np.concatenate(buffer, axis=0)
                    buffer = []
                    transcribe_audio(audio_data)

def transcribe_audio(audio_np):
    print("⏳ Processing...")

    audio_int16 = (audio_np * 32767).astype(np.int16)
    temp_file = "temp_audio.wav"

    import wave
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    with open(temp_file, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(temp_file, f.read()),
            model="whisper-large-v3-turbo",
            response_format="text"
        )

    print(f"You said: {result.strip()}")

if __name__ == "__main__":
    try:
        capture_audio()
    except KeyboardInterrupt:
        print("\n🛑 Exited.")
