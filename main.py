import os
from dotenv import load_dotenv
import torch
from pyannote.audio import Pipeline

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

print("Loading Pipeline")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

pipeline.to(torch.device("cuda"))

print("Pipeline Loaded")

print("2. Running Diarization (This might take a minute)...")

diarization = pipeline("your_audio_file.wav")

print("\n--- RAW DIARIZATION OUTPUT ---")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.2f} --> {turn.end:.2f}]: {speaker}")