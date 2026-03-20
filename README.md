# Pyannote Speaker Diarization

A lightweight Python project for speaker diarization (identifying "who spoke when") using the `pyannote.audio` pipeline. This implementation is optimized for Windows with NVIDIA GPU (CUDA 12.6) acceleration.

## ✨ Features

- **Speaker Diarization**: Automatically detect speakers and their respective segments in audio files.
- **GPU Accelerated**: Configured to run on NVIDIA RTX 40-series (or similar) using PyTorch with CUDA 12.6 support.
- **Secure Configuration**: Uses `.env` files to manage Hugging Face Access Tokens securely.

## 📋 Prerequisites

Before running the project, you need:

1.  **Hugging Face Account**: [Create one here](https://huggingface.co/join).
2.  **Model Access**: Visit the following pages and accept the terms of service:
    - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3.  **Hugging Face Token**: Generate an access token from [your settings](https://huggingface.co/settings/tokens).

## 🚀 Installation

### 1. Create and Activate Virtual Environment (Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install PyTorch with CUDA 12.6 support

This specific version is known to work well with modern NVIDIA GPUs (RTX 4050/4060+).

```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install Project Dependencies

```powershell
pip install pyannote.audio python-dotenv
```

## ⚙️ Configuration

Create a file named `.env` in the root directory and add your Hugging Face token:

```env
HF_TOKEN=your_hugging_face_token_here
```

## 📂 Usage

1.  Place your audio file (e.g., `your_audio_file.wav`) in the project directory.
2.  Ensure `main.py` points to your audio file:
    ```python
    diarization = pipeline("your_audio_file.wav")
    ```
3.  Run the diarization script:

```powershell
python main.py
```

### Example Output

The script will output speaker segments in the following format:
```text
[0.00 --> 4.52]: SPEAKER_00
[4.52 --> 8.12]: SPEAKER_01
...
```

## 🛠️ Main Dependencies

- `pyannote.audio`: Speaker diarization pipeline.
- `torch`: Deep learning framework for inference.
- `python-dotenv`: Environment variable management.