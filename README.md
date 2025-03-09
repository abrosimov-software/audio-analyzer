# Audio Analyzer

A comprehensive toolset for processing, analyzing, and extracting insights from audio conversations.

## Features

- **Transcription**: Convert audio to text with timestamps using Whisper models
- **Speaker Diarization**: Identify different speakers in conversations
- **Topic Clustering**: Group conversation segments by semantic topics
- **Summarization**: Generate structured summaries of conversation topics

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Docker (for containerized deployment)
- Hugging Face account with access to required models
- Ollama for running local language models

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/abrosimov-software/audio-analyzer.git
cd audio-analyzer
```

### 2. Set up environment

#### Option A: Using Conda

```bash
conda env create -n audio-analyzer python=3.10
conda activate audio-analyzer
pip install -r requirements.txt
```

#### Option B: Using DevContainer VSCode Extension

1. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on Mac)
2. Search for "Dev Containers: Reopen in Container"
3. Select it to start the containerized development environment

### 3. Get API Tokens

#### Hugging Face Token

1. Create or log in to your [Hugging Face account](https://huggingface.co/join)
2. Generate a new token at https://huggingface.co/settings/tokens (grant "Read access to contents of all public gated repos you can access" permission)
3. Request access to the following models:
   - [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/speaker-diarization-3.0](https://huggingface.co/pyannote/speaker-diarization-3.0)

### 4. Set up Ollama (Local LLM)

1. Install Ollama by following instructions at [ollama.ai](https://ollama.ai)
2. Pull required models:
   ```bash
   ollama pull qwen2.5:3b
   ```

### 5. Configure environment variables

Create a `.env` file in the project root with the following:

HF_TOKEN=your_huggingface_token_here
OLLAMA_HOST=http://host.docker.internal:11434

If not using Docker, use `OLLAMA_HOST=http://localhost:11434` instead.

## Usage

### Running Jupyter Notebooks

Navigate to the `notebooks/audio_processing/` directory to run example workflows.

## Troubleshooting

### Ollama Connection Issues from Docker

If you're running this project in a Docker container and Ollama on the host, you may encounter connection issues. Solutions:

1. **Use `host.docker.internal`**: This special DNS name resolves to the host machine IP (default in our `.env`)

2. **Use explicit host IP**: Replace `host.docker.internal` with your actual host IP address

3. **Run with host networking**:
   ```bash
   docker run --network=host audio-analyzer
   ```

### Hugging Face Model Access

If you see "401 Unauthorized" or "403 Forbidden" errors with Hugging Face models:
1. Verify your token is correct
2. Check if you've requested access to the gated models