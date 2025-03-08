from typing import Dict, List, Tuple, Union, Optional
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

class TranscriptionService:
    """
    Handles audio/video transcription using Whisper models.
    
    Attributes:
        model: The loaded Whisper model
        processor: The text processor for the model
        device: The compute device (cuda, mps, or cpu)
        torch_dtype: The tensor data type
        hf_token: The Hugging Face token
    """
    
    def __init__(
            self, 
            model_id: str = "openai/whisper-large-v3-turbo", 
            device: Optional[str] = None, 
            hf_token: Optional[str] = None
        ) -> None:
        """
        Initialize the transcription service with the specified model.
        
        Args:
            model_id: The Hugging Face model ID for the Whisper model
            device: Computing device, if None will be auto-selected
            hf_token: The Hugging Face token
        """
        # Setup device
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Get token from parameter or environment
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Load model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            use_auth_token=self.hf_token
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, use_auth_token=self.hf_token)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True
        )
        
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio or video file to text.
        
        Args:
            audio_path: Path to the audio or video file
            
        Returns:
            Transcribed text
        """
        result = self.pipe(audio_path)
        return result["text"]
    
    def transcribe_with_chunks(self, audio_path: str) -> List[Dict[str, Union[Tuple[float, float], str]]]:
        """
        Transcribe audio and return text with timestamp chunks.
        
        Args:
            audio_path: Path to the audio or video file
            
        Returns:
            List of dictionaries containing timestamp and text chunks
        """
        audio_data, sr = librosa.load(audio_path, sr=16000)

        input_features = self.processor(
            audio_data,
            sampling_rate=sr,
            return_tensors="pt",
            truncation=False
        ).input_features

        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        generated_ids = self.model.generate(
            input_features, 
            return_timestamps=True, 
            return_segments=True
        )

        transcript = self.processor.batch_decode(
            generated_ids["sequences"], 
            skip_special_tokens=True, 
            output_offsets=True
        )

        return transcript[0]["offsets"]
            
