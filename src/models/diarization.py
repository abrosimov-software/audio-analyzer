from typing import Dict, List, Tuple, Optional, Any
import torch
import os
import tempfile
from pathlib import Path
from pyannote.audio import Pipeline
from pydub import AudioSegment

class SpeakerDiarizationService:
    """
    Handles speaker identification and segmentation for both audio and video files.
    
    Attributes:
        pipeline: PyAnnote diarization pipeline
        device: The compute device (cuda or cpu)
        model_id: ID of the model used
        hf_token: Hugging Face access token
        supported_audio_formats: Set of supported audio file extensions
        supported_video_formats: Set of supported video file extensions
    """

    # Define supported formats
    supported_audio_formats = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    supported_video_formats = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

    def __init__(
        self, 
        model_id: str = "pyannote/speaker-diarization-3.1", 
        device: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> None:
        """
        Initialize the speaker diarization service.
        
        Args:
            model_id: The Hugging Face model ID for diarization
            device: Computing device, if None will be auto-selected
            hf_token: Hugging Face access token for model access
        """
        # Setup device
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Other parameters
        self.model_id = model_id
        
        # Get token from parameter or environment
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Initialize pipeline
        self.pipeline = Pipeline.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_token
        )
        
        # Set device
        self.pipeline.to(torch.device(self.device))
        
    def is_video_file(self, file_path: str) -> bool:
        """
        Determine if the given file is a video based on its extension.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            True if file is a video, False otherwise
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_video_formats
        
    def extract_audio_from_video(self, video_path: str) -> Tuple[str, bool]:
        """
        Extract audio track from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple containing path to extracted audio file and a flag indicating if
            the file is temporary and should be deleted after use
        """
        # Create temporary file for audio
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Extract audio using pydub
        video = AudioSegment.from_file(video_path)
        video.export(temp_audio_path, format="wav")
        
        return temp_audio_path, True
        
    def diarize(self, media_path: str) -> List[Dict[str, Any]]:
        """
        Identify speakers in the audio or video file.
        
        Automatically detects if the file is a video and extracts the audio
        track for processing if needed.
        
        Args:
            media_path: Path to the audio or video file
            
        Returns:
            List of segments with speaker information
        """
        temp_file_created = False
        audio_path = media_path
        
        try:
            # Check if input is a video file
            if self.is_video_file(media_path):
                audio_path, temp_file_created = self.extract_audio_from_video(media_path)
            
            # Run diarization on audio
            diarization = self.pipeline(audio_path)
            
            # Format output
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            return segments
            
        finally:
            # Clean up temporary file if created
            if temp_file_created and os.path.exists(audio_path):
                os.unlink(audio_path)
    