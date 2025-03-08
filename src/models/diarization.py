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
        
    def diarize(
        self, 
        media_path: str,
        fast_mode: bool = False,
        use_chunking: bool = True,
        chunk_duration: float = 300.0
    ) -> List[Dict[str, Any]]:
        """
        Identify speakers in the audio or video file with performance optimizations.
        
        Args:
            media_path: Path to the audio or video file
            fast_mode: If True, prioritize speed over accuracy
            use_chunking: Whether to process long files in chunks
            chunk_duration: Duration of each chunk in seconds (if chunking)
            
        Returns:
            List of segments with speaker information
        """
        
        # For longer files, use chunking approach
        if use_chunking:
            # Estimate audio duration
            try:
                audio = AudioSegment.from_file(media_path)
                duration = len(audio) / 1000.0  # in seconds
                
                # If audio is longer than chunk_duration, use chunked processing
                if duration > chunk_duration:
                    return self.diarize_with_chunking(
                        media_path, 
                        chunk_duration=chunk_duration,
                        overlap_duration=min(15.0, chunk_duration * 0.05)  # 5% overlap
                    )
            except:
                pass
        
        # Standard processing for shorter files
        temp_files = []
        audio_path = media_path
        
        try:
            # Check if input is a video file
            if self.is_video_file(media_path):
                audio_path, is_temp = self.extract_audio_from_video(media_path)
                if is_temp:
                    temp_files.append(audio_path)
            
            # Preprocess audio for faster processing
            processed_path, is_temp = self.preprocess_audio(audio_path)
            if is_temp:
                temp_files.append(processed_path)
            
            # Run diarization on processed audio
            diarization = self.pipeline(processed_path)
            
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
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def diarize_with_chunking(
        self, 
        media_path: str, 
        chunk_duration: float = 300.0,  # 5 minutes per chunk
        overlap_duration: float = 15.0  # 15 seconds overlap
    ) -> List[Dict[str, Any]]:
        """
        Process long audio files by breaking them into overlapping chunks.
        
        Args:
            media_path: Path to the audio or video file
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            List of segments with speaker information
        """
        temp_file_created = False
        audio_path = media_path
        
        try:
            # Extract audio if input is a video file
            if self.is_video_file(media_path):
                audio_path, temp_file_created = self.extract_audio_from_video(media_path)
            
            # Get audio duration
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000.0  # Convert to seconds
            
            # Process in chunks
            all_segments = []
            speaker_mapping = {}  # To maintain consistent speaker IDs across chunks
            next_speaker_id = 0
            
            for start_time in range(0, int(total_duration), int(chunk_duration - overlap_duration)):
                # Define chunk boundaries
                chunk_start = max(0, start_time)
                chunk_end = min(total_duration, start_time + chunk_duration)
                
                # Skip if we've reached the end
                if chunk_start >= total_duration:
                    break
                
                # Extract chunk
                chunk = audio[int(chunk_start * 1000):int(chunk_end * 1000)]
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_chunk:
                    chunk_path = temp_chunk.name
                    chunk.export(chunk_path, format="wav")
                
                try:
                    # Process chunk
                    diarization = self.pipeline(chunk_path)
                    
                    # Adjust timestamps and map speakers
                    chunk_segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        # Adjust timestamps relative to original audio
                        adjusted_start = chunk_start + turn.start
                        adjusted_end = chunk_start + turn.end
                        
                        # Skip segments in overlap region (will be processed in next chunk)
                        # except for the last chunk
                        if (adjusted_start >= chunk_end - overlap_duration and 
                            chunk_end < total_duration):
                            continue
                        
                        # Map speaker IDs to be consistent across chunks
                        if speaker not in speaker_mapping:
                            speaker_mapping[speaker] = f"SPEAKER_{next_speaker_id:02d}"
                            next_speaker_id += 1
                        
                        chunk_segments.append({
                            "start": adjusted_start,
                            "end": adjusted_end,
                            "speaker": speaker_mapping[speaker]
                        })
                    
                    all_segments.extend(chunk_segments)
                finally:
                    # Clean up temporary chunk file
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x["start"])
            
            # Merge adjacent segments with same speaker
            merged_segments = []
            if all_segments:
                current = all_segments[0]
                for segment in all_segments[1:]:
                    # If same speaker and close enough, merge
                    if (segment["speaker"] == current["speaker"] and 
                        segment["start"] - current["end"] < 0.5):  # 500ms threshold
                        current["end"] = segment["end"]
                    else:
                        merged_segments.append(current)
                        current = segment
                
                merged_segments.append(current)
            
            return merged_segments
            
        finally:
            # Clean up temporary file if created
            if temp_file_created and os.path.exists(audio_path):
                os.unlink(audio_path)


    def preprocess_audio(
        self, 
        audio_path: str, 
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Tuple[str, bool]:
        """
        Preprocess audio for faster diarization by downsampling.
        
        Args:
            audio_path: Path to the audio file
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels (1 for mono)
            
        Returns:
            Tuple containing path to processed audio file and a flag indicating
            if the file is temporary and should be deleted after use
        """
        # Create temporary file for processed audio
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Load and resample audio using pydub
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if needed
        if audio.channels > 1 and channels == 1:
            audio = audio.set_channels(1)
        
        # Downsample if needed
        if audio.frame_rate > sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        
        # Export processed audio
        audio.export(temp_audio_path, format="wav")
        
        return temp_audio_path, True