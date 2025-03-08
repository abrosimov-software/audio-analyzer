from typing import Dict, List, Tuple, Union, Optional, Any

class TranscriptProcessor:
    """
    Processes transcription outputs and combines them with other data sources.
    
    This class provides utilities for post-processing transcription data,
    such as merging transcribed text with speaker identification information.
    """
    
    @staticmethod
    def merge_transcription_with_speakers(
        transcription_chunks: List[Dict[str, Union[Tuple[float, float], str]]], 
        speaker_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge transcription chunks with speaker information.
        
        For each transcription chunk, identifies the dominant speaker by
        calculating temporal overlap with speaker segments.
        
        Args:
            transcription_chunks: List of transcription chunks with timestamps
                Each chunk should have a "timestamp" key with (start, end) tuple and a "text" key
            speaker_segments: List of speaker segments with timestamps
                Each segment should have "start", "end", and "speaker" keys
            
        Returns:
            List of dictionaries containing timestamp, text, and speaker information
        """
        result = []
        
        for chunk in transcription_chunks:
            start_time, end_time = chunk["timestamp"]
            text = chunk["text"]
            
            # Find the dominant speaker for this chunk
            speakers = {}
            for segment in speaker_segments:
                # Skip non-overlapping segments
                if segment["end"] <= start_time or segment["start"] >= end_time:
                    continue
                
                # Calculate overlap duration
                overlap_start = max(start_time, segment["start"])
                overlap_end = min(end_time, segment["end"])
                overlap_duration = overlap_end - overlap_start
                
                speaker = segment["speaker"]
                if speaker in speakers:
                    speakers[speaker] += overlap_duration
                else:
                    speakers[speaker] = overlap_duration
            
            # Get the speaker with most overlap
            dominant_speaker = max(speakers.items(), key=lambda x: x[1])[0] if speakers else "Unknown"
            
            result.append({
                "timestamp": (start_time, end_time),
                "text": text,
                "speaker": dominant_speaker
            })
            
        return result

    @staticmethod
    def format_transcript_for_display(
        merged_transcript: List[Dict[str, Any]], 
        include_timestamps: bool = True
    ) -> str:
        """
        Format the merged transcript into a readable string format.
        
        Args:
            merged_transcript: List of transcript chunks with speaker information
            include_timestamps: Whether to include timestamps in the output
            
        Returns:
            Formatted transcript as a string
        """
        formatted_lines = []
        
        for item in merged_transcript:
            speaker = item["speaker"]
            text = item["text"].strip()
            
            if include_timestamps:
                start, end = item["timestamp"]
                time_str = f"[{start:.2f}s - {end:.2f}s]"
                formatted_lines.append(f"{speaker} {time_str}: {text}")
            else:
                formatted_lines.append(f"{speaker}: {text}")
                
        return "\n".join(formatted_lines)