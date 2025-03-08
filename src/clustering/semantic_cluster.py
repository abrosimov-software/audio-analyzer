from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import Counter

class SemanticCluster:
    """
    Performs semantic clustering on transcript data to segment conversations by topic.
    
    This class uses BERTopic to identify topics in text and then applies segmentation
    algorithms to divide transcripts into coherent semantic sections.
    
    Attributes:
        topic_model: The BERTopic model used for topic modeling
        segment_threshold: Threshold for determining segment boundaries
        min_segment_length: Minimum number of utterances in a segment
        topics_data: DataFrame containing topic information after fitting
        segments: List of detected segments after segmentation
    """
    
    def __init__(
        self, 
        topic_model: Optional[BERTopic] = None,
        segment_threshold: float = 0.5,
        min_segment_length: int = 3
    ) -> None:
        """
        Initialize the semantic clustering service.
        
        Args:
            topic_model: Pre-initialized BERTopic model (creates a new one if None)
            segment_threshold: Threshold for topic shift detection (0.0-1.0)
            min_segment_length: Minimum number of utterances in a segment
        """
        self.topic_model = topic_model if topic_model else BERTopic(verbose=True)
        self.segment_threshold = segment_threshold
        self.min_segment_length = min_segment_length
        self.topics_data = None
        self.segments = []
        
    def fit_transform(
        self, 
        transcript: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Process transcript to identify topics and group utterances by topic.
        
        Args:
            transcript: List of transcript utterances with text and timestamps
            
        Returns:
            List of lists, where each inner list contains utterances belonging to a topic
        """
        # Save the original transcript
        self.transcript = transcript
        
        # Check if transcript is too short for meaningful segmentation
        # Minimum threshold for running the full algorithm
        MIN_UTTERANCES_FOR_SEGMENTATION = 10
        
        if len(transcript) < MIN_UTTERANCES_FOR_SEGMENTATION:
            # For very short transcripts, create a single topic/segment
            self._handle_short_transcript(transcript)
            return self.get_topics_with_utterances()
        
        # Regular processing for normal-length transcripts
        texts = [segment["text"] for segment in transcript]
        timestamps = [segment["timestamp"][0] for segment in transcript]
        
        # Fit BERTopic model
        topics, probs = self.topic_model.fit_transform(texts)
        self.topics = topics
        self.probs = probs
        
        # Get topics over time for visualization
        # Ensure at least 1 bin
        nr_bins = max(1, min(20, len(texts) // 5))
        topics_over_time = self.topic_model.topics_over_time(
            texts,
            timestamps,
            nr_bins=nr_bins
        )
        
        self.topics_data = topics_over_time
        
        # Also create the traditional segments for visualization and analysis
        self.segments = self._create_segments(transcript, topics)
        
        # Return the transcript items grouped by topic
        return self.get_topics_with_utterances()
    
    def _handle_short_transcript(self, transcript: List[Dict[str, Any]]) -> None:
        """
        Handle very short transcripts by treating them as a single topic.
        
        Args:
            transcript: Short transcript data
        """
        # Try to run BERTopic on the short transcript to get keywords
        texts = [segment["text"] for segment in transcript]
        
        try:
            # Attempt to get topic information
            topics, probs = self.topic_model.fit_transform(texts)
            self.topics = topics
            
            # If all utterances were assigned to topic -1 (outlier),
            # reassign them to topic 0 for consistency
            if all(t == -1 for t in topics):
                self.topics = [0] * len(topics)
            
        except Exception as e:
            # If topic modeling fails on short text, assign all to topic 0
            self.topics = [0] * len(transcript)
            print(f"Notice: Short transcript processed as single topic. {str(e)}")
        
        # Create a single segment for the entire transcript
        start_time = transcript[0]["timestamp"][0]
        end_time = transcript[-1]["timestamp"][1]
        
        # Try to get keywords if possible
        try:
            topic_id = 0
            topic_info = self.topic_model.get_topic(topic_id)
            topic_words = [word for word, _ in topic_info[:5]] if topic_info else []
        except:
            topic_id = 0
            topic_words = []
        
        # Create a single segment
        segment = {
            "start_index": 0,
            "end_index": len(transcript) - 1,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "transcript": transcript,
            "dominant_topic": topic_id,
            "topic_keywords": topic_words,
            "topic_distribution": {topic_id: len(transcript)}
        }
        
        self.segments = [segment]
        
        # Also ensure _detect_topic_boundaries won't fail
        self.min_segment_length = min(2, len(transcript))
    
    def _create_segments(
        self, 
        transcript: List[Dict[str, Any]], 
        topics: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Create segments based on topic shifts.
        
        Args:
            transcript: Original transcript data
            topics: Topic assignments from BERTopic
            
        Returns:
            List of segmented topics with metadata
        """
        # Find significant topic shifts using a sliding window approach
        boundaries = self._detect_topic_boundaries(topics)
        
        # Create segments based on boundaries
        segments = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1] - 1  # -1 because the boundary is the start of the next segment
            
            segment_transcript = transcript[start_idx:end_idx+1]
            segment_topics = topics[start_idx:end_idx+1]
            
            # Get the dominant topic in this segment
            topic_counter = Counter(segment_topics)
            dominant_topic = topic_counter.most_common(1)[0][0]
            
            # Skip outlier segments (topic -1)
            if dominant_topic == -1 and len(topic_counter) > 1:
                # Try to find the next most common topic that isn't -1
                for topic, _ in topic_counter.most_common():
                    if topic != -1:
                        dominant_topic = topic
                        break
            
            # Get topic info
            topic_info = self.topic_model.get_topic(dominant_topic) if dominant_topic != -1 else []
            topic_words = [word for word, _ in topic_info]
            
            # Calculate segment metadata
            start_time = segment_transcript[0]["timestamp"][0]
            end_time = segment_transcript[-1]["timestamp"][1]
            
            # Create segment object
            segment = {
                "start_index": start_idx,
                "end_index": end_idx,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "transcript": segment_transcript,
                "dominant_topic": dominant_topic,
                "topic_keywords": topic_words[:5] if topic_words else [],
                "topic_distribution": dict(topic_counter)
            }
            
            segments.append(segment)
        
        return segments
    
    def _detect_topic_boundaries(self, topics: List[int]) -> List[int]:
        """
        Detect boundaries where topics shift significantly.
        
        Args:
            topics: List of topic assignments
        
        Returns:
            List of indices representing segment boundaries
        """
        # Handle very short transcripts
        if len(topics) <= self.min_segment_length * 2:
            return [0, len(topics)]
        
        # Always include the first utterance as a boundary
        boundaries = [0]
        
        # Use a sliding window to detect significant topic shifts
        # Ensure window size is appropriate for transcript length
        window_size = min(5, max(3, len(topics) // 20))
        
        i = window_size
        while i < len(topics) - window_size:
            window_before = topics[i-window_size:i]
            window_after = topics[i:i+window_size]
            
            # Count topic distributions in each window
            before_counter = Counter(window_before)
            after_counter = Counter(window_after)
            
            # Calculate the dominant topics for each window
            dom_before = before_counter.most_common(1)[0][0] if before_counter else -1
            dom_after = after_counter.most_common(1)[0][0] if after_counter else -1
            
            # Check if dominant topic changed
            if dom_before != dom_after:
                # Calculate how distinct the distributions are
                topic_shift_score = self._calculate_topic_shift(before_counter, after_counter)
                
                if topic_shift_score > self.segment_threshold:
                    # If the last boundary is too close, replace it
                    if boundaries and i - boundaries[-1] < self.min_segment_length:
                        boundaries[-1] = i
                    else:
                        boundaries.append(i)
                    
                    # Skip ahead to avoid detecting multiple boundaries in the same area
                    i += window_size
            
            i += 1
        
        # Always include the last utterance as a boundary
        if len(topics) - 1 not in boundaries:
            boundaries.append(len(topics))
        
        return boundaries
    
    def _calculate_topic_shift(self, counter1: Counter, counter2: Counter) -> float:
        """
        Calculate a topic shift score between two topic distributions.
        
        Args:
            counter1: Topic distribution before
            counter2: Topic distribution after
            
        Returns:
            A score from 0-1 indicating the degree of topic shift
        """
        # Get all unique topics
        all_topics = set(counter1.keys()) | set(counter2.keys())
        
        # Convert to probability distributions
        total1 = sum(counter1.values())
        total2 = sum(counter2.values())
        
        # Calculate Jensen-Shannon divergence (simplified)
        score = 0
        for topic in all_topics:
            p1 = counter1.get(topic, 0) / total1 if total1 else 0
            p2 = counter2.get(topic, 0) / total2 if total2 else 0
            
            # Simple absolute difference
            score += abs(p1 - p2)
        
        # Normalize
        return min(1.0, score / 2.0)
    
    
    def format_segmented_transcript(self) -> str:
        """
        Format the segmented transcript for display.
        
        Returns:
            Formatted string with topic segments and associated transcript
        """
        if not self.segments:
            return "No segments available. Run fit_transform first."
        
        result = []
        
        for i, segment in enumerate(self.segments):
            topic_id = segment["dominant_topic"]
            keywords = ", ".join(segment["topic_keywords"]) if segment["topic_keywords"] else "No specific keywords"
            
            # Add segment header
            result.append(f"\n{'='*80}")
            result.append(f"SEGMENT {i+1}: Topic {topic_id} - {keywords}")
            result.append(f"Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s (Duration: {segment['duration']:.2f}s)")
            result.append(f"{'='*80}\n")
            
            # Add transcript for this segment
            for utterance in segment["transcript"]:
                speaker = utterance.get("speaker", "Unknown")
                text = utterance["text"]
                start, end = utterance["timestamp"]
                result.append(f"{speaker} [{start:.2f}s - {end:.2f}s]: {text}")
            
            result.append("\n")
        
        return "\n".join(result)

    def get_topics_with_utterances(self) -> List[List[Dict[str, Any]]]:
        """
        Group utterances by topic regardless of their position in the transcript.
        
        Returns:
            A list of lists where:
            - Each outer list represents a topic
            - Each inner list contains all utterances (with timestamps and speakers)
              that belong to that topic, in their original chronological order
            
        This format organizes utterances by semantic similarity rather than 
        consecutive segments, allowing for non-contiguous topic discussions.
        """
        if not hasattr(self, 'topics') or not self.topics:
            raise ValueError("No topics available. Run fit_transform first.")
        
        # Create a dictionary to group utterances by topic
        topic_groups = {}
        
        # Iterate through the transcript and group by topic
        for i, (utterance, topic) in enumerate(zip(self.transcript, self.topics)):
            # Initialize the list for this topic if it doesn't exist
            if topic not in topic_groups:
                topic_groups[topic] = []
            
            # Add the utterance to the appropriate topic group
            topic_groups[topic].append(utterance)
        
        # Convert the dictionary to a list of lists, sorted by topic ID
        # (excluding -1 which is the outlier topic)
        sorted_topics = sorted([t for t in topic_groups.keys() if t != -1])
        if -1 in topic_groups:
            sorted_topics.append(-1)  # Add outlier topic at the end if it exists
        
        topic_utterances = [topic_groups[topic] for topic in sorted_topics]
        
        return topic_utterances

    def get_topic_info(self) -> List[Dict[str, Any]]:
        """
        Get information about each topic identified in the transcript.
        
        Returns:
            List of dictionaries containing topic information
        """
        if not hasattr(self, 'topics') or not self.topics:
            return []
        
        # Get unique topics (excluding -1 which is the outlier topic)
        unique_topics = sorted(set([t for t in self.topics if t != -1]))
        if -1 in self.topics:
            unique_topics.append(-1)  # Add outlier topic at the end if it exists
        
        topic_info = []
        for topic in unique_topics:
            # Get keywords for this topic
            keywords = []
            if topic != -1:
                topic_words = self.topic_model.get_topic(topic)
                keywords = [word for word, _ in topic_words[:5]]
            
            # Count utterances for this topic
            count = sum(1 for t in self.topics if t == topic)
            
            # Get timestamps for first and last occurrence
            topic_indices = [i for i, t in enumerate(self.topics) if t == topic]
            first_timestamp = self.transcript[topic_indices[0]]["timestamp"][0] if topic_indices else 0
            last_timestamp = self.transcript[topic_indices[-1]]["timestamp"][1] if topic_indices else 0
            
            topic_info.append({
                "topic_id": topic,
                "keywords": keywords,
                "utterance_count": count,
                "first_occurrence": first_timestamp,
                "last_occurrence": last_timestamp
            })
        
        return topic_info
