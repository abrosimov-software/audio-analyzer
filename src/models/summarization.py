from typing import List, Dict, Any, Optional, Tuple, Union
import ollama
import os

class SummarizationService:
    """
    Provides summarization functionality for transcript topics using Ollama.
    
    This service takes transcript data grouped by topic and generates structured
    summaries that include both a main summary and notable events with quotes.
    
    Attributes:
        model_name: The Ollama model used for summarization
        max_tokens: Maximum number of tokens in the generated summary
        temperature: Temperature parameter for text generation
        ollama_host: Host address for Ollama API
    """
    
    def __init__(
        self, 
        model_name: str = "qwen2.5:3b",
        max_tokens: int = 1024, 
        temperature: float = 0.7,
        ollama_host: Optional[str] = None
    ) -> None:
        """
        Initialize the summarization service using Ollama.
        
        Args:
            model_name: Name of the model to use with Ollama
            max_tokens: Maximum number of tokens in the generated summary
            temperature: Temperature parameter for controlling randomness
            ollama_host: Host address for Ollama API (defaults to env var or localhost)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
        
        self.client = ollama.Client(host=self.ollama_host)

    def summarize_topic(
        self, 
        topic_transcript: str
    ) -> str:
        """
        Generate a structured summary for a single topic.
        
        Args:
            topic_transcript: Formatted transcript text for the topic
            
        Returns:
            A structured summary with main points and notable quotes
        """
        
        # Prepare the prompt for summarization
        prompt = f"""Please summarize the following conversation transcript:

{topic_transcript}

Create a concise summary with two sections:
1. Main summary: Provide a cohesive overview of the key points discussed
2. Notable events and quotes: List 2-3 significant moments with direct quotes and their timestamps

Format your response exactly like this:
'''
## Summary
[Your cohesive summary here]

## Notable Moments
- [HH:MM:SS] Quote: "[Exact quote]" - [Brief context]
- [HH:MM:SS] Quote: "[Exact quote]" - [Brief context]
'''
"""
        
        # Call Ollama for generation
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        
        # Extract the generated text
        return response['message']['content']
    
    def summarize_all_topics(
        self, 
        topic_transcripts: List[str]
    ) -> List[str]:
        """
        Generate summaries for all topics in a transcript.
        
        Args:
            topic_transcripts: List of formatted transcript strings, one per topic
            
        Returns:
            List of structured summaries, one for each topic
        """
        summaries = []
        
        for transcript in topic_transcripts:
            
            # Generate summary for this topic
            summary = self.summarize_topic(transcript)
            summaries.append(summary)
        
        return summaries