from src.models.transcription import TranscriptionService
from src.models.diarization import SpeakerDiarizationService
from src.processing.transcript_processor import TranscriptProcessor
from src.clustering.semantic_cluster import SemanticCluster
from src.models.summarization import SummarizationService
import os
from pathlib import Path


def pipeline(mediafile):
    transcription_service = TranscriptionService()
    diarization_service = SpeakerDiarizationService()
    processor = TranscriptProcessor()
    TEMP_DIR = "temp_files"
    Path(TEMP_DIR).mkdir(exist_ok=True)
    save_path = os.path.join(TEMP_DIR, mediafile.name)
    with open(save_path, "wb") as f:
        f.write(mediafile.getbuffer())
        
    mediafile = save_path
    
    try:
        
        sample_transcription = transcription_service.transcribe_with_chunks(mediafile)
        
        
        sample_diarization = diarization_service.diarize(mediafile)
        
        sample_transcript = processor.merge_transcription_with_speakers(sample_transcription, sample_diarization)

        cluster_service = SemanticCluster()

        clusters = cluster_service.fit_transform(
            sample_transcript
        )
        formatted_clusters = processor.format_transcript_by_topic(clusters)
        summarization_service = SummarizationService()
        
        summaries = summarization_service.summarize_all_topics(formatted_clusters)
        #summary = summarization.summarize(transcription.transcribe(mediafile), diarization.diarize(mediafile))
        
        print("Done")
        
        return summaries
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


#print(pipeline("data/test_data/bel.mp4"))