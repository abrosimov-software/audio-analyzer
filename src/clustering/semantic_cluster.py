from typing import List, Dict, Any, Tuple, Optional, Union, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

class SemanticCluster:
    """
    Groups semantically related text chunks from transcripts into topic clusters.
    
    This class analyzes transcript segments to identify semantic boundaries where
    conversation topics change, then groups similar segments together regardless
    of their position in the timeline.
    
    Attributes:
        model: SentenceTransformer model for generating text embeddings
        threshold: Similarity threshold for clustering (0-1)
        min_chunk_length: Minimum text length to consider for clustering
        clustering_algorithm: Algorithm to use for clustering ('agglomerative' or 'dbscan')
    """
    
    def __init__(
        self, 
        model_name: str = "all-mpnet-base-v2", 
        threshold: float = 0.75,
        min_chunk_length: int = 10,
        clustering_algorithm: str = "agglomerative"
    ) -> None:
        """
        Initialize the semantic clustering service.
        
        Args:
            model_name: Name of the sentence transformer model
                Default is 'all-mpnet-base-v2' which offers better semantic understanding
                than smaller models like MiniLM
            threshold: Similarity threshold for clustering (0-1)
                Higher values create more granular clusters
            min_chunk_length: Minimum character length for chunks to be considered
                Very short segments often lack sufficient context for accurate embedding
            clustering_algorithm: Algorithm to use for clustering
                'agglomerative' (default) or 'dbscan'
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.min_chunk_length = min_chunk_length
        self.clustering_algorithm = clustering_algorithm
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings where each row corresponds to a text
        """
        return self.model.encode(texts, show_progress_bar=True)
    
    def cluster_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        distance_threshold: Optional[float] = None,
        consider_speakers: bool = True,
        min_cluster_size: int = 2
    ) -> List[List[Dict[str, Any]]]:
        """
        Group chunks into semantically related clusters.
        
        Args:
            chunks: List of transcript chunks with at minimum:
                - "text": The text content
                - "timestamp": Tuple of (start_time, end_time)
                - "speaker" (optional): Speaker identifier
            distance_threshold: Optional custom threshold (0-1)
                Override the default threshold set during initialization
            consider_speakers: Whether to use speaker changes as potential topic boundaries
                This can improve clustering when different speakers discuss different topics
            min_cluster_size: Minimum number of chunks required to form a cluster
                Helps prevent over-fragmentation
            
        Returns:
            List of clusters, where each cluster is a list of chunks
        """
        if not chunks:
            return []
        
        # Filter out chunks that are too short
        valid_chunks = [c for c in chunks if len(c.get("text", "")) >= self.min_chunk_length]
        
        # Extract text for embedding
        texts = [chunk["text"] for chunk in valid_chunks]
        
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Factor in speaker changes if requested
        if consider_speakers and all("speaker" in chunk for chunk in valid_chunks):
            # Enhance embeddings with speaker information
            embeddings = self._consider_speaker_changes(embeddings, valid_chunks)
        
        # Perform clustering
        threshold = distance_threshold or self.threshold
        if self.clustering_algorithm == "dbscan":
            clustering = DBSCAN(
                eps=1.0 - threshold,  # Convert similarity to distance
                min_samples=min_cluster_size,
                metric="cosine"
            ).fit(embeddings)
        else:  # Default to agglomerative
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - threshold,  # Convert similarity to distance
                metric="cosine",
                linkage="average"
            ).fit(embeddings)
        
        # Group chunks by cluster
        cluster_ids = clustering.labels_
        clustered_chunks: Dict[int, List[Dict[str, Any]]] = {}
        
        for i, cluster_id in enumerate(cluster_ids):
            # Skip noise points (cluster_id = -1 in DBSCAN)
            if cluster_id == -1:
                continue
                
            if cluster_id not in clustered_chunks:
                clustered_chunks[cluster_id] = []
            clustered_chunks[cluster_id].append(valid_chunks[i])
        
        # Sort chunks within each cluster by timestamp
        result = list(clustered_chunks.values())
        for cluster in result:
            cluster.sort(key=lambda x: x["timestamp"][0])
            
        # Sort clusters themselves by first timestamp
        result.sort(key=lambda x: x[0]["timestamp"][0] if x else float('inf'))
        
        return result
    
    def _consider_speaker_changes(
        self, 
        embeddings: np.ndarray, 
        chunks: List[Dict[str, Any]],
        speaker_change_weight: float = 0.1
    ) -> np.ndarray:
        """
        Adjust embeddings to account for speaker changes.
        
        Args:
            embeddings: Original text embeddings
            chunks: Transcript chunks with speaker information
            speaker_change_weight: Weight given to speaker changes (0-1)
                Higher values make speaker changes more significant in clustering
                
        Returns:
            Modified embeddings that factor in speaker changes
        """
        # Create a copy to avoid modifying the original
        modified_embeddings = embeddings.copy()
        
        # Add a small distance between chunks when speakers change
        for i in range(1, len(chunks)):
            if chunks[i].get("speaker") != chunks[i-1].get("speaker"):
                # Slightly shift the embedding to increase distance
                shift_vector = np.random.normal(0, speaker_change_weight, size=embeddings.shape[1])
                modified_embeddings[i] = modified_embeddings[i] + shift_vector
                # Renormalize to unit length
                modified_embeddings[i] = modified_embeddings[i] / np.linalg.norm(modified_embeddings[i])
                
        return modified_embeddings
    
    def visualize_clusters(
        self, 
        chunks: List[Dict[str, Any]], 
        clusters: List[List[Dict[str, Any]]]
    ) -> plt.Figure:
        """
        Visualize the clustering of transcript chunks.
        
        Creates a timeline visualization showing how chunks are clustered,
        with each cluster represented by a different color.
        
        Args:
            chunks: Original list of transcript chunks
            clusters: Clustered chunks as returned by cluster_chunks
            
        Returns:
            Matplotlib figure object containing the visualization
        """
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Create a mapping of chunks to cluster IDs
        chunk_to_cluster: Dict[Tuple[float, float], int] = {}
        for cluster_id, cluster in enumerate(clusters):
            for chunk in cluster:
                # Use timestamp as a unique identifier
                chunk_to_cluster[chunk["timestamp"]] = cluster_id
        
        # Plot each chunk as a bar on the timeline
        for i, chunk in enumerate(chunks):
            timestamp = chunk["timestamp"]
            start_time, end_time = timestamp
            duration = end_time - start_time
            
            # Get cluster ID or -1 if not clustered
            cluster_id = chunk_to_cluster.get(timestamp, -1)
            
            # Set color based on cluster ID
            color = f'C{cluster_id % 10}' if cluster_id >= 0 else 'gray'
            
            # Plot the chunk
            ax.barh(0, duration, left=start_time, height=0.5, color=color, alpha=0.7)
            
        # Set up the plot
        ax.set_yticks([])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Transcript Clusters Over Time')
        
        return fig
    
    def get_cluster_topics(
        self, 
        clusters: List[List[Dict[str, Any]]],
        num_keywords: int = 5
    ) -> List[str]:
        """
        Extract representative keywords or topics for each cluster.
        
        This is a simple keyword extraction implementation. For more advanced
        topic modeling, consider using additional NLP libraries.
        
        Args:
            clusters: Clustered chunks as returned by cluster_chunks
            num_keywords: Number of keywords to extract per cluster
            
        Returns:
            List of topic descriptions, one per cluster
        """
        # This is a placeholder - in a full implementation, you would use
        # more sophisticated topic modeling or keyword extraction
        topics = []
        
        for cluster in clusters:
            # Combine all text in the cluster
            all_text = " ".join([chunk["text"] for chunk in cluster])
            
            # For now, just take the first few words as the "topic"
            # In a real implementation, use proper keyword extraction
            words = all_text.split()[:num_keywords]
            topic = " ".join(words) + "..."
            
            topics.append(topic)
            
        return topics