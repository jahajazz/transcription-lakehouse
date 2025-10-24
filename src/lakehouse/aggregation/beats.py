"""
Beat generation from spans using semantic similarity.

Aggregates spans into beats (semantic meaning units) using embedding-based
similarity or heuristic fallback methods. Beats represent coherent topics
or meaning units within an episode.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from lakehouse.aggregation.base import SpanAggregator
from lakehouse.ids import generate_beat_id
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class BeatGenerator(SpanAggregator):
    """
    Generates beats from spans using semantic similarity.
    
    A beat is a semantic meaning unit composed of one or more spans that
    share topical coherence. Beat boundaries are determined by drops in
    semantic similarity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize beat generator.
        
        Args:
            config: Configuration dictionary with options:
                - similarity_threshold: Cosine similarity threshold (default: 0.7)
                - min_spans_per_beat: Minimum spans per beat (default: 1)
                - max_spans_per_beat: Maximum spans per beat or None (default: None)
                - window_size: Sliding window size for similarity (default: 3)
                - use_embeddings: Whether to use embeddings (default: True)
                - fallback_method: Method when embeddings unavailable (default: "heuristic")
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.min_spans_per_beat = self.config.get("min_spans_per_beat", 1)
        self.max_spans_per_beat = self.config.get("max_spans_per_beat", None)
        self.window_size = self.config.get("window_size", 3)
        self.use_embeddings = self.config.get("use_embeddings", True)
        self.fallback_method = self.config.get("fallback_method", "heuristic")
        
        logger.debug(
            f"BeatGenerator initialized: similarity_threshold={self.similarity_threshold}, "
            f"window_size={self.window_size}, use_embeddings={self.use_embeddings}"
        )
    
    def get_artifact_type(self) -> str:
        """Get artifact type name."""
        return "beat"
    
    def aggregate(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate beats from spans.
        
        Args:
            spans: List of span records (can be from multiple episodes)
        
        Returns:
            List of beat records
        
        Example:
            >>> generator = BeatGenerator()
            >>> beats = generator.aggregate(spans)
            >>> len(beats) <= len(spans)  # Beats consolidate spans
            True
        """
        if not self.validate_input(spans):
            return []
        
        # Group by episode and process each separately
        episodes = self.group_by_episode(spans)
        
        all_beats = []
        for episode_id, episode_spans in episodes.items():
            logger.info(f"Generating beats for episode {episode_id} ({len(episode_spans)} spans)")
            
            # Sort spans by time
            sorted_spans = self.sort_by_time(episode_spans)
            
            # Generate beats for this episode
            if self.use_embeddings:
                # Try embedding-based approach
                episode_beats = self._generate_beats_with_embeddings(episode_id, sorted_spans)
            else:
                # Use heuristic fallback
                episode_beats = self._generate_beats_heuristic(episode_id, sorted_spans)
            
            all_beats.extend(episode_beats)
            
            logger.info(f"Generated {len(episode_beats)} beats from {len(episode_spans)} spans for {episode_id}")
        
        # Compute and log statistics
        stats = self.compute_statistics(all_beats)
        self.log_statistics(stats)
        
        return all_beats
    
    def _generate_beats_with_embeddings(
        self,
        episode_id: str,
        spans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate beats using embedding-based semantic similarity.
        
        This method expects spans to have an 'embedding' field. If embeddings
        are not available, falls back to heuristic method.
        
        Args:
            episode_id: Episode identifier
            spans: Sorted list of spans from one episode
        
        Returns:
            List of beat records
        """
        # Check if spans have embeddings
        if not spans or "embedding" not in spans[0]:
            logger.warning(
                f"Spans missing embeddings for {episode_id}, "
                f"falling back to {self.fallback_method} method"
            )
            return self._generate_beats_heuristic(episode_id, spans)
        
        try:
            # Extract embeddings
            embeddings = [np.array(span["embedding"]) for span in spans]
            
            # Compute similarity-based boundaries
            boundaries = self._compute_semantic_boundaries(embeddings)
            
            # Create beats from boundaries
            beats = self._create_beats_from_boundaries(episode_id, spans, boundaries)
            
            return beats
            
        except Exception as e:
            logger.error(f"Error in embedding-based beat generation: {e}")
            logger.info(f"Falling back to {self.fallback_method} method")
            return self._generate_beats_heuristic(episode_id, spans)
    
    def _compute_semantic_boundaries(self, embeddings: List[np.ndarray]) -> List[int]:
        """
        Compute beat boundaries based on semantic similarity drops.
        
        Uses a sliding window approach to detect when similarity drops
        below the threshold, indicating a topic shift.
        
        Args:
            embeddings: List of embedding vectors
        
        Returns:
            List of boundary indices (where new beats start)
        """
        if len(embeddings) <= 1:
            return [0]
        
        boundaries = [0]  # First span always starts a beat
        
        for i in range(1, len(embeddings)):
            # Compute similarity with previous span
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            # Check if similarity drops below threshold
            if similarity < self.similarity_threshold:
                boundaries.append(i)
                logger.debug(f"Beat boundary at span {i} (similarity: {similarity:.3f})")
        
        return boundaries
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Clip to [0, 1] range (cosine can be negative)
        return max(0.0, min(1.0, float(similarity)))
    
    def _generate_beats_heuristic(
        self,
        episode_id: str,
        spans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate beats using heuristic rules (no embeddings required).
        
        Heuristics used:
        - Speaker changes indicate potential beat boundaries
        - Large time gaps indicate beat boundaries
        - Fixed maximum beat duration
        
        Args:
            episode_id: Episode identifier
            spans: Sorted list of spans from one episode
        
        Returns:
            List of beat records
        """
        if not spans:
            return []
        
        logger.debug(f"Using heuristic method for beat generation ({len(spans)} spans)")
        
        boundaries = [0]
        current_speaker = spans[0].get("speaker")
        current_duration = 0.0
        max_beat_duration = 180.0  # 3 minutes maximum per beat
        
        for i in range(1, len(spans)):
            span = spans[i]
            previous_span = spans[i-1]
            
            # Check for speaker change
            if span.get("speaker") != current_speaker:
                boundaries.append(i)
                current_speaker = span.get("speaker")
                current_duration = 0.0
                logger.debug(f"Beat boundary at span {i} (speaker change)")
                continue
            
            # Check for time gap
            time_gap = span.get("start_time", 0.0) - previous_span.get("end_time", 0.0)
            if time_gap > 5.0:  # 5 second gap
                boundaries.append(i)
                current_duration = 0.0
                logger.debug(f"Beat boundary at span {i} (time gap: {time_gap:.1f}s)")
                continue
            
            # Check for maximum duration
            span_duration = span.get("duration", 0.0)
            current_duration += span_duration
            if current_duration > max_beat_duration:
                boundaries.append(i)
                current_duration = 0.0
                logger.debug(f"Beat boundary at span {i} (max duration exceeded)")
                continue
        
        # Create beats from boundaries
        return self._create_beats_from_boundaries(episode_id, spans, boundaries)
    
    def _create_beats_from_boundaries(
        self,
        episode_id: str,
        spans: List[Dict[str, Any]],
        boundaries: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Create beat records from boundary indices.
        
        Args:
            episode_id: Episode identifier
            spans: List of span records
            boundaries: List of indices where beats start
        
        Returns:
            List of beat records
        """
        beats = []
        
        # Add final boundary to simplify loop
        boundaries_with_end = boundaries + [len(spans)]
        
        for beat_idx in range(len(boundaries_with_end) - 1):
            start_idx = boundaries_with_end[beat_idx]
            end_idx = boundaries_with_end[beat_idx + 1]
            
            # Extract spans for this beat
            beat_spans = spans[start_idx:end_idx]
            
            # Check min/max spans constraints
            if len(beat_spans) < self.min_spans_per_beat:
                logger.debug(f"Skipping beat with {len(beat_spans)} spans (< {self.min_spans_per_beat})")
                continue
            
            if self.max_spans_per_beat and len(beat_spans) > self.max_spans_per_beat:
                logger.debug(f"Trimming beat to {self.max_spans_per_beat} spans (had {len(beat_spans)})")
                beat_spans = beat_spans[:self.max_spans_per_beat]
            
            # Create beat record
            beat = self._create_beat(episode_id, beat_spans, len(beats))
            if beat:
                beats.append(beat)
        
        return beats
    
    def _create_beat(
        self,
        episode_id: str,
        spans: List[Dict[str, Any]],
        position: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a beat record from a list of spans.
        
        Args:
            episode_id: Episode identifier
            spans: List of contiguous spans
            position: Position of this beat within the episode
        
        Returns:
            Beat record dictionary or None if invalid
        """
        if not spans:
            return None
        
        # Compute time range
        start_time = spans[0].get("start_time", 0.0)
        end_time = spans[-1].get("end_time", 0.0)
        duration = end_time - start_time
        
        # Concatenate text from all spans
        text = " ".join(span.get("text", "") for span in spans)
        
        # Collect span IDs
        span_ids = [span.get("span_id") for span in spans]
        
        # Generate deterministic beat ID
        beat_id = generate_beat_id(
            episode_id=episode_id,
            position=position,
            span_ids=span_ids,
            text=text,
        )
        
        # Build beat record
        beat = {
            "beat_id": beat_id,
            "episode_id": episode_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "text": text,
            "span_ids": span_ids,
            "topic_label": None,  # Optional, can be added later with topic modeling
        }
        
        return beat
    
    def compute_statistics(self, beats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about generated beats.
        
        Args:
            beats: List of beat records
        
        Returns:
            Statistics dictionary
        """
        if not beats:
            return {
                "count": 0,
                "artifact_type": self.get_artifact_type(),
            }
        
        # Compute duration statistics
        durations = [beat.get("duration", 0.0) for beat in beats]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0.0
        
        # Count spans per beat
        span_counts = [len(beat.get("span_ids", [])) for beat in beats]
        total_spans = sum(span_counts)
        avg_spans_per_beat = total_spans / len(span_counts) if span_counts else 0.0
        
        return {
            "count": len(beats),
            "artifact_type": self.get_artifact_type(),
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "min_duration": min(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "total_spans": total_spans,
            "avg_spans_per_beat": avg_spans_per_beat,
        }


def generate_beats(
    spans: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate beats from spans.
    
    Args:
        spans: List of span records
        config: Configuration dictionary
    
    Returns:
        List of beat records
    
    Example:
        >>> beats = generate_beats(spans, config={"similarity_threshold": 0.7})
        >>> beats[0]["beat_id"]
        'bet_e1b4f4e5c2d9_000000_a7b6c5d4'
    """
    generator = BeatGenerator(config)
    return generator.aggregate(spans)

