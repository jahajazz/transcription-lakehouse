"""
Section generation from beats using time and semantic boundaries.

Creates 5-12 minute logical blocks from beats, balancing time constraints
with semantic coherence. Sections represent major segments of an episode.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from lakehouse.aggregation.base import BeatAggregator
from lakehouse.ids import generate_section_id
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class SectionGenerator(BeatAggregator):
    """
    Generates sections from beats using time and semantic boundaries.
    
    A section is a 5-12 minute logical block composed of multiple beats.
    Sections balance time constraints with semantic coherence, preferring
    to break at natural topic boundaries.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize section generator.
        
        Args:
            config: Configuration dictionary with options:
                - target_duration_minutes: Target section duration (default: 8.0)
                - min_duration_minutes: Minimum section duration (default: 5.0)
                - max_duration_minutes: Maximum section duration (default: 12.0)
                - allow_semantic_overflow: Allow exceeding max for semantics (default: True)
                - boundary_similarity_threshold: Similarity threshold (default: 0.5)
                - prefer_time_boundaries: Prefer time over semantics (default: False)
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.target_duration_minutes = self.config.get("target_duration_minutes", 8.0)
        self.min_duration_minutes = self.config.get("min_duration_minutes", 5.0)
        self.max_duration_minutes = self.config.get("max_duration_minutes", 12.0)
        self.allow_semantic_overflow = self.config.get("allow_semantic_overflow", True)
        self.boundary_similarity_threshold = self.config.get("boundary_similarity_threshold", 0.5)
        self.prefer_time_boundaries = self.config.get("prefer_time_boundaries", False)
        
        # Convert to seconds for internal use
        self.target_duration = self.target_duration_minutes * 60.0
        self.min_duration = self.min_duration_minutes * 60.0
        self.max_duration = self.max_duration_minutes * 60.0
        
        logger.debug(
            f"SectionGenerator initialized: target={self.target_duration_minutes}min, "
            f"range={self.min_duration_minutes}-{self.max_duration_minutes}min, "
            f"allow_overflow={self.allow_semantic_overflow}"
        )
    
    def get_artifact_type(self) -> str:
        """Get artifact type name."""
        return "section"
    
    def aggregate(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate sections from beats.
        
        Args:
            beats: List of beat records (can be from multiple episodes)
        
        Returns:
            List of section records
        
        Example:
            >>> generator = SectionGenerator()
            >>> sections = generator.aggregate(beats)
            >>> all(5 <= s["duration_minutes"] <= 12 for s in sections)
            True
        """
        if not self.validate_input(beats):
            return []
        
        # Group by episode and process each separately
        episodes = self.group_by_episode(beats)
        
        all_sections = []
        for episode_id, episode_beats in episodes.items():
            logger.info(f"Generating sections for episode {episode_id} ({len(episode_beats)} beats)")
            
            # Sort beats by time
            sorted_beats = self.sort_by_time(episode_beats)
            
            # Generate sections for this episode
            episode_sections = self._generate_sections_for_episode(episode_id, sorted_beats)
            all_sections.extend(episode_sections)
            
            logger.info(
                f"Generated {len(episode_sections)} sections from {len(episode_beats)} beats for {episode_id}"
            )
        
        # Compute and log statistics
        stats = self.compute_statistics(all_sections)
        self.log_statistics(stats)
        
        return all_sections
    
    def _generate_sections_for_episode(
        self,
        episode_id: str,
        beats: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate sections for a single episode.
        
        Args:
            episode_id: Episode identifier
            beats: Sorted list of beats from one episode
        
        Returns:
            List of section records
        """
        if not beats:
            return []
        
        sections = []
        current_section_beats = []
        current_duration = 0.0
        
        for i, beat in enumerate(beats):
            beat_duration = beat.get("duration", 0.0)
            
            # Check if adding this beat would exceed limits
            potential_duration = current_duration + beat_duration
            
            # Decide whether to start a new section
            should_break = self._should_break_section(
                current_section_beats,
                beat,
                current_duration,
                potential_duration,
                i,
                beats,
            )
            
            if should_break and current_section_beats:
                # Create section from current beats
                section = self._create_section(episode_id, current_section_beats, len(sections))
                if section:
                    sections.append(section)
                
                # Start new section
                current_section_beats = [beat]
                current_duration = beat_duration
            else:
                # Add beat to current section
                current_section_beats.append(beat)
                current_duration += beat_duration
        
        # Don't forget the last section
        if current_section_beats:
            section = self._create_section(episode_id, current_section_beats, len(sections))
            if section:
                sections.append(section)
        
        return sections
    
    def _should_break_section(
        self,
        current_beats: List[Dict[str, Any]],
        next_beat: Dict[str, Any],
        current_duration: float,
        potential_duration: float,
        next_index: int,
        all_beats: List[Dict[str, Any]],
    ) -> bool:
        """
        Determine if a new section should start.
        
        Balances time constraints with semantic boundaries.
        
        Args:
            current_beats: Beats in current section
            next_beat: Beat being considered
            current_duration: Current section duration in seconds
            potential_duration: Duration if next beat is added
            next_index: Index of next beat
            all_beats: All beats in episode
        
        Returns:
            True if section should break, False otherwise
        """
        # Don't break on first beat
        if not current_beats:
            return False
        
        # Check minimum duration - must have at least this much
        if current_duration < self.min_duration:
            return False
        
        # If we prefer time boundaries, break at target regardless of semantics
        if self.prefer_time_boundaries:
            if potential_duration >= self.target_duration:
                logger.debug(f"Breaking section at {current_duration/60:.1f}min (time-based)")
                return True
            return False
        
        # Check if we've exceeded maximum duration
        if potential_duration > self.max_duration:
            if self.allow_semantic_overflow:
                # Check if there's a good semantic boundary nearby
                if self._has_semantic_boundary(current_beats[-1], next_beat):
                    logger.debug(
                        f"Breaking section at {current_duration/60:.1f}min "
                        f"(semantic boundary, duration={potential_duration/60:.1f}min)"
                    )
                    return True
                # Otherwise, keep going despite exceeding max
                logger.debug(f"Allowing semantic overflow beyond {self.max_duration/60:.1f}min")
                return False
            else:
                # Hard limit - must break
                logger.debug(f"Breaking section at {current_duration/60:.1f}min (max duration)")
                return True
        
        # Check if we're near target and there's a semantic boundary
        if current_duration >= self.target_duration * 0.8:  # Within 80% of target
            if self._has_semantic_boundary(current_beats[-1], next_beat):
                logger.debug(
                    f"Breaking section at {current_duration/60:.1f}min "
                    f"(semantic boundary near target)"
                )
                return True
        
        # Check for large time gap (natural break point)
        if current_beats:
            time_gap = next_beat.get("start_time", 0.0) - current_beats[-1].get("end_time", 0.0)
            if time_gap > 30.0:  # 30 second gap
                logger.debug(f"Breaking section due to {time_gap:.1f}s time gap")
                return True
        
        return False
    
    def _has_semantic_boundary(
        self,
        beat1: Dict[str, Any],
        beat2: Dict[str, Any],
    ) -> bool:
        """
        Check if there's a semantic boundary between two beats.
        
        Args:
            beat1: First beat
            beat2: Second beat
        
        Returns:
            True if semantic boundary exists, False otherwise
        """
        # If beats have embeddings, use similarity
        if "embedding" in beat1 and "embedding" in beat2:
            try:
                emb1 = np.array(beat1["embedding"])
                emb2 = np.array(beat2["embedding"])
                
                similarity = self._cosine_similarity(emb1, emb2)
                
                # Low similarity indicates topic shift (semantic boundary)
                if similarity < self.boundary_similarity_threshold:
                    logger.debug(f"Semantic boundary detected (similarity: {similarity:.3f})")
                    return True
                
                return False
            except Exception as e:
                logger.debug(f"Error computing similarity: {e}")
                return False
        
        # Without embeddings, use heuristics
        # Check for different topic labels if available
        if beat1.get("topic_label") and beat2.get("topic_label"):
            if beat1["topic_label"] != beat2["topic_label"]:
                return True
        
        return False
    
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
        
        # Clip to [0, 1] range
        return max(0.0, min(1.0, float(similarity)))
    
    def _create_section(
        self,
        episode_id: str,
        beats: List[Dict[str, Any]],
        position: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a section record from a list of beats.
        
        Args:
            episode_id: Episode identifier
            beats: List of contiguous beats
            position: Position of this section within the episode
        
        Returns:
            Section record dictionary or None if invalid
        """
        if not beats:
            return None
        
        # Compute time range
        start_time = beats[0].get("start_time", 0.0)
        end_time = beats[-1].get("end_time", 0.0)
        duration_minutes = (end_time - start_time) / 60.0
        
        # Concatenate text from all beats
        text = " ".join(beat.get("text", "") for beat in beats)
        
        # Collect beat IDs
        beat_ids = [beat.get("beat_id") for beat in beats]
        
        # Generate deterministic section ID
        section_id = generate_section_id(
            episode_id=episode_id,
            position=position,
            beat_ids=beat_ids,
            text=text,
        )
        
        # Build section record
        section = {
            "section_id": section_id,
            "episode_id": episode_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "text": text,
            "beat_ids": beat_ids,
        }
        
        return section
    
    def compute_statistics(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about generated sections.
        
        Args:
            sections: List of section records
        
        Returns:
            Statistics dictionary
        """
        if not sections:
            return {
                "count": 0,
                "artifact_type": self.get_artifact_type(),
            }
        
        # Compute duration statistics (in minutes)
        durations = [section.get("duration_minutes", 0.0) for section in sections]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0.0
        
        # Count beats per section
        beat_counts = [len(section.get("beat_ids", [])) for section in sections]
        total_beats = sum(beat_counts)
        avg_beats_per_section = total_beats / len(beat_counts) if beat_counts else 0.0
        
        # Check how many are within target range
        in_range = sum(1 for d in durations if self.min_duration_minutes <= d <= self.max_duration_minutes)
        in_range_pct = (in_range / len(durations) * 100) if durations else 0.0
        
        return {
            "count": len(sections),
            "artifact_type": self.get_artifact_type(),
            "total_duration_minutes": total_duration,
            "avg_duration_minutes": avg_duration,
            "min_duration_minutes": min(durations) if durations else 0.0,
            "max_duration_minutes": max(durations) if durations else 0.0,
            "total_beats": total_beats,
            "avg_beats_per_section": avg_beats_per_section,
            "in_target_range": in_range,
            "in_target_range_pct": in_range_pct,
        }


def generate_sections(
    beats: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate sections from beats.
    
    Args:
        beats: List of beat records
        config: Configuration dictionary
    
    Returns:
        List of section records
    
    Example:
        >>> sections = generate_sections(beats, config={"target_duration_minutes": 8.0})
        >>> sections[0]["section_id"]
        'sec_e1b4f4e5c2d9_000000_d4c5b6a7'
    """
    generator = SectionGenerator(config)
    return generator.aggregate(beats)

