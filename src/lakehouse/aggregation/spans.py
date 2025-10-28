"""
Span generation from utterances.

Consolidates single-speaker contiguous utterances into spans,
maintaining references to constituent utterances and computing
accurate timestamps and durations.
"""

from typing import Any, Dict, List, Optional

from lakehouse.aggregation.base import UtteranceAggregator
from lakehouse.ids import generate_span_id
from lakehouse.logger import get_default_logger
from lakehouse.speaker_roles import enrich_spans_with_speaker_metadata  # Task 5.6
from lakehouse.config import load_speaker_roles  # Task 5.6


logger = get_default_logger()


class SpanGenerator(UtteranceAggregator):
    """
    Generates spans from utterances.
    
    A span is a contiguous sequence of utterances from a single speaker.
    Spans break on speaker changes or when there's a significant silence gap.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize span generator.
        
        Args:
            config: Configuration dictionary with options:
                - min_duration: Minimum span duration in seconds (default: 1.0)
                - max_silence_gap: Maximum silence gap to still be contiguous (default: 0.5)
                - break_on_speaker_change: Break spans on speaker change (default: True)
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.min_duration = self.config.get("min_duration", 1.0)
        self.max_silence_gap = self.config.get("max_silence_gap", 0.5)
        self.break_on_speaker_change = self.config.get("break_on_speaker_change", True)
        
        logger.debug(
            f"SpanGenerator initialized: min_duration={self.min_duration}s, "
            f"max_silence_gap={self.max_silence_gap}s, "
            f"break_on_speaker_change={self.break_on_speaker_change}"
        )
    
    def get_artifact_type(self) -> str:
        """Get artifact type name."""
        return "span"
    
    def aggregate(self, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate spans from utterances.
        
        Args:
            utterances: List of utterance records (must all be from same episode)
        
        Returns:
            List of span records
        
        Example:
            >>> generator = SpanGenerator()
            >>> spans = generator.aggregate(utterances)
            >>> len(spans) < len(utterances)  # Spans consolidate utterances
            True
        """
        if not self.validate_input(utterances):
            return []
        
        # Group by episode and process each separately
        episodes = self.group_by_episode(utterances)
        
        all_spans = []
        for episode_id, episode_utterances in episodes.items():
            logger.info(f"Generating spans for episode {episode_id} ({len(episode_utterances)} utterances)")
            
            # Sort utterances by time
            sorted_utterances = self.sort_by_time(episode_utterances)
            
            # Generate spans for this episode
            episode_spans = self._generate_spans_for_episode(episode_id, sorted_utterances)
            all_spans.extend(episode_spans)
            
            logger.info(f"Generated {len(episode_spans)} spans from {len(episode_utterances)} utterances for {episode_id}")
        
        # Task 5.6: Enrich spans with speaker metadata (speaker_canonical, speaker_role, is_expert)
        try:
            speaker_config = load_speaker_roles()
            all_spans = enrich_spans_with_speaker_metadata(all_spans, speaker_config)
            logger.info(f"Enriched {len(all_spans)} spans with speaker metadata")
        except Exception as e:
            logger.warning(f"Failed to enrich spans with speaker metadata: {e}")
            # Add default values if enrichment fails
            for span in all_spans:
                if 'speaker_canonical' not in span:
                    span['speaker_canonical'] = span.get('speaker', 'Unknown')
                if 'speaker_role' not in span:
                    span['speaker_role'] = 'other'
                if 'is_expert' not in span:
                    span['is_expert'] = False
        
        # Compute and log statistics
        stats = self.compute_statistics(all_spans)
        self.log_statistics(stats)
        
        return all_spans
    
    def _generate_spans_for_episode(
        self,
        episode_id: str,
        utterances: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate spans for a single episode.
        
        Args:
            episode_id: Episode identifier
            utterances: Sorted list of utterances from one episode
        
        Returns:
            List of span records
        """
        if not utterances:
            return []
        
        spans = []
        current_span_utterances = [utterances[0]]
        
        for i in range(1, len(utterances)):
            current_utt = utterances[i]
            previous_utt = utterances[i - 1]
            
            # Check if we should start a new span
            should_break = self._should_break_span(previous_utt, current_utt)
            
            if should_break:
                # Finish current span and start new one
                span = self._create_span(episode_id, current_span_utterances, len(spans))
                if span:
                    spans.append(span)
                
                # Start new span
                current_span_utterances = [current_utt]
            else:
                # Continue current span
                current_span_utterances.append(current_utt)
        
        # Don't forget the last span
        if current_span_utterances:
            span = self._create_span(episode_id, current_span_utterances, len(spans))
            if span:
                spans.append(span)
        
        return spans
    
    def _should_break_span(
        self,
        previous_utt: Dict[str, Any],
        current_utt: Dict[str, Any],
    ) -> bool:
        """
        Determine if a span should break between two utterances.
        
        Args:
            previous_utt: Previous utterance
            current_utt: Current utterance
        
        Returns:
            True if span should break, False otherwise
        """
        # Break on speaker change
        if self.break_on_speaker_change:
            if previous_utt.get("speaker") != current_utt.get("speaker"):
                return True
        
        # Break on large silence gap
        silence_gap = current_utt.get("start", 0.0) - previous_utt.get("end", 0.0)
        if silence_gap > self.max_silence_gap:
            logger.debug(f"Breaking span due to {silence_gap:.2f}s silence gap")
            return True
        
        return False
    
    def _create_span(
        self,
        episode_id: str,
        utterances: List[Dict[str, Any]],
        position: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a span record from a list of utterances.
        
        Args:
            episode_id: Episode identifier
            utterances: List of contiguous utterances
            position: Position of this span within the episode
        
        Returns:
            Span record dictionary or None if invalid
        """
        if not utterances:
            return None
        
        # Get speaker (should be same for all utterances in span)
        speaker = utterances[0].get("speaker", "Unknown")
        
        # Compute time range
        start_time = utterances[0].get("start", 0.0)
        end_time = utterances[-1].get("end", 0.0)
        duration = end_time - start_time
        
        # Check minimum duration
        if duration < self.min_duration:
            logger.debug(f"Skipping span with duration {duration:.2f}s (< {self.min_duration}s)")
            return None
        
        # Concatenate text
        text = " ".join(utt.get("text", "") for utt in utterances)
        
        # Collect utterance IDs
        utterance_ids = [utt.get("utterance_id") for utt in utterances]
        
        # Generate deterministic span ID
        span_id = generate_span_id(
            episode_id=episode_id,
            position=position,
            speaker=speaker,
            utterance_ids=utterance_ids,
            text=text,
        )
        
        # Build span record
        span = {
            "span_id": span_id,
            "episode_id": episode_id,
            "speaker": speaker,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "text": text,
            "utterance_ids": utterance_ids,
        }
        
        return span
    
    def compute_statistics(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about generated spans.
        
        Args:
            spans: List of span records
        
        Returns:
            Statistics dictionary
        """
        if not spans:
            return {
                "count": 0,
                "artifact_type": self.get_artifact_type(),
            }
        
        # Compute duration statistics
        durations = [span.get("duration", 0.0) for span in spans]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0.0
        
        # Count utterances per span
        utterance_counts = [len(span.get("utterance_ids", [])) for span in spans]
        total_utterances = sum(utterance_counts)
        avg_utterances_per_span = total_utterances / len(utterance_counts) if utterance_counts else 0.0
        
        # Count unique speakers
        speakers = set(span.get("speaker") for span in spans)
        
        return {
            "count": len(spans),
            "artifact_type": self.get_artifact_type(),
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "min_duration": min(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "total_utterances": total_utterances,
            "avg_utterances_per_span": avg_utterances_per_span,
            "unique_speakers": len(speakers),
        }


def generate_spans(
    utterances: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate spans from utterances.
    
    Args:
        utterances: List of utterance records
        config: Configuration dictionary
    
    Returns:
        List of span records
    
    Example:
        >>> spans = generate_spans(utterances)
        >>> spans[0]["span_id"]
        'spn_e1b4f4e5c2d9_000000_f3e2d1c0'
    """
    generator = SpanGenerator(config)
    return generator.aggregate(utterances)

