"""
Base classes and interfaces for aggregation strategies.

Provides abstract base classes for implementing different aggregation
strategies (spans, beats, sections) with a common interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class AggregationStrategy(ABC):
    """
    Abstract base class for aggregation strategies.
    
    All aggregation implementations (spans, beats, sections) should
    inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize aggregation strategy.
        
        Args:
            config: Configuration dictionary for the strategy
        """
        self.config = config or {}
        logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def aggregate(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform aggregation on input data.
        
        Args:
            input_data: List of input records to aggregate
        
        Returns:
            List of aggregated records
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement aggregate()")
    
    @abstractmethod
    def get_artifact_type(self) -> str:
        """
        Get the artifact type name for this aggregation strategy.
        
        Returns:
            Artifact type string (e.g., "span", "beat", "section")
        """
        raise NotImplementedError("Subclasses must implement get_artifact_type()")
    
    def validate_input(self, input_data: List[Dict[str, Any]]) -> bool:
        """
        Validate input data before aggregation.
        
        Default implementation checks that input is non-empty.
        Subclasses can override for more specific validation.
        
        Args:
            input_data: List of input records
        
        Returns:
            True if valid, False otherwise
        """
        if not input_data:
            logger.warning(f"{self.__class__.__name__}: Empty input data")
            return False
        
        return True
    
    def compute_statistics(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about aggregated data.
        
        Default implementation provides basic counts.
        Subclasses can override for more detailed statistics.
        
        Args:
            aggregated_data: List of aggregated records
        
        Returns:
            Dictionary with statistics
        """
        return {
            "count": len(aggregated_data),
            "artifact_type": self.get_artifact_type(),
        }
    
    def log_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Log statistics about aggregation results.
        
        Args:
            stats: Statistics dictionary
        """
        artifact_type = stats.get("artifact_type", "unknown")
        count = stats.get("count", 0)
        logger.info(f"Generated {count} {artifact_type} artifacts")


class UtteranceAggregator(AggregationStrategy):
    """
    Base class for aggregations that work on utterances.
    
    Provides common functionality for aggregating utterance-level data.
    """
    
    def validate_input(self, input_data: List[Dict[str, Any]]) -> bool:
        """
        Validate utterance input data.
        
        Args:
            input_data: List of utterance records
        
        Returns:
            True if valid, False otherwise
        """
        if not super().validate_input(input_data):
            return False
        
        # Check that utterances have required fields
        required_fields = {"utterance_id", "episode_id", "start", "end", "speaker", "text"}
        
        for idx, utterance in enumerate(input_data):
            missing = required_fields - set(utterance.keys())
            if missing:
                logger.error(
                    f"Utterance at index {idx} missing required fields: {missing}"
                )
                return False
        
        return True
    
    def group_by_episode(self, utterances: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group utterances by episode ID.
        
        Args:
            utterances: List of utterance records
        
        Returns:
            Dictionary mapping episode_id to list of utterances
        """
        episodes = {}
        
        for utterance in utterances:
            episode_id = utterance.get("episode_id")
            if not episode_id:
                logger.warning("Utterance missing episode_id, skipping")
                continue
            
            if episode_id not in episodes:
                episodes[episode_id] = []
            
            episodes[episode_id].append(utterance)
        
        logger.debug(f"Grouped {len(utterances)} utterances into {len(episodes)} episodes")
        return episodes
    
    def sort_by_time(self, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort utterances by start time.
        
        Args:
            utterances: List of utterance records
        
        Returns:
            Sorted list of utterances
        """
        return sorted(utterances, key=lambda u: u.get("start", 0.0))


class SpanAggregator(AggregationStrategy):
    """
    Base class for aggregations that work on spans.
    
    Provides common functionality for aggregating span-level data.
    """
    
    def validate_input(self, input_data: List[Dict[str, Any]]) -> bool:
        """
        Validate span input data.
        
        Args:
            input_data: List of span records
        
        Returns:
            True if valid, False otherwise
        """
        if not super().validate_input(input_data):
            return False
        
        # Check that spans have required fields
        required_fields = {"span_id", "episode_id", "start_time", "end_time", "speaker", "text"}
        
        for idx, span in enumerate(input_data):
            missing = required_fields - set(span.keys())
            if missing:
                logger.error(
                    f"Span at index {idx} missing required fields: {missing}"
                )
                return False
        
        return True
    
    def group_by_episode(self, spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group spans by episode ID.
        
        Args:
            spans: List of span records
        
        Returns:
            Dictionary mapping episode_id to list of spans
        """
        episodes = {}
        
        for span in spans:
            episode_id = span.get("episode_id")
            if not episode_id:
                logger.warning("Span missing episode_id, skipping")
                continue
            
            if episode_id not in episodes:
                episodes[episode_id] = []
            
            episodes[episode_id].append(span)
        
        logger.debug(f"Grouped {len(spans)} spans into {len(episodes)} episodes")
        return episodes
    
    def sort_by_time(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort spans by start time.
        
        Args:
            spans: List of span records
        
        Returns:
            Sorted list of spans
        """
        return sorted(spans, key=lambda s: s.get("start_time", 0.0))


class BeatAggregator(AggregationStrategy):
    """
    Base class for aggregations that work on beats.
    
    Provides common functionality for aggregating beat-level data.
    """
    
    def validate_input(self, input_data: List[Dict[str, Any]]) -> bool:
        """
        Validate beat input data.
        
        Args:
            input_data: List of beat records
        
        Returns:
            True if valid, False otherwise
        """
        if not super().validate_input(input_data):
            return False
        
        # Check that beats have required fields
        required_fields = {"beat_id", "episode_id", "start_time", "end_time", "text"}
        
        for idx, beat in enumerate(input_data):
            missing = required_fields - set(beat.keys())
            if missing:
                logger.error(
                    f"Beat at index {idx} missing required fields: {missing}"
                )
                return False
        
        return True
    
    def group_by_episode(self, beats: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group beats by episode ID.
        
        Args:
            beats: List of beat records
        
        Returns:
            Dictionary mapping episode_id to list of beats
        """
        episodes = {}
        
        for beat in beats:
            episode_id = beat.get("episode_id")
            if not episode_id:
                logger.warning("Beat missing episode_id, skipping")
                continue
            
            if episode_id not in episodes:
                episodes[episode_id] = []
            
            episodes[episode_id].append(beat)
        
        logger.debug(f"Grouped {len(beats)} beats into {len(episodes)} episodes")
        return episodes
    
    def sort_by_time(self, beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort beats by start time.
        
        Args:
            beats: List of beat records
        
        Returns:
            Sorted list of beats
        """
        return sorted(beats, key=lambda b: b.get("start_time", 0.0))

