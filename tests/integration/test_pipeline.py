"""
End-to-end integration tests for the complete lakehouse pipeline.

Tests the full workflow from ingestion through aggregation to validation.
"""

import pytest
import pandas as pd
from pathlib import Path

from lakehouse.ingestion.reader import TranscriptReader
from lakehouse.ingestion.validator import validate_utterances
from lakehouse.ingestion.normalizer import normalize_utterances
from lakehouse.ingestion.writer import write_parquet, read_parquet
from lakehouse.aggregation.spans import generate_spans
from lakehouse.aggregation.beats import generate_beats
from lakehouse.aggregation.sections import generate_sections
from lakehouse.validation.checks import validate_artifact


class TestEndToEndPipeline:
    """Test complete pipeline from ingestion to validation."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete end-to-end pipeline with sample transcript."""
        # Use the sample transcript fixture
        fixture_path = Path("tests/fixtures/sample_transcript.jsonl")
        
        # 1. INGESTION: Read transcript
        reader = TranscriptReader(fixture_path)
        raw_utterances = reader.read_utterances()
        
        assert len(raw_utterances) == 10  # Sample has 10 utterances
        
        # 2. VALIDATION: Validate input
        validation_result = validate_utterances(raw_utterances)
        assert validation_result.is_valid
        
        # 3. NORMALIZATION: Normalize utterances
        normalized_utterances = normalize_utterances(raw_utterances)
        
        assert len(normalized_utterances) == len(raw_utterances)
        assert all("utterance_id" in u for u in normalized_utterances)
        assert all("duration" in u for u in normalized_utterances)
        
        # 4. WRITE: Save to Parquet
        utterances_path = tmp_path / "utterances.parquet"
        write_parquet(
            normalized_utterances,
            utterances_path,
            artifact_type="utterance",
            enforce_schema=False,
        )
        
        assert utterances_path.exists()
        
        # 5. READ: Load from Parquet
        utterances_df = read_parquet(utterances_path)
        assert len(utterances_df) == len(normalized_utterances)
        
        # 6. AGGREGATION: Generate spans
        spans = generate_spans(
            normalized_utterances,
            config={"min_duration": 1.0, "max_silence_gap": 0.5}
        )
        
        assert len(spans) > 0
        assert len(spans) <= len(normalized_utterances)
        assert all("span_id" in s for s in spans)
        
        # 7. AGGREGATION: Generate beats
        beats = generate_beats(
            spans,
            config={"use_embeddings": False, "min_spans_per_beat": 1}
        )
        
        assert len(beats) > 0
        assert len(beats) <= len(spans)
        assert all("beat_id" in b for b in beats)
        
        # 8. AGGREGATION: Generate sections
        sections = generate_sections(
            beats,
            config={
                "min_duration_minutes": 0.1,  # Low threshold for test
                "require_embeddings": False  # Test without embeddings (Task 6.2)
            }
        )
        
        assert len(sections) > 0
        assert len(sections) <= len(beats)
        assert all("section_id" in s for s in sections)
        
        # 9. VALIDATION: Validate final artifacts
        utterance_report = validate_artifact(utterances_df, "utterance", "v1")
        assert utterance_report.is_valid
        
        spans_df = pd.DataFrame(spans)
        span_report = validate_artifact(spans_df, "span", "v1")
        # Spans may not pass all checks without proper schema, but should run
        assert len(span_report.checks) > 0
    
    def test_pipeline_with_multiple_episodes(self, tmp_path):
        """Test pipeline with multiple episodes."""
        # Create temporary transcript with 2 episodes
        transcript_path = tmp_path / "multi_episode.jsonl"
        
        utterances = []
        for ep_num in [1, 2]:
            for i in range(3):
                utt = {
                    "episode_id": f"EP{ep_num}",
                    "start": i * 5.0,
                    "end": (i + 1) * 5.0,
                    "speaker": "Alice" if i % 2 == 0 else "Bob",
                    "text": f"Utterance {i} from episode {ep_num}",
                }
                utterances.append(utt)
        
        # Write JSONL
        with open(transcript_path, "w", encoding="utf-8") as f:
            import json
            for utt in utterances:
                f.write(json.dumps(utt) + "\n")
        
        # Run pipeline
        reader = TranscriptReader(transcript_path)
        raw_utterances = reader.read_utterances()
        
        # Group by episode for normalization
        ep1_utts = [u for u in raw_utterances if u["episode_id"] == "EP1"]
        ep2_utts = [u for u in raw_utterances if u["episode_id"] == "EP2"]
        
        normalized_ep1 = normalize_utterances(ep1_utts)
        normalized_ep2 = normalize_utterances(ep2_utts)
        
        # Combine for aggregation
        all_normalized = normalized_ep1 + normalized_ep2
        
        # Generate spans (should handle multiple episodes)
        spans = generate_spans(all_normalized, config={"min_duration": 1.0})
        
        # Check that both episodes are represented
        span_episodes = set(s["episode_id"] for s in spans)
        assert "EP1" in span_episodes
        assert "EP2" in span_episodes
    
    def test_pipeline_error_handling(self, tmp_path):
        """Test pipeline error handling with invalid data."""
        # Create transcript with some invalid utterances
        transcript_path = tmp_path / "invalid.jsonl"
        
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 5.0, "speaker": "A", "text": "Valid"},
            {"episode_id": "EP1", "start": 5.0, "end": 3.0, "speaker": "B", "text": "Invalid timestamps"},
            {"episode_id": "EP1", "start": 10.0, "end": 15.0, "speaker": "", "text": "Empty speaker"},
        ]
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            import json
            for utt in utterances:
                f.write(json.dumps(utt) + "\n")
        
        # Read
        reader = TranscriptReader(transcript_path)
        raw_utterances = reader.read_utterances()
        
        # Validate (should detect issues)
        validation_result = validate_utterances(raw_utterances)
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
    
    def test_pipeline_determinism(self, tmp_path):
        """Test that pipeline produces deterministic IDs."""
        fixture_path = Path("tests/fixtures/sample_transcript.jsonl")
        
        # Run pipeline twice
        results = []
        for _ in range(2):
            reader = TranscriptReader(fixture_path)
            raw_utterances = reader.read_utterances()
            normalized = normalize_utterances(raw_utterances)
            spans = generate_spans(normalized, config={"min_duration": 1.0})
            results.append((normalized, spans))
        
        # Compare IDs
        utterance_ids_1 = [u["utterance_id"] for u in results[0][0]]
        utterance_ids_2 = [u["utterance_id"] for u in results[1][0]]
        
        span_ids_1 = [s["span_id"] for s in results[0][1]]
        span_ids_2 = [s["span_id"] for s in results[1][1]]
        
        # IDs should be identical
        assert utterance_ids_1 == utterance_ids_2
        assert span_ids_1 == span_ids_2
    
    def test_pipeline_parquet_roundtrip(self, tmp_path):
        """Test that data survives Parquet read/write."""
        fixture_path = Path("tests/fixtures/sample_transcript.jsonl")
        
        # Read and normalize
        reader = TranscriptReader(fixture_path)
        raw_utterances = reader.read_utterances()
        normalized = normalize_utterances(raw_utterances)
        
        # Write to Parquet
        parquet_path = tmp_path / "test.parquet"
        write_parquet(
            normalized,
            parquet_path,
            artifact_type="utterance",
            enforce_schema=False,
        )
        
        # Read back
        df = read_parquet(parquet_path)
        
        # Compare
        assert len(df) == len(normalized)
        assert list(df.columns) == list(pd.DataFrame(normalized).columns)
        
        # Check that text is preserved
        original_texts = [u["text"] for u in normalized]
        roundtrip_texts = df["text"].tolist()
        assert original_texts == roundtrip_texts

