"""
Unit tests for embedding generation modules.

Tests embedding model interfaces and generators with mocked models
to avoid requiring model downloads or API calls.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from lakehouse.embeddings.models import (
    EmbeddingModel,
    SentenceTransformerModel,
    OpenAIEmbeddingModel,
    create_embedding_model,
)
from lakehouse.embeddings.generator import (
    EmbeddingGenerator,
    generate_embeddings,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_texts() -> List[str]:
    """Create sample texts for embedding."""
    return [
        "This is the first test sentence.",
        "Here is another sentence for testing.",
        "Finally, a third sentence to embed.",
    ]


@pytest.fixture
def sample_spans():
    """Create sample span artifacts."""
    return [
        {
            "span_id": "spn_test_000000_abc",
            "episode_id": "EP1",
            "speaker": "Alice",
            "text": "This is the first span.",
            "start_time": 0.0,
            "end_time": 60.0,
        },
        {
            "span_id": "spn_test_000001_def",
            "episode_id": "EP1",
            "speaker": "Bob",
            "text": "This is the second span.",
            "start_time": 60.0,
            "end_time": 120.0,
        },
        {
            "span_id": "spn_test_000002_ghi",
            "episode_id": "EP2",
            "speaker": "Alice",
            "text": "This is the third span.",
            "start_time": 0.0,
            "end_time": 60.0,
        },
    ]


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = Mock(spec=EmbeddingModel)
    model.model_name = "mock-model"
    model.get_embedding_dimension.return_value = 384
    model.get_model_info.return_value = {
        "model_name": "mock-model",
        "model_type": "MockModel",
        "embedding_dimension": 384,
    }
    
    # Mock encode method to return random embeddings
    def mock_encode(texts, batch_size=32, show_progress=False):
        if isinstance(texts, str):
            texts = [texts]
        # Return random embeddings
        return np.random.rand(len(texts), 384).astype(np.float32)
    
    model.encode = Mock(side_effect=mock_encode)
    
    return model


# ============================================================================
# EmbeddingModel Tests
# ============================================================================

class TestEmbeddingModelInterface:
    """Test abstract EmbeddingModel interface."""
    
    def test_embedding_model_is_abstract(self):
        """Test that EmbeddingModel cannot be instantiated directly."""
        # Should be able to create mock or subclass
        model = Mock(spec=EmbeddingModel)
        assert hasattr(model, "encode")
        assert hasattr(model, "get_embedding_dimension")
    
    def test_get_model_info(self, mock_embedding_model):
        """Test getting model information."""
        info = mock_embedding_model.get_model_info()
        
        assert "model_name" in info
        assert "model_type" in info
        assert "embedding_dimension" in info


# ============================================================================
# SentenceTransformerModel Tests (Mocked)
# ============================================================================

class TestSentenceTransformerModel:
    """Test SentenceTransformerModel with mocked sentence-transformers."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_model_initialization(self, mock_st_class):
        """Test model initialization with mocked sentence-transformers."""
        # Setup mock
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_instance.device = "cpu"
        mock_st_class.return_value = mock_st_instance
        
        # Initialize model
        model = SentenceTransformerModel("all-MiniLM-L6-v2")
        
        assert model.model_name == "all-MiniLM-L6-v2"
        assert model.get_embedding_dimension() == 384
        mock_st_class.assert_called_once()
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_single_text(self, mock_st_class):
        """Test encoding a single text."""
        # Setup mock
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_instance.device = "cpu"
        mock_st_instance.encode.return_value = np.random.rand(1, 384)
        mock_st_class.return_value = mock_st_instance
        
        # Initialize and encode
        model = SentenceTransformerModel()
        embeddings = model.encode("Test text")
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 384
        mock_st_instance.encode.assert_called_once()
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_multiple_texts(self, mock_st_class, sample_texts):
        """Test encoding multiple texts."""
        # Setup mock
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_instance.device = "cpu"
        mock_st_instance.encode.return_value = np.random.rand(len(sample_texts), 384)
        mock_st_class.return_value = mock_st_instance
        
        # Initialize and encode
        model = SentenceTransformerModel()
        embeddings = model.encode(sample_texts)
        
        assert embeddings.shape[0] == len(sample_texts)
        assert embeddings.shape[1] == 384
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_batch_size_parameter(self, mock_st_class, sample_texts):
        """Test that batch_size parameter is used."""
        # Setup mock
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_instance.device = "cpu"
        mock_st_instance.encode.return_value = np.random.rand(len(sample_texts), 384)
        mock_st_class.return_value = mock_st_instance
        
        # Initialize and encode with custom batch size
        model = SentenceTransformerModel()
        embeddings = model.encode(sample_texts, batch_size=2)
        
        # Check that encode was called with batch_size parameter
        call_kwargs = mock_st_instance.encode.call_args[1]
        assert call_kwargs["batch_size"] == 2


# ============================================================================
# OpenAIEmbeddingModel Tests (Mocked)
# ============================================================================

class TestOpenAIEmbeddingModel:
    """Test OpenAIEmbeddingModel with mocked OpenAI API."""
    
    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key not provided"):
                OpenAIEmbeddingModel()
    
    @patch('openai.OpenAI')
    def test_initialization_with_api_key(self, mock_openai_class):
        """Test initialization with API key."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        model = OpenAIEmbeddingModel(api_key="test-key")
        
        assert model.model_name == "text-embedding-3-small"
        assert model.get_embedding_dimension() == 1536
        mock_openai_class.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_encode_single_text(self, mock_openai_class):
        """Test encoding a single text with OpenAI."""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=np.random.rand(1536).tolist())]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Initialize and encode
        model = OpenAIEmbeddingModel(api_key="test-key")
        embeddings = model.encode("Test text")
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 1536
        mock_client.embeddings.create.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_encode_batch(self, mock_openai_class, sample_texts):
        """Test encoding a batch of texts."""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=np.random.rand(1536).tolist()) 
            for _ in sample_texts
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Initialize and encode
        model = OpenAIEmbeddingModel(api_key="test-key")
        embeddings = model.encode(sample_texts)
        
        assert embeddings.shape[0] == len(sample_texts)
        assert embeddings.shape[1] == 1536
    
    @patch('openai.OpenAI')
    def test_batch_processing(self, mock_openai_class):
        """Test that large batches are processed in chunks."""
        # Create many texts
        many_texts = ["text " + str(i) for i in range(100)]
        
        # Setup mock client
        mock_client = MagicMock()
        
        def mock_create(**kwargs):
            batch_size = len(kwargs["input"])
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=np.random.rand(1536).tolist()) 
                for _ in range(batch_size)
            ]
            return mock_response
        
        mock_client.embeddings.create = Mock(side_effect=mock_create)
        mock_openai_class.return_value = mock_client
        
        # Initialize and encode with small batch size
        model = OpenAIEmbeddingModel(api_key="test-key")
        embeddings = model.encode(many_texts, batch_size=32)
        
        assert embeddings.shape[0] == len(many_texts)
        # Should have been called multiple times for batching
        assert mock_client.embeddings.create.call_count >= 3


# ============================================================================
# Model Factory Tests
# ============================================================================

class TestCreateEmbeddingModel:
    """Test the create_embedding_model factory function."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_create_local_model(self, mock_st_class):
        """Test creating a local model."""
        # Setup mock
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_instance.device = "cpu"
        mock_st_class.return_value = mock_st_instance
        
        model = create_embedding_model("local")
        
        assert isinstance(model, SentenceTransformerModel)
        assert model.model_name == "all-MiniLM-L6-v2"
    
    @patch('openai.OpenAI')
    def test_create_openai_model(self, mock_openai_class):
        """Test creating an OpenAI model."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        model = create_embedding_model("openai", api_key="test-key")
        
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == "text-embedding-3-small"
    
    def test_create_unknown_provider(self):
        """Test that unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_embedding_model("unknown")


# ============================================================================
# EmbeddingGenerator Tests
# ============================================================================

class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""
    
    def test_generator_initialization_default(self):
        """Test generator initialization with default config."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
            "generation": {"batch_size": 32},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_model = Mock(spec=EmbeddingModel)
            mock_model.model_name = "test-model"
            mock_model.get_model_info.return_value = {
                "model_name": "test-model",
                "model_type": "MockModel",
                "embedding_dimension": 384,
            }
            mock_create.return_value = mock_model
            
            generator = EmbeddingGenerator(config=config)
            
            assert generator.provider == "local"
            assert generator.model_name == "test-model"
            assert generator.batch_size == 32
    
    def test_generate_embeddings_for_spans(self, sample_spans, mock_embedding_model):
        """Test generating embeddings for spans."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
            "generation": {"batch_size": 32},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator(config=config)
            results = generator.generate(sample_spans, artifact_type="span")
            
            # Should generate embeddings for all spans
            assert len(results) == len(sample_spans)
            
            # Check result structure
            for result in results:
                assert "artifact_id" in result
                assert "artifact_type" in result
                assert "embedding" in result
                assert "model_name" in result
                assert result["artifact_type"] == "span"
                assert len(result["embedding"]) == 384
    
    def test_generate_embeddings_with_episode_grouping(self, sample_spans, mock_embedding_model):
        """Test generating embeddings grouped by episode."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
            "generation": {"batch_size": 32},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator(config=config)
            results_by_episode = generator.generate_for_episode(sample_spans, artifact_type="span")
            
            # Should group by episode
            assert "EP1" in results_by_episode
            assert "EP2" in results_by_episode
            assert len(results_by_episode["EP1"]) == 2
            assert len(results_by_episode["EP2"]) == 1
    
    def test_text_truncation(self, mock_embedding_model):
        """Test that long text is truncated."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
            "generation": {"batch_size": 32, "max_text_length": 100},
        }
        
        # Create span with very long text
        long_spans = [{
            "span_id": "spn_test",
            "episode_id": "EP1",
            "text": "x" * 1000,  # 1000 characters
        }]
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator(config=config)
            results = generator.generate(long_spans, artifact_type="span")
            
            # Should still generate embeddings
            assert len(results) == 1
            
            # Check that encode was called (text should be truncated internally)
            mock_embedding_model.encode.assert_called_once()
    
    def test_empty_artifacts_list(self, mock_embedding_model):
        """Test handling of empty artifacts list."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator(config=config)
            results = generator.generate([], artifact_type="span")
            
            assert len(results) == 0
    
    def test_get_model_info(self, mock_embedding_model):
        """Test getting model information from generator."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator(config=config)
            info = generator.get_model_info()
            
            assert "model_name" in info
            assert "model_type" in info
            assert "embedding_dimension" in info


# ============================================================================
# Integration Tests
# ============================================================================

class TestEmbeddingPipeline:
    """Test full embedding generation pipeline."""
    
    def test_convenience_function(self, sample_spans, mock_embedding_model):
        """Test convenience function for embedding generation."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
            "generation": {"batch_size": 32},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            results = generate_embeddings(
                sample_spans,
                artifact_type="span",
                config=config,
            )
            
            assert len(results) == len(sample_spans)
            assert all("embedding" in r for r in results)
    
    def test_embedding_dimensions_match(self, sample_spans, mock_embedding_model):
        """Test that all embeddings have same dimension."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            results = generate_embeddings(
                sample_spans,
                artifact_type="span",
                config=config,
            )
            
            # All embeddings should have same dimension
            dimensions = [len(r["embedding"]) for r in results]
            assert len(set(dimensions)) == 1
            assert dimensions[0] == 384
    
    def test_artifact_id_mapping(self, sample_spans, mock_embedding_model):
        """Test that embeddings are correctly mapped to artifact IDs."""
        config = {
            "model": {"provider": "local", "name": "test-model"},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            results = generate_embeddings(
                sample_spans,
                artifact_type="span",
                config=config,
            )
            
            # Check that artifact IDs match
            result_ids = [r["artifact_id"] for r in results]
            span_ids = [s["span_id"] for s in sample_spans]
            
            assert result_ids == span_ids
    
    def test_different_artifact_types(self, mock_embedding_model):
        """Test embedding generation for different artifact types."""
        beats = [
            {
                "beat_id": "bet_test_000000",
                "episode_id": "EP1",
                "text": "Beat content",
            }
        ]
        
        config = {
            "model": {"provider": "local", "name": "test-model"},
        }
        
        with patch('lakehouse.embeddings.generator.create_embedding_model') as mock_create:
            mock_create.return_value = mock_embedding_model
            
            results = generate_embeddings(
                beats,
                artifact_type="beat",
                config=config,
            )
            
            assert len(results) == 1
            assert results[0]["artifact_type"] == "beat"
            assert results[0]["artifact_id"] == "bet_test_000000"

