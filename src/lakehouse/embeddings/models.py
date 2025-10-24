"""
Embedding model interfaces and implementations.

Provides abstract interface for embedding generation with concrete
implementations for local sentence-transformers models and OpenAI API.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    
    All embedding model implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name or identifier of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement encode()")
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (e.g., 384, 768, 1536)
        """
        raise NotImplementedError("Subclasses must implement get_embedding_dimension()")
    
    def get_model_info(self) -> dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "embedding_dimension": self.get_embedding_dimension(),
        }


class SentenceTransformerModel(EmbeddingModel):
    """
    Local sentence-transformers model wrapper.
    
    Uses the sentence-transformers library for local embedding generation.
    Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        """
        Initialize sentence-transformers model.
        
        Args:
            model_name: HuggingFace model name (default: "all-MiniLM-L6-v2")
            device: Device to use ("cpu", "cuda", "mps") or None for auto
            normalize_embeddings: Whether to normalize embeddings to unit length
            **kwargs: Additional parameters for SentenceTransformer
        """
        super().__init__(model_name, **kwargs)
        
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._embedding_dim = None
        
        # Lazy load model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                **self.config,
            )
            
            # Get embedding dimension
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Model loaded successfully: {self.model_name} "
                f"(dimension: {self._embedding_dim}, device: {self._model.device})"
            )
            
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s) using sentence-transformers.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings
        
        Example:
            >>> model = SentenceTransformerModel()
            >>> embeddings = model.encode(["Hello world", "Test text"])
            >>> embeddings.shape
            (2, 384)
        """
        if self._model is None:
            self._load_model()
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        
        logger.debug(f"Generated embeddings for {len(texts)} texts (dimension: {embeddings.shape[1]})")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None and self._model is not None:
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._embedding_dim or 384  # Default fallback


class OpenAIEmbeddingModel(EmbeddingModel):
    """
    OpenAI embeddings API wrapper.
    
    Uses OpenAI's embeddings API for cloud-based embedding generation.
    Default model: text-embedding-3-small (1536-dim)
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize OpenAI embeddings model.
        
        Args:
            model_name: OpenAI model name (default: "text-embedding-3-small")
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Additional parameters for OpenAI client
        """
        super().__init__(model_name, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._embedding_dim = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            
            # Set embedding dimension based on model
            dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._embedding_dim = dimension_map.get(self.model_name, 1536)
            
            logger.info(
                f"OpenAI client initialized with model: {self.model_name} "
                f"(dimension: {self._embedding_dim})"
            )
            
        except ImportError:
            logger.error(
                "openai package not installed. "
                "Install with: pip install openai"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s) using OpenAI API.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing (OpenAI supports up to 2048 texts)
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings
        
        Example:
            >>> model = OpenAIEmbeddingModel()
            >>> embeddings = model.encode(["Hello world", "Test text"])
            >>> embeddings.shape
            (2, 1536)
        """
        if self._client is None:
            self._init_client()
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if show_progress:
                    logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size}: {e}")
                raise
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        logger.debug(f"Generated embeddings for {len(texts)} texts (dimension: {embeddings.shape[1]})")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim or 1536


def create_embedding_model(
    provider: str = "local",
    model_name: Optional[str] = None,
    **kwargs,
) -> EmbeddingModel:
    """
    Factory function to create an embedding model.
    
    Args:
        provider: Model provider ("local" for sentence-transformers, "openai" for OpenAI)
        model_name: Model name (uses defaults if not provided)
        **kwargs: Additional parameters for the model
    
    Returns:
        EmbeddingModel instance
    
    Raises:
        ValueError: If provider is not recognized
    
    Example:
        >>> model = create_embedding_model("local")
        >>> model = create_embedding_model("openai", api_key="...")
    """
    provider = provider.lower()
    
    if provider == "local":
        default_model = "all-MiniLM-L6-v2"
        model_name = model_name or default_model
        return SentenceTransformerModel(model_name=model_name, **kwargs)
    
    elif provider == "openai":
        default_model = "text-embedding-3-small"
        model_name = model_name or default_model
        return OpenAIEmbeddingModel(model_name=model_name, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: 'local', 'openai'"
        )

