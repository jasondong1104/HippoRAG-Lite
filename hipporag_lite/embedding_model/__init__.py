from .base import EmbeddingConfig, BaseEmbeddingModel
from .Qwen3 import Qwen3EmbeddingModel
from .Qwen import QwenEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "Qwen/Qwen3-Embedding-8B" in embedding_model_name:
        return Qwen3EmbeddingModel
    elif 'text-embedding' in embedding_model_name:
        return QwenEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"