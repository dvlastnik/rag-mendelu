import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastembed import TextEmbedding

from text_embedding.base import BaseDenseEmbeddingLibrary
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedLibrary(BaseDenseEmbeddingLibrary):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str) -> TextEmbedding:
        try:
            return TextEmbedding(model_name=model_name)
        except Exception as e:
            if self._is_corrupt_cache_error(e):
                logger.warning(
                    "Corrupt fastembed cache detected for '%s'. Clearing and retrying...",
                    model_name,
                )
                self._clear_model_cache(model_name)
                return TextEmbedding(model_name=model_name)
            raise

    @staticmethod
    def _is_corrupt_cache_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in ("no such file", "file doesn't exist", "nosuchfile"))

    @staticmethod
    def _clear_model_cache(model_name: str) -> None:
        hf_repo = ""
        for info in TextEmbedding.list_supported_models():
            if info.get("model") == model_name:
                hf_repo = (info.get("sources") or {}).get("hf", "")
                break
        if not hf_repo:
            logger.warning(
                "Could not find HuggingFace repo for '%s'; skipping cache clear.", model_name
            )
            return

        cache_key = "models--" + hf_repo.replace("/", "--")
        candidate_roots = [
            d for d in [
                os.environ.get("FASTEMBED_CACHE_DIR"),
                os.path.join(tempfile.gettempdir(), "fastembed_cache"),
                str(Path.home() / ".cache" / "fastembed"),
            ]
            if d
        ]
        for root in candidate_roots:
            target = Path(root) / cache_key
            if target.exists():
                logger.info("Removing corrupt cache at %s", target)
                shutil.rmtree(target)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [embedding.tolist() for embedding in self.model.embed(texts)]

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def get_current_model(self) -> str:
        return self.model_name

    def get_embedding_dim(self) -> int:
        info = self._get_model_info(self.model_name)
        if info and 'dim' in info:
            return info['dim']
        # Fallback: embed a test string and measure
        test = list(self.model.embed(["test"]))
        return len(test[0])

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Return True if *model_name* is in fastembed's known model list."""
        return any(m['model'] == model_name for m in TextEmbedding.list_supported_models())

    @staticmethod
    def _get_model_info(model_name: str) -> Optional[dict]:
        for m in TextEmbedding.list_supported_models():
            if m['model'] == model_name:
                return m
        return None
