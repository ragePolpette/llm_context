from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import importlib
import math
import os
import re
import threading
from typing import Any, Optional


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class DummyEmbedder(Embedder):
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("DummyEmbedder cannot produce embeddings; configure a real embedder such as local-st")


class LocalHashEmbedder(Embedder):
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be > 0")

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        vector = [0.0 for _ in range(self.dim)]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class LocalSentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.model_source = model_name
        self.model_name = normalize_embedding_model_id(model_name)
        self.device = device
        try:
            st_module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use local embeddings"
            ) from exc
        model_class = getattr(st_module, "SentenceTransformer", None)
        if model_class is None:
            raise RuntimeError("SentenceTransformer not available")
        if self.device:
            self.model = model_class(self.model_source, device=self.device)
        else:
            self.model = model_class(self.model_source)
        self.batch_size = _get_env_int("ST_BATCH_SIZE", None, 32)
        self.normalize = _get_env_bool("ST_NORMALIZE", None, True)
        # HF fast tokenizers raise "Already borrowed" when shared across threads.
        self._encode_lock = threading.Lock()

    def _encode(self, texts: list[str]) -> Any:
        with self._encode_lock:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
            )

    def embed(self, text: str) -> list[float]:
        return _to_float_list(self._encode([text])[0])

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_to_float_list(vector) for vector in self._encode(texts)]


def normalize_embedding_model_id(model_name: Any) -> str:
    """Return the stable model id used as DB key for embedding rows."""
    raw = str(model_name or "").strip()
    if not raw:
        return raw
    normalized_path = raw.replace("\\", "/")
    match = re.search(r"(?:^|/)models--([^/]+?)(?:/snapshots/|$)", normalized_path)
    if match:
        return match.group(1).replace("--", "/")
    return raw


def _to_float_list(values: Any) -> list[float]:
    if isinstance(values, list):
        return [float(value) for value in values]
    if isinstance(values, tuple):
        return [float(value) for value in values]
    if hasattr(values, "__iter__"):
        return [float(value) for value in values]
    raise RuntimeError("Embedding values are not iterable")


def _get_env_int(key: str, override: Optional[int], default: int) -> int:
    if override is not None:
        return int(override)
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_bool(key: str, override: Optional[bool], default: bool) -> bool:
    if override is not None:
        return bool(override)
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default
