from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import importlib
import logging
import math
import os
import random
import re
import time
from typing import Any, Optional


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class DummyEmbedder(Embedder):
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("Integra qui il provider embeddings")


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

    def embed(self, text: str) -> list[float]:
        vectors = self.model.encode(
            [text],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
        )
        return _to_float_list(vectors[0])

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
        )
        return [_to_float_list(vector) for vector in vectors]


class GeminiEmbedder(Embedder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Gemini disabilitato: questa installazione usa solo embedder locali.
        raise RuntimeError("Gemini disabilitato in questa installazione")

    def embed(self, text: str) -> list[float]:
        # Metodo non disponibile quando Gemini è disabilitato.
        raise RuntimeError("Gemini disabilitato in questa installazione")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        # Metodo non disponibile quando Gemini è disabilitato.
        raise RuntimeError("Gemini disabilitato in questa installazione")


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


def _extract_embeddings_values(response: Any) -> list[list[float]]:
    if response is None:
        raise RuntimeError("Empty embedding response")
    if hasattr(response, "embeddings") and response.embeddings:
        return [_extract_values(item) for item in response.embeddings]
    if isinstance(response, dict):
        if "embeddings" in response and response["embeddings"]:
            return [_extract_values(item) for item in response["embeddings"]]
    embedding_attr = getattr(response, "embedding", None)
    if embedding_attr is not None:
        return [_extract_values(embedding_attr)]
    if isinstance(response, dict):
        if "embedding" in response:
            return [_extract_values(response["embedding"])]
    raise RuntimeError("Unexpected embedding response format")


def _extract_values(obj: Any) -> list[float]:
    if isinstance(obj, dict):
        if "values" in obj:
            return _to_float_list(obj["values"])
        if "embedding" in obj:
            embedding = obj["embedding"]
            if isinstance(embedding, dict) and "values" in embedding:
                return _to_float_list(embedding["values"])
        raise RuntimeError("Embedding values not found")
    values_attr = getattr(obj, "values", None)
    if values_attr is not None:
        if callable(values_attr):
            return _to_float_list(values_attr())
        return _to_float_list(values_attr)
    inner = getattr(obj, "embedding", None)
    if inner is not None:
        inner_values = getattr(inner, "values", None)
        if inner_values is not None:
            if callable(inner_values):
                return _to_float_list(inner_values())
            return _to_float_list(inner_values)
    raise RuntimeError("Embedding values not found")


def _to_float_list(values: Any) -> list[float]:
    if isinstance(values, list):
        return [float(value) for value in values]
    if isinstance(values, tuple):
        return [float(value) for value in values]
    if hasattr(values, "__iter__"):
        return [float(value) for value in values]
    raise RuntimeError("Embedding values are not iterable")


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    if "unavailable" in message or "rate limit" in message:
        return True
    return False


def _compute_backoff(attempt: int, base: float, cap: float) -> float:
    delay = min(cap, base * (2**attempt))
    jitter = min(1.0, delay * 0.25)
    return delay + random.uniform(0.0, jitter)


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


def _get_env_float(key: str, override: Optional[float], default: float) -> float:
    if override is not None:
        return float(override)
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
