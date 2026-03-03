from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional, get_args, get_origin

import os

import yaml


DEFAULT_INCLUDE_EXTENSIONS = [
    ".md",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".cs",
    ".csproj",
    ".asp",
    ".aspx",
    ".ashx",
    ".config",
]

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "__pycache__",
    "bin",
    "obj",
    ".vs",
    "packages",
    "testresults",
    "third-party-notices.txt",
]


@dataclass
class AppConfig:
    embedding_dim: int = 384
    ivfflat_lists: int = 100
    max_bytes: int = 2_000_000
    chunk_size: int = 1200
    chunk_overlap: int = 300
    md_chunk_size: int = 1600
    code_chunk_size: int = 2500
    min_chunk_chars: int = 200
    include_dirs: list[str] = field(default_factory=list)
    assets_template_only: bool = False
    assets_root_dir: str = "pubblico/assets"
    assets_template_dir: str = "pubblico/assets/template"
    scope_map: dict[str, list[str]] = field(default_factory=dict)
    incremental_ingest: bool = True
    vector_weight: float = 0.5
    keyword_weight: float = 0.5
    max_chunks_per_doc: int = 5
    min_score: float = 0.20
    header_penalty: float = 0.05
    rerank_enabled: bool = False
    default_doc_type: str = "code"
    exclude_globs: list[str] = field(default_factory=list)
    never_index_ext: list[str] = field(default_factory=list)


def parse_csv_list(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def load_config(path: Optional[str]) -> AppConfig:
    if path is None:
        return AppConfig()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    config = AppConfig()
    for field in fields(config):
        if field.name in data:
            setattr(config, field.name, _coerce_value(data[field.name], field.type))
    return config


def _coerce_value(value: Any, target_type: Any) -> Any:
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    if target_type is bool:
        return _coerce_bool(value)
    origin = get_origin(target_type)
    if origin is list:
        return _coerce_list(value)
    if origin is dict:
        return _coerce_dict(value)
    return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return parse_csv_list(value)
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _coerce_dict(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        result: dict[str, list[str]] = {}
        for key, items in value.items():
            result[str(key)] = _coerce_list(items)
        return result
    return {}
