from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
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
    symbol_search_enabled: bool = True
    multi_project_enabled: bool = False
    default_project_id: str = "myproj"
    projects_registry_path: Path = field(default_factory=lambda: Path("projects.yaml"))
    projects_state_path: Path = field(default_factory=lambda: Path("projects.state.json"))
    ingest_enabled: bool = False
    write_enabled: bool = False


def parse_csv_list(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def load_config(path: Optional[str]) -> AppConfig:
    if path is None:
        config = AppConfig()
        _apply_env_overrides(config)
        return config
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    config = AppConfig()
    for field in fields(config):
        if field.name in data:
            setattr(config, field.name, _coerce_value(data[field.name], field.type))
    config_dir = Path(path).resolve().parent
    config.projects_registry_path = _resolve_path(config.projects_registry_path, config_dir)
    config.projects_state_path = _resolve_path(config.projects_state_path, config_dir)
    _apply_env_overrides(config)
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
    if target_type is Path:
        return Path(value)
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


def _resolve_path(value: Path | str, base_dir: Path) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _apply_env_overrides(config: AppConfig) -> None:
    write_enabled_raw = os.getenv("LLM_CONTEXT_WRITE_ENABLED")
    ingest_enabled_raw = os.getenv("LLM_CONTEXT_INGEST_ENABLED")

    if write_enabled_raw is not None:
        config.write_enabled = _coerce_bool(write_enabled_raw)
    if ingest_enabled_raw is not None:
        config.ingest_enabled = _coerce_bool(ingest_enabled_raw)
    elif write_enabled_raw is not None:
        # Dashboard/runtime can expose a single write toggle. When explicitly
        # enabled, keep ingest aligned unless a more specific ingest override
        # is provided.
        config.ingest_enabled = config.write_enabled
