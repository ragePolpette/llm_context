from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

import os
import re

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
    assets_root_dir: str = ""
    assets_template_dir: str = ""
    scope_map: dict[str, list[str]] = field(default_factory=dict)
    path_aliases: dict[str, str] = field(default_factory=dict)
    product_targets: dict[str, list[str]] = field(default_factory=dict)
    product_aliases: dict[str, str] = field(default_factory=dict)
    fts_config: str = "simple"
    incremental_ingest: bool = True
    vector_weight: float = 0.5
    keyword_weight: float = 0.5
    max_chunks_per_doc: int = 5
    min_score: float = 0.20
    min_score_floor: float = 0.15
    header_penalty: float = 0.05
    default_doc_type: str = "code"
    exclude_globs: list[str] = field(default_factory=list)
    never_index_ext: list[str] = field(default_factory=list)
    symbol_search_enabled: bool = True
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
    resolved_path = Path(path).resolve()
    data = _load_config_payload(resolved_path, seen_paths=set())
    config = AppConfig()
    type_hints = get_type_hints(AppConfig)
    for config_field in fields(config):
        if config_field.name in data:
            setattr(
                config,
                config_field.name,
                _coerce_value(data[config_field.name], type_hints[config_field.name]),
            )
    config.fts_config = validate_fts_config(config.fts_config)
    config_dir = resolved_path.parent
    config.projects_registry_path = _resolve_path(config.projects_registry_path, config_dir)
    config.projects_state_path = _resolve_path(config.projects_state_path, config_dir)
    _apply_env_overrides(config)
    return config


def _load_config_payload(path: Path, *, seen_paths: set[Path]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in seen_paths:
        raise ValueError(f"Config extends cycle detected for: {resolved_path}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {resolved_path}")

    extends_value = data.get("extends")
    current_seen = set(seen_paths)
    current_seen.add(resolved_path)
    merged: dict[str, Any] = {}
    if extends_value is not None:
        extends_path = Path(str(extends_value)).expanduser()
        if not extends_path.is_absolute():
            extends_path = (resolved_path.parent / extends_path).resolve()
        merged.update(_load_config_payload(extends_path, seen_paths=current_seen))

    for key, value in data.items():
        if key == "extends":
            continue
        merged[key] = value
    return merged


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
        args = get_args(target_type)
        value_type = args[1] if len(args) == 2 else Any
        return _coerce_dict(value, value_type)
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


def _coerce_dict(value: Any, value_type: Any = Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if get_origin(value_type) is list:
        return {str(key): _coerce_list(items) for key, items in value.items()}
    if value_type is str:
        return {str(key): "" if item is None else str(item) for key, item in value.items()}
    return {str(key): item for key, item in value.items()}


_FTS_CONFIG_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")


def validate_fts_config(value: Any) -> str:
    """Validate a PostgreSQL text search configuration name (e.g. simple, english)."""
    name = str(value or "").strip()
    if not _FTS_CONFIG_PATTERN.match(name):
        raise ValueError(f"Invalid fts_config: {value!r}")
    return name


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
