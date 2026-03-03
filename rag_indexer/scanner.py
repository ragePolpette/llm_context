from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import fnmatch
import os


SOURCE_TYPE_MAP = {
    ".md": "md",
    ".txt": "text",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".cs": "code",
    ".aspx": "code",
    ".ashx": "code",
    ".json": "config",
    ".yaml": "config",
    ".yml": "config",
    ".toml": "config",
}


def scan_files(
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    include_dirs: Optional[list[str]] = None,
    exclude_globs: Optional[list[str]] = None,
    never_index_ext: Optional[list[str]] = None,
    assets_template_only: bool = False,
    assets_root_dir: str = "pubblico/assets",
    assets_template_dir: str = "pubblico/assets/template",
) -> Iterable[Path]:
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if not _is_excluded_name(name, exclude_patterns)
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            rel_path = _relative_posix(path, root)
            if include_dirs and not _is_included_path(rel_path, include_dirs):
                continue
            if _is_excluded_path(path, exclude_patterns):
                continue
            if exclude_globs and _matches_globs(rel_path, exclude_globs):
                continue
            if never_index_ext and path.suffix.lower() in {
                ext.lower() for ext in never_index_ext if ext
            }:
                continue
            if assets_template_only and _is_assets_non_template(
                rel_path, assets_root_dir, assets_template_dir
            ):
                continue
            if not _should_include(path, include_patterns):
                continue
            yield path


def _decode_text(data: bytes) -> str:
    """Try UTF-8, then Windows-1252, then latin-1 as fallback."""
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return data.decode("utf-8", errors="replace")


def read_text(path: Path, max_bytes: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        size = path.stat().st_size
    except OSError:
        return None, "stat_failed"
    if size > max_bytes:
        return None, "too_large"
    try:
        data = path.read_bytes()
    except OSError:
        return None, "read_failed"
    if _is_binary(data):
        return None, "binary"
    text = _decode_text(data)
    return text, None


def get_source_type(path: Path) -> str:
    return SOURCE_TYPE_MAP.get(path.suffix.lower(), "text")


def _is_excluded_name(name: str, exclude_patterns: list[str]) -> bool:
    name_lower = name.lower()
    for pattern in exclude_patterns:
        if not pattern:
            continue
        pattern_lower = pattern.lower()
        if pattern_lower == name_lower:
            return True
        if _pattern_match(name_lower, pattern_lower):
            return True
    return False


def _is_excluded_path(path: Path, exclude_patterns: list[str]) -> bool:
    parts = [part.lower() for part in path.parts]
    for pattern in exclude_patterns:
        if not pattern:
            continue
        pattern_lower = pattern.lower()
        if pattern_lower in parts:
            return True
        if _pattern_match(str(path).lower(), pattern_lower):
            return True
    return False


def _should_include(path: Path, include_patterns: list[str]) -> bool:
    if not include_patterns:
        return True
    name = path.name.lower()
    suffix = path.suffix.lower()
    for pattern in include_patterns:
        if not pattern:
            continue
        pattern_lower = pattern.lower()
        if any(ch in pattern_lower for ch in "*?["):
            if fnmatch.fnmatch(name, pattern_lower):
                return True
        elif pattern_lower.startswith("."):
            if suffix == pattern_lower:
                return True
        elif name == pattern_lower:
            return True
    return False


def _relative_posix(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    return rel.as_posix().lstrip("./")


def _is_included_path(rel_path: str, include_dirs: list[str]) -> bool:
    if not include_dirs:
        return True
    rel_norm = rel_path.replace("\\", "/").lower()
    for pattern in include_dirs:
        if not pattern:
            continue
        pattern_norm = pattern.replace("\\", "/").strip("/").lower()
        has_wildcard = any(ch in pattern_norm for ch in "*?[")
        if "/" in pattern_norm:
            if has_wildcard:
                if Path(rel_norm).match(pattern_norm):
                    return True
            else:
                if rel_norm == pattern_norm or rel_norm.startswith(pattern_norm + "/"):
                    return True
        else:
            if has_wildcard:
                if fnmatch.fnmatch(rel_norm, pattern_norm):
                    return True
            else:
                if rel_norm == pattern_norm or rel_norm.startswith(pattern_norm + "/"):
                    return True
    return False


def _is_assets_non_template(
    rel_path: str, assets_root_dir: str, assets_template_dir: str
) -> bool:
    root_norm = assets_root_dir.replace("\\", "/").strip("/")
    template_norm = assets_template_dir.replace("\\", "/").strip("/")
    if not root_norm:
        return False
    if rel_path == root_norm or rel_path.startswith(root_norm + "/"):
        if rel_path == template_norm or rel_path.startswith(template_norm + "/"):
            return False
        return True
    return False


def _pattern_match(value: str, pattern: str) -> bool:
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatch(value, pattern)
    return False


def _matches_globs(rel_path: str, exclude_globs: list[str]) -> bool:
    if not exclude_globs:
        return False
    rel_norm = rel_path.replace("\\", "/").lower()
    for pattern in exclude_globs:
        if not pattern:
            continue
        pattern_norm = pattern.replace("\\", "/").lower()
        if fnmatch.fnmatch(rel_norm, pattern_norm):
            return True
    return False


def _is_binary(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:2048]
    if b"\x00" in sample:
        return True
    control_bytes = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        11,
        12,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        127,
    }
    control_count = sum(1 for byte in sample if byte in control_bytes)
    ratio = control_count / max(1, len(sample))
    return ratio > 0.3
