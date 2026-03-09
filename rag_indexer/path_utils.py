"""
Path utility functions for path prefix normalization and resolution.

Shared across mcp_server.py, mcp_server_http.py, and cli.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def normalize_path_prefix(value: Optional[str]) -> Optional[str]:
    """Normalize a path prefix by converting slashes and applying aliases."""
    if not value:
        return None
    normalized = value.replace("/", "\\").lstrip(".\\")
    normalized = apply_path_aliases(normalized)
    return normalized


def resolve_path_prefix(
    path_prefix: Optional[str],
    file_path: Optional[str],
    query_text: Optional[str],
    auto_scope: bool,
    scope_map: dict[str, list[str]],
) -> Optional[str]:
    """
    Resolve path prefix from explicit prefix, file path, query text, or auto-scope.

    Priority:
    1. Explicit path_prefix
    2. Path detected in file_path
    3. Path detected in query_text
    4. Auto-scoped prefix from query keywords
    """
    if path_prefix:
        return normalize_path_prefix(path_prefix)
    if not file_path:
        detected = path_prefix_from_text(query_text)
        if detected:
            return detected
        return auto_scope_prefix(query_text, scope_map) if auto_scope else None
    return path_prefix_from_file(file_path)


def path_prefix_from_file(file_path: str) -> Optional[str]:
    """Extract path prefix from a file path, converting to relative if absolute."""
    normalized = normalize_path_prefix(file_path)
    if not normalized:
        return None
    path_obj = Path(normalized)
    if path_obj.is_absolute():
        try:
            path_obj = path_obj.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            path_obj = Path(normalized)
    path_str = path_obj.as_posix().replace("/", "\\")
    if path_str.endswith("\\"):
        return path_str
    if Path(path_str).suffix:
        return path_str
    return path_str + "\\"


def path_prefix_from_text(text: Optional[str]) -> Optional[str]:
    """Extract path prefix from text by detecting file paths in query."""
    if not text:
        return None
    pattern = re.compile(
        r"(?P<path>(?:[A-Za-z]:\\)?[\w\-/\\ .]+\.(?:cs|csproj|js|ts|md|txt|json|yaml|yml|toml|html|aspx|ashx|asp|config))",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    best = max(matches, key=lambda match: len(match.group("path")))
    return normalize_path_prefix(best.group("path"))


def apply_path_aliases(path: str) -> str:
    """Apply project-specific path aliases/transformations."""
    normalized = path
    normalized = normalized.replace(
        "pubblico\\assets\\pdf\\", "pubblico\\assets\\template\\pdf\\"
    )
    normalized = normalized.replace(
        "pubblico\\assets\\pdf/", "pubblico\\assets\\template\\pdf/"
    )
    return normalized


def auto_scope_prefix(
    query_text: Optional[str],
    scope_map: dict[str, list[str]],
) -> Optional[str]:
    """
    Auto-detect path prefix from query text using keyword mapping.

    Args:
        query_text: The search query
        scope_map: Dict mapping path prefixes to keyword lists

    Returns:
        Best matching path prefix or None
    """
    if not query_text or not scope_map:
        return None
    text = query_text.lower()
    best_match: Optional[tuple[str, str]] = None
    for prefix, keywords in scope_map.items():
        for keyword in keywords:
            key = keyword.lower().strip()
            if not key:
                continue
            if key in text:
                if best_match is None or len(key) > len(best_match[1]):
                    best_match = (prefix, key)
    if best_match is None:
        return None
    return normalize_path_prefix(best_match[0])


def parse_bool(value: str) -> bool:
    """Parse string value to boolean (handles 'true', '1', 'yes', etc.)."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return bool(value)
