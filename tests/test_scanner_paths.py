from __future__ import annotations

from pathlib import Path

from rag_indexer.ingest import _normalize_path, _relative_path
from rag_indexer.path_utils import (
    apply_path_aliases,
    normalize_path_prefix,
    path_prefix_from_file,
    resolve_path_prefix,
    to_posix,
)
from rag_indexer.scanner import _relative_posix, scan_files


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("content\n", encoding="utf-8")


def _scan(root: Path, **kwargs) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in scan_files(root, include_patterns=[], exclude_patterns=[], **kwargs)
    }


def test_exclude_globs_match_dot_directories_and_dot_files(tmp_path):
    for relative in [
        ".local/x.py",
        ".local/nested/y.md",
        ".env.json",
        "src/.env.local",
        "src/app.py",
        "local/keep.py",
    ]:
        _touch(tmp_path, relative)

    found = _scan(tmp_path, exclude_globs=[".local/**", "*.env*"])

    assert found == {"src/app.py", "local/keep.py"}


def test_include_dirs_match_dot_directories(tmp_path):
    _touch(tmp_path, ".github/workflows/ci.yml")
    _touch(tmp_path, "github/other.yml")

    assert _scan(tmp_path, include_dirs=[".github"]) == {".github/workflows/ci.yml"}


def test_relative_posix_keeps_leading_dots(tmp_path):
    assert _relative_posix(tmp_path / ".local" / "x", tmp_path) == ".local/x"
    assert _relative_posix(tmp_path / ".env.json", tmp_path) == ".env.json"


def test_to_posix_only_strips_leading_dot_slash_segments():
    assert to_posix("./.local/x") == ".local/x"
    assert to_posix("././src\\api\\") == "src/api/"
    assert to_posix(".env.json") == ".env.json"
    assert to_posix("../shared/x") == "../shared/x"


def test_normalize_path_prefix_returns_posix():
    assert normalize_path_prefix("src\\api\\Controllers\\") == "src/api/Controllers/"
    assert normalize_path_prefix(".\\.config\\app") == ".config/app"
    assert normalize_path_prefix("") is None


def test_path_aliases_come_from_configuration_only():
    assert apply_path_aliases("web/assets/pdf/a.html") == "web/assets/pdf/a.html"
    aliases = {"web\\assets\\pdf\\": "web/assets/template/pdf/"}
    assert apply_path_aliases("web/assets/pdf/a.html", aliases) == "web/assets/template/pdf/a.html"
    assert (
        resolve_path_prefix(
            path_prefix="web\\assets\\pdf\\a.html",
            file_path=None,
            query_text=None,
            auto_scope=False,
            scope_map={},
            path_aliases=aliases,
        )
        == "web/assets/template/pdf/a.html"
    )


def test_path_prefix_from_file_uses_posix_separators():
    assert path_prefix_from_file("src\\api\\Billing.cs") == "src/api/Billing.cs"
    assert path_prefix_from_file("src/api") == "src/api/"


def test_auto_scope_normalizes_legacy_backslash_scope_keys():
    scope_map = {"src\\billing\\": ["invoice"]}

    assert (
        resolve_path_prefix(
            path_prefix=None,
            file_path=None,
            query_text="invoice totals",
            auto_scope=True,
            scope_map=scope_map,
        )
        == "src/billing/"
    )


def test_ingest_paths_are_posix(tmp_path):
    path = tmp_path / "src" / "Api" / "Billing.cs"

    assert _relative_path(path, tmp_path) == "src/Api/Billing.cs"
    assert _normalize_path("src\\Api\\Billing.cs") == "src/api/billing.cs"
