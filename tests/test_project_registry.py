from pathlib import Path

import pytest
import yaml

from rag_indexer.config import load_config
from rag_indexer.project_registry import load_project_registry


def test_load_config_resolves_registry_and_state_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "multi_project_enabled": True,
                "projects_registry_path": "nested/projects.yaml",
                "projects_state_path": "runtime/projects.state.json",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.multi_project_enabled is True
    assert config.projects_registry_path == (tmp_path / "nested" / "projects.yaml").resolve()
    assert config.projects_state_path == (tmp_path / "runtime" / "projects.state.json").resolve()


def test_load_config_supports_extends(tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "multi_project_enabled": False,
                "include_dirs": ["src"],
                "projects_registry_path": "projects.base.yaml",
            }
        ),
        encoding="utf-8",
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "multi_project_enabled": True,
                "projects_state_path": "runtime/projects.state.json",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(child_path))

    assert config.multi_project_enabled is True
    assert config.include_dirs == ["src"]
    assert config.projects_registry_path == (tmp_path / "projects.base.yaml").resolve()
    assert config.projects_state_path == (tmp_path / "runtime" / "projects.state.json").resolve()


def test_load_config_rejects_extends_cycle(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump({"extends": "second.yaml"}), encoding="utf-8")
    second.write_text(yaml.safe_dump({"extends": "first.yaml"}), encoding="utf-8")

    with pytest.raises(ValueError, match="extends cycle"):
        load_config(str(first))


def test_project_registry_loads_projects_and_runtime_state(tmp_path):
    registry_path = tmp_path / "projects.yaml"
    state_path = tmp_path / "projects.state.json"

    registry_path.write_text(
        yaml.safe_dump(
            {
                "projects": [
                    {
                        "project_id": "alpha",
                        "display_name": "Alpha",
                        "root_path": "repos/alpha",
                        "ingest_enabled": True,
                        "write_enabled": False,
                        "include_profile": "default",
                        "retrieval_profile": "default",
                        "include_dirs": ["src", "docs"],
                        "exclude_globs": ["*.env"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        """
{
  "projects": {
    "alpha": {
      "last_ingest_status": "success",
      "last_successful_ingest_at": "2026-03-16T20:00:00Z",
      "last_ingest_duration_sec": 12.5,
      "index_version": "v1"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    registry = load_project_registry(registry_path, state_path)
    project = registry.require_project("alpha")

    assert registry.project_count() == 1
    assert project.root_path == (tmp_path / "repos" / "alpha").resolve()
    assert project.include_dirs == ["src", "docs"]
    assert project.last_ingest_status == "success"
    assert project.last_successful_ingest_at == "2026-03-16T20:00:00Z"
    assert project.index_version == "v1"


def test_project_registry_rejects_duplicate_project_ids(tmp_path):
    registry_path = tmp_path / "projects.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "projects": [
                    {"project_id": "alpha", "root_path": "repos/alpha"},
                    {"project_id": "alpha", "root_path": "repos/alpha-copy"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate project_id"):
        load_project_registry(registry_path)


def test_project_registry_rejects_missing_root_path(tmp_path):
    registry_path = tmp_path / "projects.yaml"
    registry_path.write_text(
        yaml.safe_dump({"projects": [{"project_id": "alpha"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires root_path"):
        load_project_registry(registry_path)


def test_project_registry_persists_runtime_state(tmp_path):
    registry_path = tmp_path / "projects.yaml"
    state_path = tmp_path / "projects.state.json"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "projects": [
                    {
                        "project_id": "alpha",
                        "display_name": "Alpha",
                        "root_path": "repos/alpha",
                        "ingest_enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = load_project_registry(registry_path, state_path)
    updated = registry.save_runtime_state(
        "alpha",
        last_ingest_status="success",
        last_successful_ingest_at="2026-03-16T21:00:00Z",
        index_fingerprint="abc123",
    )

    assert updated.last_ingest_status == "success"
    reloaded = load_project_registry(registry_path, state_path).require_project("alpha")
    assert reloaded.last_successful_ingest_at == "2026-03-16T21:00:00Z"
    assert reloaded.index_fingerprint == "abc123"
