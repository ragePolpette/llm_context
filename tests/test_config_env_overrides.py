from __future__ import annotations

from pathlib import Path

from rag_indexer.config import load_config


def test_load_config_applies_write_override_to_write_and_ingest(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("write_enabled: false\ningest_enabled: false\n", encoding="utf-8")

    monkeypatch.setenv("LLM_CONTEXT_WRITE_ENABLED", "true")
    monkeypatch.delenv("LLM_CONTEXT_INGEST_ENABLED", raising=False)

    config = load_config(str(config_path))

    assert config.write_enabled is True
    assert config.ingest_enabled is True


def test_load_config_allows_explicit_ingest_override(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("write_enabled: false\ningest_enabled: false\n", encoding="utf-8")

    monkeypatch.setenv("LLM_CONTEXT_WRITE_ENABLED", "true")
    monkeypatch.setenv("LLM_CONTEXT_INGEST_ENABLED", "false")

    config = load_config(str(config_path))

    assert config.write_enabled is True
    assert config.ingest_enabled is False


