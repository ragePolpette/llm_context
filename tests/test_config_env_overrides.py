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




def test_load_config_coerces_yaml_scalars_to_declared_types(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("LLM_CONTEXT_WRITE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_CONTEXT_INGEST_ENABLED", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'embedding_dim: "512"',
                'min_score: "0.3"',
                'ingest_enabled: "false"',
                'write_enabled: "no"',
                "include_dirs: src, docs",
                "scope_map:",
                "  src/billing/: invoice",
                "path_aliases:",
                "  old/: new/",
                "product_targets:",
                "  shop: [shop, storefront]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.embedding_dim == 512
    assert config.min_score == 0.3
    assert config.ingest_enabled is False
    assert config.write_enabled is False
    assert config.include_dirs == ["src", "docs"]
    assert config.scope_map == {"src/billing/": ["invoice"]}
    assert config.path_aliases == {"old/": "new/"}
    assert config.product_targets == {"shop": ["shop", "storefront"]}
    assert isinstance(config.projects_registry_path, Path)


def test_default_config_has_no_domain_specific_values():
    from rag_indexer.config import AppConfig

    config = AppConfig()

    assert config.assets_root_dir == ""
    assert config.assets_template_dir == ""
    assert config.scope_map == {}
    assert config.path_aliases == {}
    assert config.product_targets == {}
    assert config.fts_config == "simple"


def test_unknown_legacy_keys_are_ignored(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("rerank_enabled: true\nchunk_size: 900\n", encoding="utf-8")

    config = load_config(str(config_path))

    assert config.chunk_size == 900
    assert not hasattr(config, "rerank_enabled")


def test_fts_config_is_validated(tmp_path: Path):
    import pytest

    config_path = tmp_path / "config.yaml"
    config_path.write_text("fts_config: italian_unaccent\n", encoding="utf-8")
    assert load_config(str(config_path)).fts_config == "italian_unaccent"

    config_path.write_text("fts_config: \"simple'); DROP TABLE x; --\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid fts_config"):
        load_config(str(config_path))


def test_repository_config_files_have_no_duplicate_keys():
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        keys = [loader.construct_object(key_node, deep=deep) for key_node, _ in node.value]
        duplicates = {key for key in keys if keys.count(key) > 1}
        assert not duplicates, f"duplicate YAML keys: {duplicates}"
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    root = Path(__file__).resolve().parent.parent
    for name in ["config.yaml", "config.example.yaml", "config.ingest.yaml", "config.rework.yaml"]:
        with (root / name).open(encoding="utf-8") as handle:
            yaml.load(handle, Loader=UniqueKeyLoader)


def test_example_config_loads_with_generic_values():
    root = Path(__file__).resolve().parent.parent
    config = load_config(str(root / "config.example.yaml"))

    assert config.include_dirs == ["src", "docs", "web/*.html"]
    assert config.fts_config == "simple"
    assert "src/billing/" in config.scope_map
    assert config.product_aliases == {"shop": "storefront"}
