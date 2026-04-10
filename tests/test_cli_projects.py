from types import SimpleNamespace

import pytest
import yaml

import cli


def _write_cli_config(
    tmp_path,
    *,
    ingest_enabled=False,
    write_enabled=False,
    projects=None,
):
    if projects is None:
        projects = [
            {
                "project_id": "alpha",
                "display_name": "Alpha",
                "root_path": "repos/alpha",
                "ingest_enabled": True,
            },
            {
                "project_id": "beta",
                "display_name": "Beta",
                "root_path": "repos/beta",
                "ingest_enabled": False,
            },
        ]
    registry_path = tmp_path / "projects.yaml"
    registry_path.write_text(
        yaml.safe_dump({"projects": projects}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ingest_enabled": ingest_enabled,
                "write_enabled": write_enabled,
                "projects_registry_path": registry_path.name,
                "projects_state_path": "projects.state.json",
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_list_projects_command_prints_registry_json(monkeypatch, tmp_path, capsys):
    config_path = _write_cli_config(tmp_path)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["cli.py", "--config", str(config_path), "list-projects", "--json"],
    )

    cli.main()
    output = capsys.readouterr().out

    assert '"count": 2' in output
    assert '"project_id": "alpha"' in output


def test_ingest_command_rejects_unknown_project(monkeypatch, tmp_path):
    config_path = _write_cli_config(tmp_path, ingest_enabled=True)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "cli.py",
            "--config",
            str(config_path),
            "ingest",
            "--dsn",
            "postgresql://ctx",
            "--project-id",
            "missing",
        ],
    )

    with pytest.raises(ValueError, match="Unknown project_id"):
        cli.main()


def test_run_ingest_command_allows_registered_repo_root_project(monkeypatch, tmp_path):
    config_path = _write_cli_config(
        tmp_path,
        ingest_enabled=True,
        projects=[
            {
                "project_id": "llm_context",
                "display_name": "LLM Context",
                "root_path": ".",
                "include_dirs": ["rag_indexer"],
                "ingest_enabled": True,
            }
        ],
    )
    config = cli.load_config(str(config_path))
    registry = cli.load_project_registry(
        config.projects_registry_path,
        config.projects_state_path,
    )
    captured = {}

    class FakeConn:
        def close(self):
            return None

    class FakeStore:
        def __init__(self, conn, embedding_dim):
            self.embedding_dim = embedding_dim

        def get_repo_stats(self, repo_id):
            assert repo_id == "llm_context"
            return {
                "indexed_documents": 1,
                "indexed_chunks": 2,
                "indexed_symbols": 0,
            }

    def fake_ingest_project_v2(**kwargs):
        captured["root"] = str(kwargs["root"])
        return SimpleNamespace(
            deleted_rows=0,
            files_scanned=3,
            files_indexed=1,
            files_skipped=2,
            chunks_created=2,
            chunks_inserted=2,
            duration_sec=0.5,
        )

    monkeypatch.setattr(cli, "_build_embedder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "get_connection", lambda dsn: FakeConn())
    monkeypatch.setattr(cli, "RagStore", FakeStore)
    monkeypatch.setattr(cli, "ingest_project_v2", fake_ingest_project_v2)

    args = SimpleNamespace(
        dsn="postgresql://ctx",
        embedder="dummy",
        embedding_dim=None,
        gemini_api_key=None,
        gemini_model="text-embedding-004",
        local_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        root=None,
        include=None,
        exclude=None,
        include_dirs=None,
        assets_template_only=None,
        assets_root_dir=None,
        assets_template_dir=None,
        max_bytes=None,
        chunk_size=None,
        chunk_overlap=None,
        md_chunk_size=None,
        code_chunk_size=None,
        min_chunk_chars=None,
        incremental=False,
    )

    stats = cli._run_ingest_command(args, config, registry, "llm_context")

    assert stats.files_indexed == 1
    assert captured["root"] == str(tmp_path.resolve())

def test_run_ingest_command_updates_project_runtime_state(monkeypatch, tmp_path):
    config_path = _write_cli_config(tmp_path, ingest_enabled=True)
    config = cli.load_config(str(config_path))
    registry = cli.load_project_registry(
        config.projects_registry_path,
        config.projects_state_path,
    )

    class FakeConn:
        def close(self):
            return None

    class FakeStore:
        def __init__(self, conn, embedding_dim):
            self.embedding_dim = embedding_dim

        def get_repo_stats(self, repo_id):
            assert repo_id == "alpha"
            return {
                "indexed_documents": 4,
                "indexed_chunks": 8,
                "indexed_symbols": 2,
            }

    monkeypatch.setattr(cli, "_build_embedder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "get_connection", lambda dsn: FakeConn())
    monkeypatch.setattr(cli, "RagStore", FakeStore)
    monkeypatch.setattr(
        cli,
        "ingest_project_v2",
        lambda **kwargs: SimpleNamespace(
            deleted_rows=1,
            files_scanned=10,
            files_indexed=4,
            files_skipped=6,
            chunks_created=8,
            chunks_inserted=8,
            duration_sec=2.75,
        ),
    )

    args = SimpleNamespace(
        dsn="postgresql://ctx",
        embedder="dummy",
        embedding_dim=None,
        gemini_api_key=None,
        gemini_model="text-embedding-004",
        local_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        root=None,
        include=None,
        exclude=None,
        include_dirs=None,
        assets_template_only=None,
        assets_root_dir=None,
        assets_template_dir=None,
        max_bytes=None,
        chunk_size=None,
        chunk_overlap=None,
        md_chunk_size=None,
        code_chunk_size=None,
        min_chunk_chars=None,
        incremental=False,
    )

    stats = cli._run_ingest_command(args, config, registry, "alpha")
    reloaded = cli.load_project_registry(
        config.projects_registry_path,
        config.projects_state_path,
    ).require_project("alpha")

    assert stats.files_indexed == 4
    assert reloaded.last_ingest_status == "success"
    assert reloaded.last_successful_ingest_at is not None
    assert reloaded.index_version == "v2"
    assert reloaded.index_manifest is not None
    assert reloaded.index_manifest.last_ingest_status == "success"
    assert reloaded.index_manifest.indexed_documents == 4
    assert reloaded.index_manifest.indexed_chunks == 8
    assert reloaded.index_manifest.indexed_symbols == 2

