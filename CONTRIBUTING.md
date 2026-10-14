# Contributing

## Workflow

1. Open an issue for non-trivial changes before starting work.
2. Create a branch from `main` (for example `fix/<topic>` or `feat/<topic>`).
3. Keep changes focused. Update `README.md` when behavior or configuration changes.
4. Open a pull request against `main` and fill in the pull request template. CI must pass.

## Development setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional symbol extraction support:

```bash
pip install -e ".[symbols]"
```

## Tests

```bash
python -m pytest -q
```

The test suite does not require a running database; tests that need one are skipped.

## Lint

```bash
ruff check .
```

Ruff is configured in `pyproject.toml` and runs in CI.

## Security rules

- Never commit secrets, DSNs with credentials, API keys or real `.env` files. Use `.env.example` and `config.example.yaml` for documented defaults.
- Do not commit local data, model caches, logs or database files; they are covered by `.gitignore`.
- Do not widen the default network exposure (host, allowed hosts, allowed origins) or enable ingest and write operations by default.
- Keep retrieval scoped to an explicit project.
- Report vulnerabilities as described in `SECURITY.md`, not in public issues.
