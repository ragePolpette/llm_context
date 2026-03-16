from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import json

import yaml


@dataclass(frozen=True)
class ProjectRuntimeState:
    last_ingest_status: Optional[str] = None
    last_ingest_started_at: Optional[str] = None
    last_ingest_finished_at: Optional[str] = None
    last_successful_ingest_at: Optional[str] = None
    last_ingest_duration_sec: Optional[float] = None
    index_version: Optional[str] = None
    index_fingerprint: Optional[str] = None


@dataclass(frozen=True)
class ProjectRecord:
    project_id: str
    display_name: str
    root_path: Path
    include_profile: Optional[str] = None
    retrieval_profile: Optional[str] = None
    include_dirs: list[str] = field(default_factory=list)
    exclude_globs: list[str] = field(default_factory=list)
    ingest_enabled: bool = False
    write_enabled: bool = False
    runtime_state: ProjectRuntimeState = field(default_factory=ProjectRuntimeState)

    @property
    def last_ingest_status(self) -> Optional[str]:
        return self.runtime_state.last_ingest_status

    @property
    def last_successful_ingest_at(self) -> Optional[str]:
        return self.runtime_state.last_successful_ingest_at

    @property
    def index_version(self) -> Optional[str]:
        return self.runtime_state.index_version

    @property
    def index_fingerprint(self) -> Optional[str]:
        return self.runtime_state.index_fingerprint

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "display_name": self.display_name,
            "root_path": str(self.root_path),
            "include_profile": self.include_profile,
            "retrieval_profile": self.retrieval_profile,
            "include_dirs": list(self.include_dirs),
            "exclude_globs": list(self.exclude_globs),
            "ingest_enabled": self.ingest_enabled,
            "write_enabled": self.write_enabled,
            "last_ingest_status": self.runtime_state.last_ingest_status,
            "last_ingest_started_at": self.runtime_state.last_ingest_started_at,
            "last_ingest_finished_at": self.runtime_state.last_ingest_finished_at,
            "last_successful_ingest_at": self.runtime_state.last_successful_ingest_at,
            "last_ingest_duration_sec": self.runtime_state.last_ingest_duration_sec,
            "index_version": self.runtime_state.index_version,
            "index_fingerprint": self.runtime_state.index_fingerprint,
        }


class ProjectRegistry:
    def __init__(
        self,
        *,
        registry_path: Path,
        state_path: Path,
        projects: list[ProjectRecord],
    ) -> None:
        self.registry_path = registry_path
        self.state_path = state_path
        self._projects_by_id = {project.project_id: project for project in projects}

    def list_projects(self) -> list[ProjectRecord]:
        return sorted(self._projects_by_id.values(), key=lambda item: item.project_id)

    def get_project(self, project_id: str) -> Optional[ProjectRecord]:
        return self._projects_by_id.get(project_id)

    def require_project(self, project_id: str) -> ProjectRecord:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Unknown project_id: {project_id}")
        return project

    def project_count(self) -> int:
        return len(self._projects_by_id)

    def list_enabled_for_ingest(self) -> list[ProjectRecord]:
        return [project for project in self.list_projects() if project.ingest_enabled]

    def save_runtime_state(self, project_id: str, **updates: Any) -> ProjectRecord:
        project = self.require_project(project_id)
        current_state = _load_state_map(self.state_path).get(project_id, {})
        current_state.update({key: value for key, value in updates.items() if value is not None})
        state_map = _load_state_map(self.state_path)
        state_map[project_id] = current_state
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump({"projects": state_map}, handle, indent=2, ensure_ascii=False)
        updated = load_project_registry(self.registry_path, self.state_path).require_project(project_id)
        self._projects_by_id[project_id] = updated
        return updated


def load_project_registry(registry_path: Path, state_path: Optional[Path] = None) -> ProjectRegistry:
    registry_path = registry_path.resolve()
    if not registry_path.exists():
        return ProjectRegistry(registry_path=registry_path, state_path=_resolve_state_path(registry_path, state_path), projects=[])

    with registry_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    projects_payload = payload.get("projects") or []
    if not isinstance(projects_payload, list):
        raise ValueError("projects registry must contain a top-level 'projects' list")

    resolved_state_path = _resolve_state_path(registry_path, state_path)
    state_map = _load_state_map(resolved_state_path)
    projects: list[ProjectRecord] = []
    seen_project_ids: set[str] = set()

    for item in projects_payload:
        project = _load_project_record(item, registry_path.parent, state_map)
        if project.project_id in seen_project_ids:
            raise ValueError(f"Duplicate project_id in registry: {project.project_id}")
        seen_project_ids.add(project.project_id)
        projects.append(project)

    return ProjectRegistry(
        registry_path=registry_path,
        state_path=resolved_state_path,
        projects=projects,
    )


def _load_project_record(
    item: Any,
    base_dir: Path,
    state_map: dict[str, dict[str, Any]],
) -> ProjectRecord:
    if not isinstance(item, dict):
        raise ValueError("Each project entry must be an object")

    project_id = str(item.get("project_id", "")).strip()
    if not project_id:
        raise ValueError("Each project entry requires a non-empty project_id")

    display_name = str(item.get("display_name") or project_id).strip()
    root_path_raw = item.get("root_path")
    if root_path_raw is None or str(root_path_raw).strip() == "":
        raise ValueError(f"Project '{project_id}' requires root_path")

    root_path = Path(str(root_path_raw)).expanduser()
    if not root_path.is_absolute():
        root_path = (base_dir / root_path).resolve()

    state_payload = state_map.get(project_id, {})
    runtime_state = ProjectRuntimeState(
        last_ingest_status=_optional_str(state_payload.get("last_ingest_status")),
        last_ingest_started_at=_optional_str(state_payload.get("last_ingest_started_at")),
        last_ingest_finished_at=_optional_str(state_payload.get("last_ingest_finished_at")),
        last_successful_ingest_at=_optional_str(state_payload.get("last_successful_ingest_at")),
        last_ingest_duration_sec=_optional_float(state_payload.get("last_ingest_duration_sec")),
        index_version=_optional_str(state_payload.get("index_version")),
        index_fingerprint=_optional_str(state_payload.get("index_fingerprint")),
    )

    return ProjectRecord(
        project_id=project_id,
        display_name=display_name,
        root_path=root_path,
        include_profile=_optional_str(item.get("include_profile")),
        retrieval_profile=_optional_str(item.get("retrieval_profile")),
        include_dirs=_coerce_str_list(item.get("include_dirs")),
        exclude_globs=_coerce_str_list(item.get("exclude_globs")),
        ingest_enabled=bool(item.get("ingest_enabled", False)),
        write_enabled=bool(item.get("write_enabled", False)),
        runtime_state=runtime_state,
    )


def _resolve_state_path(registry_path: Path, state_path: Optional[Path]) -> Path:
    if state_path is None:
        return registry_path.with_suffix(".state.json")
    if not state_path.is_absolute():
        return (registry_path.parent / state_path).resolve()
    return state_path.resolve()


def _load_state_map(state_path: Path) -> dict[str, dict[str, Any]]:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    projects_payload = payload.get("projects") or {}
    if not isinstance(projects_payload, dict):
        raise ValueError("projects state must contain an object field named 'projects'")
    return {
        str(project_id): value
        for project_id, value in projects_payload.items()
        if isinstance(value, dict)
    }


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
