from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


KNOWN_PRODUCT_ALIASES = {
    "legacy": "legacy",
    "bpopilot": "legacy",
    "fatturhello": "fatturhello",
    "fiscobot": "fiscobot",
}


def map_work_item_to_codebase(
    *,
    summary: str,
    description: str,
    product_target_hint: Optional[str],
    project_record: Any,
    functional_context: dict[str, Any],
    max_paths: int = 6,
) -> dict[str, Any]:
    normalized_hint = _normalize_product_target(product_target_hint)
    repo_target = _infer_repo_target(project_record)
    paths = _extract_paths(functional_context, max_paths=max_paths)
    product_target = _infer_product_target(
        product_target_hint=normalized_hint,
        repo_target=repo_target,
        paths=paths,
    )
    area = _infer_area(paths)
    confidence = _compute_confidence(
        functional_context=functional_context,
        product_target=product_target,
        product_target_hint=normalized_hint,
        paths=paths,
    )
    blockers = _collect_blockers(
        summary=summary,
        description=description,
        paths=paths,
        product_target=product_target,
        product_target_hint=normalized_hint,
    )
    in_scope = bool(paths) and repo_target != "unknown"
    feasibility = _infer_feasibility(
        confidence=confidence,
        blockers=blockers,
        in_scope=in_scope,
        paths=paths,
    )
    implementation_hint = _build_implementation_hint(
        functional_context=functional_context,
        paths=paths,
        feasibility=feasibility,
    )

    return {
        "product_target": product_target,
        "repo_target": repo_target,
        "area": area,
        "in_scope": in_scope,
        "confidence": confidence,
        "feasibility": feasibility,
        "implementation_hint": implementation_hint,
        "paths": paths,
        "blockers": blockers,
    }


def _normalize_product_target(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    return KNOWN_PRODUCT_ALIASES.get(normalized, normalized or None)


def _infer_repo_target(project_record: Any) -> str:
    root_path = getattr(project_record, "root_path", None)
    if root_path:
        try:
            return Path(root_path).name or str(getattr(project_record, "project_id", "unknown"))
        except Exception:
            pass
    project_id = str(getattr(project_record, "project_id", "") or "").strip()
    return project_id or "unknown"


def _extract_paths(functional_context: dict[str, Any], *, max_paths: int) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for collection_name in ("core_files", "supporting_matches", "entry_points"):
        for item in functional_context.get(collection_name) or []:
            path = str(item.get("source_path") or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            ordered.append(path)
            if len(ordered) >= max_paths:
                return ordered
    return ordered


def _infer_product_target(
    *,
    product_target_hint: Optional[str],
    repo_target: str,
    paths: list[str],
) -> str:
    if product_target_hint in {"legacy", "fatturhello", "fiscobot"}:
        return product_target_hint

    haystacks = [repo_target, *paths]
    for value in haystacks:
        text = value.lower()
        if "fatturhello" in text:
            return "fatturhello"
        if "fiscobot" in text:
            return "fiscobot"
        if any(token in text for token in ("bpopilot", "legacy", "pubblico", "librerie", "controllers")):
            return "legacy"
    return "unknown"


def _infer_area(paths: list[str]) -> str:
    if not paths:
        return "unknown"
    segments = [path.replace("/", "\\").split("\\")[:-1] for path in paths if path]
    if not segments:
        return "unknown"

    first = segments[0]
    common: list[str] = []
    for index, segment in enumerate(first):
        if all(len(parts) > index and parts[index].lower() == segment.lower() for parts in segments[1:]):
            common.append(segment)
        else:
            break

    if common:
        return "/".join(common)

    top = first[: min(2, len(first))]
    return "/".join(top) if top else "unknown"


def _compute_confidence(
    *,
    functional_context: dict[str, Any],
    product_target: str,
    product_target_hint: Optional[str],
    paths: list[str],
) -> float:
    summary = functional_context.get("summary") or {}
    core_files = functional_context.get("core_files") or []
    if not paths:
        return 0.18

    best_score = 0.0
    total_score = 0.0
    for item in core_files:
        score = float(item.get("aggregate_score") or 0.0)
        total_score += score
        best_score = max(best_score, score)

    focus_ratio = 0.0
    if total_score > 0:
        focus_ratio = best_score / total_score

    confidence = min(best_score / 2.2, 0.52)
    confidence += min(focus_ratio * 0.22, 0.22)
    confidence += min(int(summary.get("symbol_hit_count") or 0) * 0.07, 0.14)
    confidence += min(int(summary.get("core_file_count") or 0) * 0.03, 0.09)
    if product_target_hint and product_target_hint == product_target:
        confidence += 0.08
    if product_target != "unknown":
        confidence += 0.05
    return round(max(0.05, min(confidence, 0.98)), 2)


def _collect_blockers(
    *,
    summary: str,
    description: str,
    paths: list[str],
    product_target: str,
    product_target_hint: Optional[str],
) -> list[str]:
    blockers: list[str] = []
    if not summary.strip() and not description.strip():
        blockers.append("Work item privo di summary e description utili.")
    if not paths:
        blockers.append("Nessun path rilevante trovato nella codebase.")
    if product_target_hint and product_target == "unknown":
        blockers.append("Il product_target_hint non ha trovato riscontro nella codebase.")
    return blockers


def _infer_feasibility(
    *,
    confidence: float,
    blockers: list[str],
    in_scope: bool,
    paths: list[str],
) -> str:
    if not in_scope:
        return "out_of_scope"
    if blockers and not paths:
        return "blocked"
    if confidence >= 0.74:
        return "high"
    if confidence >= 0.52:
        return "medium"
    return "low"


def _build_implementation_hint(
    *,
    functional_context: dict[str, Any],
    paths: list[str],
    feasibility: str,
) -> str:
    entry_points = functional_context.get("entry_points") or []
    if feasibility == "blocked":
        return "Prima chiarire il perimetro del work item o migliorare il mapping verso la codebase."
    if entry_points:
        primary = entry_points[0]
        hint = (
            f"Parti da {primary.get('name') or 'symbol'} in "
            f"{primary.get('source_path') or '(path sconosciuto)'}"
        )
        if len(paths) > 1:
            hint += f"; poi verifica anche {paths[1]}"
        return hint
    if paths:
        hint = f"Parti dal file {paths[0]}"
        if len(paths) > 1:
            hint += f" e segui il flusso verso {paths[1]}"
        return hint
    return "Mapping ancora troppo debole per proporre un hint implementativo affidabile."
