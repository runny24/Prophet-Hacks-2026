"""Dataset registry retrieval for the forecasting track."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from .schemas import Event

DEFAULT_DATASET = "sample-sports"
DEFAULT_REPO_URL = "https://github.com/ai-prophet/ai-prophet-datasets"
DATASET_ENV = "PA_FORECAST_DATASET"
RELEASE_ENV = "PA_FORECAST_RELEASE"
BRANCH_ENV = "PA_FORECAST_DATASET_BRANCH"
REPO_PATH_ENV = "PA_FORECAST_DATASETS_REPO_PATH"
REPO_URL_ENV = "PA_FORECAST_DATASETS_REPO_URL"

logger = logging.getLogger(__name__)


def retrieve_dataset_events(
    *,
    dataset: str | None = None,
    release_id: str | None = None,
    repo_path: str | None = None,
    repo_url: str | None = None,
    branch: str | None = None,
    include_resolved: bool = False,
) -> tuple[list[Event], str, str]:
    """Fetch forecast events from ``ai-prophet-datasets``.

    Defaults are organizer-friendly: dataset/release/branch/repo settings
    can come from environment variables, and an omitted release selects the
    newest open release, falling back to the newest release if none are open.

    Returns:
        ``(events, dataset_name, release_id)``.
    """
    task_rows, dataset_name, release_id = retrieve_dataset_tasks(
        dataset=dataset,
        release_id=release_id,
        repo_path=repo_path,
        repo_url=repo_url,
        branch=branch,
    )
    if not include_resolved:
        task_rows = [row for row in task_rows if row.get("resolved_outcome") is None]

    events = [_event_from_task(row) for row in task_rows]
    logger.info(
        "Retrieved %d event(s) from %s/%s",
        len(events),
        dataset_name,
        release_id,
    )
    return events, dataset_name, release_id


def retrieve_dataset_tasks(
    *,
    dataset: str | None = None,
    release_id: str | None = None,
    repo_path: str | None = None,
    repo_url: str | None = None,
    branch: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Fetch raw task rows from the forecasting dataset registry."""
    dataset_name = dataset or os.environ.get(DATASET_ENV) or DEFAULT_DATASET
    selected_release = release_id or os.environ.get(RELEASE_ENV)
    selected_branch = branch or os.environ.get(BRANCH_ENV) or "main"
    selected_repo_path = repo_path or os.environ.get(REPO_PATH_ENV)
    selected_repo_url = repo_url or os.environ.get(REPO_URL_ENV) or DEFAULT_REPO_URL
    local_root = Path(selected_repo_path).resolve() if selected_repo_path else None

    registry = _load_registry(
        repo_path=local_root,
        repo_url=selected_repo_url,
        branch=selected_branch,
    )
    dataset_obj = _find_dataset(registry, dataset_name)
    release = _find_release(dataset_obj, selected_release)
    task_rows = _load_tasks(
        release,
        repo_path=local_root,
        repo_url=selected_repo_url,
        branch=selected_branch,
    )
    return task_rows, dataset_name, release["id"]


def _load_registry(
    *,
    repo_path: Path | None,
    repo_url: str,
    branch: str,
) -> dict[str, Any]:
    """Load registry metadata without requiring the dataset SDK package."""
    if repo_path is not None:
        registry_path = repo_path / "registry.json"
        if registry_path.is_file():
            return json.loads(registry_path.read_text())
        return _build_local_registry(repo_path / "datasets")

    return json.loads(_http_get(repo_url, branch, "registry.json"))


def _build_local_registry(datasets_dir: Path) -> dict[str, Any]:
    """Build a minimal registry from a local datasets tree."""
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"datasets directory not found: {datasets_dir}")

    datasets: list[dict[str, Any]] = []
    for dataset_dir in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
        dataset_json = dataset_dir / "dataset.json"
        if not dataset_json.is_file():
            continue
        metadata = json.loads(dataset_json.read_text())
        releases: list[dict[str, Any]] = []
        releases_dir = dataset_dir / "releases"
        if releases_dir.is_dir():
            for release_dir in sorted(p for p in releases_dir.iterdir() if p.is_dir()):
                release_json = release_dir / "release.json"
                tasks_path = release_dir / "tasks.jsonl"
                if not release_json.is_file() or not tasks_path.is_file():
                    continue
                release = json.loads(release_json.read_text())
                task_rows = _read_jsonl(tasks_path)
                releases.append(
                    {
                        "id": release["release_id"],
                        "release_date": release["release_date"],
                        "status": release["status"],
                        "task_count": len(task_rows),
                        "resolved_count": sum(
                            1 for row in task_rows if row.get("resolved_outcome") is not None
                        ),
                        "path": str(tasks_path.relative_to(datasets_dir.parent)),
                    }
                )
        releases.sort(key=lambda item: item.get("release_date", ""), reverse=True)
        datasets.append(
            {
                "name": metadata["name"],
                "description": metadata.get("description", ""),
                "releases": releases,
            }
        )

    return {"datasets": datasets}


def _find_dataset(registry: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    """Return a dataset entry from a registry payload."""
    datasets = registry.get("datasets", [])
    for dataset in datasets:
        if dataset.get("name") == dataset_name:
            return dataset
    available = ", ".join(str(d.get("name")) for d in datasets) or "(none)"
    raise KeyError(f"dataset not found: {dataset_name}. Available datasets: {available}")


def _find_release(
    dataset_obj: dict[str, Any],
    release_id: str | None,
) -> dict[str, Any]:
    """Return the selected release, or the newest open release by default."""
    releases = dataset_obj.get("releases", [])
    if release_id:
        for release in releases:
            if release.get("id") == release_id:
                return release
        raise KeyError(f"release not found: {dataset_obj.get('name')}/{release_id}")

    for release in releases:
        if release.get("status") == "open":
            return release
    if releases:
        return releases[0]
    raise KeyError(f"dataset has no releases: {dataset_obj.get('name')}")


def _load_tasks(
    release: dict[str, Any],
    *,
    repo_path: Path | None,
    repo_url: str,
    branch: str,
) -> list[dict[str, Any]]:
    """Load task rows for a release."""
    rel_path = release.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError(f"release {release.get('id')} is missing a tasks path")
    if repo_path is not None:
        return _read_jsonl(repo_path / rel_path)
    return _parse_jsonl(_http_get(repo_url, branch, rel_path), rel_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""
    return _parse_jsonl(path.read_text(), str(path))


def _parse_jsonl(text: str, source: str) -> list[dict[str, Any]]:
    """Parse a JSONL task file."""
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_no} must be a JSON object")
        rows.append(row)
    return rows


def _http_get(repo_url: str, branch: str, repo_relative_path: str) -> str:
    """Fetch a file from a public GitHub repository via the Contents API."""
    owner, repo = _owner_repo(repo_url)
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_relative_path}"
    response = requests.get(
        url,
        params={"ref": branch},
        headers={"Accept": "application/vnd.github.raw"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def _owner_repo(repo_url: str) -> tuple[str, str]:
    """Parse ``owner/repo`` from a GitHub URL."""
    parts = repo_url.rstrip("/").removesuffix(".git").split("/")
    if len(parts) < 2:
        raise ValueError(f"cannot parse owner/repo from repo_url: {repo_url}")
    return parts[-2], parts[-1]


def _event_from_task(row: dict[str, Any]) -> Event:
    """Map a dataset task row to the current forecast event contract."""
    task_id = str(row.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("dataset task is missing task_id")

    close_time = _task_close_time(row)
    if close_time is None:
        raise ValueError(
            f"dataset task {task_id!r} is missing a forecast deadline. "
            "Add a 'predict_by', 'close_time', or 'deadline' ISO timestamp."
        )

    context = _as_str(row.get("context"))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    raw_extra = metadata.get("raw_extra") if isinstance(metadata.get("raw_extra"), dict) else {}
    source_meta = raw_extra.get("source") if isinstance(raw_extra.get("source"), dict) else {}

    category = (
        _as_str(row.get("category"))
        or _as_str(metadata.get("category"))
        or _as_str(source_meta.get("category"))
        or "General"
    )
    rules = (
        _as_str(row.get("rules"))
        or _as_str(row.get("resolution_criteria"))
        or _as_str(source_meta.get("rules"))
        or context
    )
    description = _as_str(row.get("description")) or context

    return Event(
        event_ticker=_as_str(row.get("source")) or task_id,
        market_ticker=task_id,
        title=str(row.get("title") or task_id),
        subtitle=_as_str(row.get("subtitle")),
        description=description,
        category=category,
        rules=rules,
        close_time=close_time,
        outcomes=list(row.get("outcomes") or []),
        resolved_outcome=row.get("resolved_outcome"),
    )


def _task_close_time(row: dict[str, Any]) -> datetime | None:
    """Extract the forecast deadline from common task fields."""
    raw = row.get("predict_by") or row.get("close_time") or row.get("deadline")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        value = raw
    elif isinstance(raw, str):
        try:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value


def _as_str(value: Any) -> str | None:
    """Return a non-empty string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
