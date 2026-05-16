from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai_prophet_core.forecast.dataset_retrieve import retrieve_dataset_events


def _write_dataset(repo: Path, *, include_deadline: bool = True) -> None:
    release_dir = repo / "datasets" / "sample-sports" / "releases" / "v1.0.0"
    release_dir.mkdir(parents=True)
    (repo / "datasets" / "sample-sports" / "dataset.json").write_text(
        json.dumps({"name": "sample-sports", "description": "Sports sample"}) + "\n"
    )
    (release_dir / "release.json").write_text(
        json.dumps(
            {
                "release_id": "v1.0.0",
                "release_date": "2026-05-15",
                "description": "Initial sample",
                "status": "open",
            }
        )
        + "\n"
    )
    task = {
        "task_id": "task-001",
        "title": "Will the launch happen?",
        "outcomes": ["Yes", "No"],
        "source": "manual-001",
        "context": "Resolves Yes if the launch happens.",
        "metadata": {"category": "Science and Technology"},
    }
    if include_deadline:
        task["predict_by"] = "2026-05-13T00:00:00Z"
    resolved = {
        "task_id": "task-002",
        "title": "Already resolved?",
        "outcomes": ["Yes", "No"],
        "predict_by": "2026-05-13T00:00:00Z",
        "resolved_outcome": {"value": ["Yes"]},
    }
    (release_dir / "tasks.jsonl").write_text(
        json.dumps(task) + "\n" + json.dumps(resolved) + "\n"
    )


def test_retrieve_dataset_events_maps_latest_open_release(tmp_path: Path):
    _write_dataset(tmp_path)

    events, dataset, release = retrieve_dataset_events(repo_path=str(tmp_path))

    assert dataset == "sample-sports"
    assert release == "v1.0.0"
    assert len(events) == 1
    event = events[0]
    assert event.market_ticker == "task-001"
    assert event.event_ticker == "manual-001"
    assert event.category == "Science and Technology"
    assert event.rules == "Resolves Yes if the launch happens."
    assert event.outcomes == ["Yes", "No"]


def test_retrieve_dataset_events_can_include_resolved_tasks(tmp_path: Path):
    _write_dataset(tmp_path)

    events, _, _ = retrieve_dataset_events(
        repo_path=str(tmp_path),
        include_resolved=True,
    )

    assert [e.market_ticker for e in events] == ["task-001", "task-002"]


def test_retrieve_dataset_events_requires_deadline(tmp_path: Path):
    _write_dataset(tmp_path, include_deadline=False)

    with pytest.raises(ValueError, match="missing a forecast deadline"):
        retrieve_dataset_events(repo_path=str(tmp_path))
