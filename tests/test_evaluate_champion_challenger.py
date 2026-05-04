from __future__ import annotations

from types import SimpleNamespace

from scripts import evaluate_champion_challenger as ecc


class _FakeClient:
    def __init__(self, run_tags: dict[str, str]) -> None:
        self._run_tags = run_tags

    def get_run(self, run_id: str):
        return SimpleNamespace(data=SimpleNamespace(tags=self._run_tags))


def test_validate_challenger_lineage_accepts_complete_lineage():
    client = _FakeClient({"lineage_complete": "True", "git_sha": "abc123"})

    ok, reason = ecc._validate_challenger_lineage(client, "run-1")

    assert ok is True
    assert reason == ""


def test_validate_challenger_lineage_rejects_missing_lineage_flag():
    client = _FakeClient({"git_sha": "abc123"})

    ok, reason = ecc._validate_challenger_lineage(client, "run-1")

    assert ok is False
    assert "lineage_incomplete" in reason


def test_validate_challenger_lineage_rejects_unknown_git_sha():
    client = _FakeClient({"lineage_complete": "True", "git_sha": "unknown"})

    ok, reason = ecc._validate_challenger_lineage(client, "run-1")

    assert ok is False
    assert "git_sha=unknown" in reason
