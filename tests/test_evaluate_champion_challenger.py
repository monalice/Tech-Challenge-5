from __future__ import annotations

from types import SimpleNamespace

from scripts import evaluate_champion_challenger as ecc


class _FakeClient:
    def __init__(self, run_tags: dict[str, str]) -> None:
        self._run_tags = run_tags

    def get_run(self, run_id: str):
        return SimpleNamespace(data=SimpleNamespace(tags=self._run_tags))


class _AliasClient:
    def __init__(self, alias_to_version: dict[str, tuple[str, str]]) -> None:
        self.alias_to_version = alias_to_version

    def get_model_version_by_alias(self, model_name: str, alias: str):
        del model_name
        if alias not in self.alias_to_version:
            raise RuntimeError(f"alias {alias} not found")
        version, run_id = self.alias_to_version[alias]
        return SimpleNamespace(version=version, run_id=run_id)


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


def test_resolve_candidate_version_fallbacks_to_staging_alias(monkeypatch):
    monkeypatch.setattr(ecc, "CANDIDATE_ALIAS", "candidate")
    client = _AliasClient({"Staging": ("5", "run-staging")})

    version, run_id = ecc._resolve_candidate_version(client)

    assert version == "5"
    assert run_id == "run-staging"


def test_resolve_candidate_version_raises_when_all_aliases_missing(monkeypatch):
    monkeypatch.setattr(ecc, "CANDIDATE_ALIAS", "candidate")
    client = _AliasClient({})

    try:
        ecc._resolve_candidate_version(client)
    except RuntimeError as exc:
        assert "Nenhum alias de challenger encontrado" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when no challenger aliases exist")
