"""
Post-run validation checklist for ECS training runs.

Usage:
    python scripts/validate_training_run.py [--run-id <mlflow_run_id>]

If --run-id is omitted, validates the most recent run in the experiment.
Exit codes: 0 = all checks passed, 1 = one or more checks failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "btc-hourly-forecast")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "btc_hourly_forecaster")
BASELINE_MAE = float(os.getenv("BASELINE_MAE", "0.005"))

# Tags that MUST be present and non-empty for full governance
REQUIRED_TAGS = [
    "git_sha",
    "lineage_complete",
    "fairness_checked",
    "model_name",
    "model_type",
    "owner",
    "risk_level",
    "training_data_version",
]

# Tags that should not be "unknown" for a production-ready run
KNOWN_VALUE_TAGS = ["git_sha", "training_data_version"]

# Metrics that must exist and be <= BASELINE_MAE for at least one fold
MAE_METRIC_PREFIX = "val_mae"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    rubric_ref: str = ""


@dataclass
class ValidationReport:
    run_id: str
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print(self) -> None:
        width = 70
        print("=" * width)
        print(f"  VALIDATION REPORT  —  run_id: {self.run_id}")
        print("=" * width)
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            ref = f"  [{c.rubric_ref}]" if c.rubric_ref else ""
            print(f"{icon} {c.name}{ref}")
            print(f"   {c.detail}")
        print("-" * width)
        if self.passed:
            print("RESULT: ALL CHECKS PASSED — run is production-ready")
        else:
            failed = sum(1 for c in self.checks if not c.passed)
            print(f"RESULT: {failed} CHECK(S) FAILED — see details above")
        print("=" * width)


def _get_run(client: MlflowClient, run_id: str | None) -> Any:
    if run_id:
        return client.get_run(run_id)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"[ERROR] Experiment '{EXPERIMENT_NAME}' not found in MLflow.", file=sys.stderr)
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"[ERROR] No finished runs found in experiment '{EXPERIMENT_NAME}'.", file=sys.stderr)
        sys.exit(1)
    return runs[0]


def check_required_tags(tags: dict[str, str]) -> list[CheckResult]:
    results = []
    for tag in REQUIRED_TAGS:
        value = tags.get(tag, "")
        passed = bool(value)
        results.append(
            CheckResult(
                name=f"tag:{tag}",
                passed=passed,
                detail=f"value={value!r}" if passed else f"MISSING or empty (got {value!r})",
                rubric_ref="governance/lineage",
            )
        )
    return results


def check_known_values(tags: dict[str, str]) -> list[CheckResult]:
    results = []
    for tag in KNOWN_VALUE_TAGS:
        value = tags.get(tag, "unknown")
        passed = value not in ("", "unknown", "None", "none")
        results.append(
            CheckResult(
                name=f"known_value:{tag}",
                passed=passed,
                detail=(
                    f"value={value!r}"
                    if passed
                    else (
                        f"Value is placeholder: {value!r}"
                        " — bake SHA via CD or inject runtime env var"
                    )
                ),
                rubric_ref="governance/reproducibility",
            )
        )
    return results


def check_lineage_complete(tags: dict[str, str]) -> CheckResult:
    value = tags.get("lineage_complete", "False")
    passed = value.lower() in ("true", "1", "yes")
    alert = tags.get("lineage_alert", "")
    detail = (
        "lineage_complete=True"
        if passed
        else f"lineage_complete={value!r}, alert={alert!r}"
    )
    return CheckResult(
        name="lineage_complete",
        passed=passed,
        detail=detail,
        rubric_ref="governance/lineage",
    )


def check_fairness(tags: dict[str, str]) -> CheckResult:
    value = tags.get("fairness_checked", "False")
    passed = value.lower() in ("true", "1", "yes")
    alert = tags.get("fairness_alert", "")
    detail = (
        "fairness_checked=True"
        if passed
        else (
            f"fairness_checked={value!r}, alert={alert!r}"
            " — ensure evaluation/fairness_report.json is present in the container"
        )
    )
    return CheckResult(
        name="fairness_checked",
        passed=passed,
        detail=detail,
        rubric_ref="governance/fairness",
    )


def check_metrics_vs_baseline(metrics: dict[str, float]) -> list[CheckResult]:
    results = []
    mae_keys = [k for k in metrics if k.startswith(MAE_METRIC_PREFIX)]

    if not mae_keys:
        results.append(
            CheckResult(
                name="metrics:val_mae_present",
                passed=False,
                detail=f"No metrics starting with '{MAE_METRIC_PREFIX}' found in run",
                rubric_ref="performance/mae",
            )
        )
        return results

    results.append(
        CheckResult(
            name="metrics:val_mae_present",
            passed=True,
            detail=f"Found: {sorted(mae_keys)}",
            rubric_ref="performance/mae",
        )
    )

    best_mae = min(metrics[k] for k in mae_keys)
    beats_baseline = best_mae <= BASELINE_MAE
    results.append(
        CheckResult(
            name="metrics:best_val_mae_vs_baseline",
            passed=beats_baseline,
            detail=f"best_val_mae={best_mae:.6f}, baseline={BASELINE_MAE:.6f}"
            + ("  ✓ beats baseline" if beats_baseline else "  ✗ does NOT beat baseline"),
            rubric_ref="performance/champion_challenger",
        )
    )

    # Check for fold consistency (CV std not too high)
    if len(mae_keys) > 1:
        fold_values = [metrics[k] for k in sorted(mae_keys)]
        mean_mae = sum(fold_values) / len(fold_values)
        std_mae = (sum((v - mean_mae) ** 2 for v in fold_values) / len(fold_values)) ** 0.5
        cv_stable = std_mae / mean_mae < 0.3  # <30% relative std
        results.append(
            CheckResult(
                name="metrics:cv_fold_stability",
                passed=cv_stable,
                detail=(
                    f"fold MAEs={[f'{v:.6f}' for v in fold_values]}, "
                    f"mean={mean_mae:.6f}, std={std_mae:.6f} "
                    f"({'stable' if cv_stable else 'HIGH VARIANCE'})"
                ),
                rubric_ref="performance/robustness",
            )
        )

    return results


def check_model_registered(client: MlflowClient, run_id: str) -> CheckResult:
    versions = client.search_model_versions(f"run_id='{run_id}'")
    registered = len(versions) > 0
    detail = (
        f"Registered as '{MLFLOW_MODEL_NAME}' version(s): {[v.version for v in versions]}"
        if registered
        else f"No model version registered in Registry for run_id={run_id}"
    )
    return CheckResult(
        name="model:registered_in_registry",
        passed=registered,
        detail=detail,
        rubric_ref="mlops/model_registry",
    )


def check_promotion_gate(client: MlflowClient) -> CheckResult:
    try:
        versions = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion")
        detail = f"Champion alias points to version={versions.version}, run_id={versions.run_id}"
        return CheckResult(
            name="promotion:champion_alias_set",
            passed=True,
            detail=detail,
            rubric_ref="mlops/champion_challenger",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="promotion:champion_alias_set",
            passed=False,
            detail=f"No '@champion' alias found: {exc}",
            rubric_ref="mlops/champion_challenger",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an ECS training run against the datathon rubric."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="MLflow run ID to validate. Defaults to latest finished run.",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    run = _get_run(client, args.run_id)
    run_id = run.info.run_id
    tags = run.data.tags or {}
    metrics = run.data.metrics or {}

    report = ValidationReport(run_id=run_id)

    for result in check_required_tags(tags):
        report.add(result)

    for result in check_known_values(tags):
        report.add(result)

    report.add(check_lineage_complete(tags))
    report.add(check_fairness(tags))

    for result in check_metrics_vs_baseline(metrics):
        report.add(result)

    report.add(check_model_registered(client, run_id))
    report.add(check_promotion_gate(client))

    report.print()
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
