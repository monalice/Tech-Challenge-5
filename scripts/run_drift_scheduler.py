from __future__ import annotations

import logging
import os
import sys
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stockcast.drift.scheduler")

RETRAIN_EXIT_CODE = 20
FAILED_EXIT_CODE = 2


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def execute_drift_check(
    api_base_url: str,
    ticker: str,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    url = f"{_normalize_base_url(api_base_url)}/admin/check-drift"
    response = requests.post(
        url,
        json={"ticker": ticker},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload: dict[str, Any] = dict(response.json())
    logger.info("Drift check executado: %s", payload)
    return payload


def main() -> None:
    api_base_url = os.getenv("DRIFT_AUTOMATION_API_URL", "http://127.0.0.1:8000")
    ticker = os.getenv("DRIFT_AUTOMATION_TICKER", "BTC-USD")

    try:
        payload = execute_drift_check(api_base_url=api_base_url, ticker=ticker)
    except Exception as exc:
        logger.exception("Falha na execução de drift one-shot: %s", exc)
        raise SystemExit(FAILED_EXIT_CODE) from exc

    automation = payload.get("automation") if isinstance(payload, dict) else None
    action = automation.get("action") if isinstance(automation, dict) else "none"

    if action == "trigger_retrain":
        logger.warning("Drift acima do limite de retrain. Sinalizando Step Functions para treino.")
        raise SystemExit(RETRAIN_EXIT_CODE)

    logger.info("Drift dentro dos limites. Nenhum retrain necessário.")
    raise SystemExit(0)


if __name__ == "__main__":
    sys.exit(main())
