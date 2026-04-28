from __future__ import annotations

import logging
import os
from typing import Any

import requests
from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore[import-not-found]

from src.serving.drift_automation import DriftAutomationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stockcast.drift.scheduler")


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
    config = DriftAutomationConfig.from_sources()
    api_base_url = os.getenv("DRIFT_AUTOMATION_API_URL", "http://127.0.0.1:8000")
    ticker = os.getenv("DRIFT_AUTOMATION_TICKER", "BTC-USD")

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        execute_drift_check,
        trigger="interval",
        hours=config.check_interval_hours,
        kwargs={
            "api_base_url": api_base_url,
            "ticker": ticker,
        },
        id="stockcast-drift-check",
        replace_existing=True,
    )

    logger.info(
        "Scheduler iniciado: intervalo=%sh endpoint=%s/admin/check-drift ticker=%s",
        config.check_interval_hours,
        _normalize_base_url(api_base_url),
        ticker,
    )

    execute_drift_check(api_base_url=api_base_url, ticker=ticker)
    scheduler.start()


if __name__ == "__main__":
    main()
