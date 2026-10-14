"""ThingsBoard telemetry upload over HTTP (paper Section 3.5, Fig. 12)."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

from crop_recommendation.edge.sensor import SoilReading

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = "https://demo.thingsboard.io"
DEFAULT_TIMEOUT_SECONDS = 10.0


def build_telemetry(
    reading: SoilReading,
    humidity: float,
    rainfall: float,
    crop: str,
    timestamp_ms: int,
) -> dict[str, Any]:
    """JSON payload with the keys the ThingsBoard dashboard widgets are bound to."""
    return {
        "N": int(reading.nitrogen),
        "P": int(reading.phosphorus),
        "K": int(reading.potassium),
        "Moisture": float(reading.moisture),
        "Temperature": float(reading.temperature),
        "pH": float(reading.ph),
        "Humidity": float(humidity),
        "Rainfall": float(rainfall),
        "RecommendedCrop": crop,
        "EC": reading.ec,
        "ts": timestamp_ms,
    }


class TelemetryClient:
    """Posts telemetry to ``<host>/api/v1/<access token>/telemetry``.

    Use as a context manager so the underlying HTTP session is always closed.
    """

    def __init__(
        self,
        access_token: str,
        *,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        if session is None:
            import requests

            session = requests.Session()
        self._session = session
        self._url = f"{host.rstrip('/')}/api/v1/{access_token}/telemetry"
        self._timeout = timeout

    def __enter__(self) -> TelemetryClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._session.close()

    def send(self, payload: dict[str, Any]) -> bool:
        """Upload ``payload``; return ``True`` on HTTP 200."""
        try:
            response = self._session.post(
                self._url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self._timeout,
            )
        except Exception as error:
            LOGGER.error("Failed to send data to ThingsBoard: %s", error)
            return False
        if response.status_code == 200:
            LOGGER.info("Data sent to ThingsBoard successfully")
            return True
        LOGGER.error(
            "Failed to send data to ThingsBoard. Status code: %s, response: %s",
            response.status_code,
            response.text,
        )
        return False
