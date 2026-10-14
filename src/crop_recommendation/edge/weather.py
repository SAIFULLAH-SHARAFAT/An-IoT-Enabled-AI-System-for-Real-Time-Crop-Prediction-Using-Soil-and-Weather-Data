"""Weather inputs: live humidity from WeatherAPI and monthly rainfall from BAMIS."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Final

LOGGER = logging.getLogger(__name__)

#: Average monthly rainfall by calendar month, from the Bangladesh Agro-Meteorological
#: Information Service (BAMIS), as used by the deployed prototype.
MONTHLY_RAINFALL: Final[dict[int, float]] = {
    1: 9.4,
    2: 36.2,
    3: 155.3,
    4: 375.6,
    5: 569.6,
    6: 818.4,
    7: 819.2,
    8: 612.6,
    9: 535.9,
    10: 223.9,
    11: 30.4,
    12: 9.4,
}

DEFAULT_API_URL: Final[str] = "https://api.weatherapi.com/v1/current.json"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0


def monthly_rainfall(month: int) -> float:
    """Rainfall figure for ``month`` (1-12); unknown months yield 0, as in the prototype."""
    return MONTHLY_RAINFALL.get(month, 0.0)


class WeatherError(RuntimeError):
    """Raised when the weather service cannot be queried."""


class WeatherClient:
    """Fetches current relative humidity (%) for a location from WeatherAPI.

    Use as a context manager so the underlying HTTP session is always closed.
    """

    def __init__(
        self,
        api_key: str,
        location: str,
        *,
        api_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        if session is None:
            import requests

            session = requests.Session()
        self._session = session
        self._api_key = api_key
        self._location = location
        self._api_url = api_url
        self._timeout = timeout

    def __enter__(self) -> WeatherClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._session.close()

    def fetch_humidity(self) -> float | None:
        """Current humidity, or ``None`` when the request fails or the field is absent."""
        try:
            response = self._session.get(
                self._api_url,
                params={"key": self._api_key, "q": self._location},
                timeout=self._timeout,
            )
            response.raise_for_status()
            humidity = response.json().get("current", {}).get("humidity")
        except Exception as error:  # network, HTTP and JSON errors are all non-fatal here
            LOGGER.error("Failed to fetch humidity data: %s", error)
            return None
        if humidity is not None:
            LOGGER.info("Humidity retrieved: %s%%", humidity)
        return None if humidity is None else float(humidity)
