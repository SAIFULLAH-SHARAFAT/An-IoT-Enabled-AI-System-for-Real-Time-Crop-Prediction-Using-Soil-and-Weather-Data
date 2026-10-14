"""Runtime configuration for the edge service. Secrets come from the environment only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from crop_recommendation.edge.thingsboard import DEFAULT_HOST
from crop_recommendation.edge.weather import DEFAULT_API_URL

WEATHER_KEY_ENV = "WEATHER_API_KEY"
THINGSBOARD_TOKEN_ENV = "TB_ACCESS_TOKEN"


class SettingsError(ValueError):
    """Raised when required edge settings are missing."""


@dataclass(frozen=True)
class EdgeSettings:
    """All parameters of ``crop-rec edge`` (defaults mirror the deployed prototype)."""

    model_path: Path | None = None
    scaler_path: Path | None = None
    encoder_path: Path | None = None
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 4800
    unit_id: int = 1
    serial_timeout: float = 1.0
    location: str = "Dhaka"
    weather_url: str = DEFAULT_API_URL
    thingsboard_host: str = DEFAULT_HOST
    weather_api_key: str | None = None
    thingsboard_token: str | None = None
    startup_delay: float = 60.0
    interval: float = 0.0
    csv_log: Path | None = None
    dry_run: bool = False

    @staticmethod
    def secrets_from_env() -> tuple[str | None, str | None]:
        return os.environ.get(WEATHER_KEY_ENV), os.environ.get(THINGSBOARD_TOKEN_ENV)

    def validate(self) -> None:
        """Check that the model location and (outside dry-run) credentials are present."""
        legacy = (self.scaler_path, self.encoder_path)
        if self.model_path is None:
            raise SettingsError("--model-path is required")
        if any(legacy) and not all(legacy):
            raise SettingsError("--scaler-path and --encoder-path must be given together")
        if self.dry_run:
            return
        if not self.weather_api_key:
            raise SettingsError(
                f"Weather API key missing: set ${WEATHER_KEY_ENV} or pass --weather-api-key"
            )
        if not self.thingsboard_token:
            raise SettingsError(
                f"ThingsBoard token missing: set ${THINGSBOARD_TOKEN_ENV} "
                "or pass --thingsboard-token"
            )
