"""Edge pipeline: read sensor -> add weather -> predict crop -> publish (paper Fig. 11)."""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from crop_recommendation.config import FEATURES
from crop_recommendation.edge.sensor import SensorError, SoilReading, SoilSensor
from crop_recommendation.edge.thingsboard import build_telemetry
from crop_recommendation.edge.weather import monthly_rainfall
from crop_recommendation.persistence import ModelBundle

LOGGER = logging.getLogger(__name__)


class HumiditySource(Protocol):
    def fetch_humidity(self) -> float | None: ...


class TelemetrySink(Protocol):
    def send(self, payload: dict[str, Any]) -> bool: ...


@dataclass(frozen=True)
class CycleResult:
    """Everything produced by one sensing-and-prediction cycle."""

    reading: SoilReading
    humidity: float
    rainfall: float
    crop: str
    payload: dict[str, Any]
    published: bool


class StaticHumidity:
    """Constant humidity for ``--dry-run``."""

    def __init__(self, humidity: float = 71.0) -> None:
        self._humidity = humidity

    def fetch_humidity(self) -> float | None:
        return self._humidity


class LogTelemetry:
    """Telemetry sink that only logs the payload (``--dry-run``)."""

    def send(self, payload: dict[str, Any]) -> bool:
        LOGGER.info("[dry-run] telemetry payload: %s", payload)
        return True


class CsvLog:
    """Append-only offline record of every cycle, with timestamps (paper Section 3.5)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists()
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload))
            if is_new:
                writer.writeheader()
            writer.writerow(payload)


def feature_frame(reading: SoilReading, humidity: float, rainfall: float) -> pd.DataFrame:
    """One-row frame with the model's eight features (EC is deliberately excluded)."""
    row = {
        "N": reading.nitrogen,
        "P": reading.phosphorus,
        "K": reading.potassium,
        "pH": reading.ph,
        "Temp(°C)": reading.temperature,
        "Humidity(%)": humidity,
        "Moisture(%)": reading.moisture,
        "Rainfall(cm)": rainfall,
    }
    frame: pd.DataFrame = pd.DataFrame([row])[list(FEATURES)]
    return frame


class EdgeService:
    """Runs the sensing pipeline once or repeatedly."""

    def __init__(
        self,
        bundle: ModelBundle,
        sensor: SoilSensor,
        weather: HumiditySource,
        telemetry: TelemetrySink,
        *,
        csv_log: CsvLog | None = None,
        clock: Callable[[], float] = time.time,
        month: Callable[[], int] = lambda: time.localtime().tm_mon,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._bundle = bundle
        self._sensor = sensor
        self._weather = weather
        self._telemetry = telemetry
        self._csv_log = csv_log
        self._clock = clock
        self._month = month
        self._sleep = sleep

    def run_once(self) -> CycleResult | None:
        """One cycle; returns ``None`` when humidity or the sensor is unavailable."""
        humidity = self._weather.fetch_humidity()
        if humidity is None:
            LOGGER.warning("Humidity is not fetched from API.")
            return None
        try:
            reading = self._sensor.read()
        except SensorError as error:
            LOGGER.error("%s", error)
            return None

        rainfall = monthly_rainfall(self._month())
        frame = feature_frame(reading, humidity, rainfall)
        LOGGER.info("Processed sensor data:\n%s", frame.to_string(index=False))

        crop = str(self._bundle.predict(frame)[0])
        LOGGER.info("Prediction: %s", crop)

        payload = build_telemetry(reading, humidity, rainfall, crop, int(self._clock() * 1000))
        published = self._telemetry.send(payload)
        if self._csv_log is not None:
            self._csv_log.append(payload)
        return CycleResult(reading, humidity, rainfall, crop, payload, published)

    def run(
        self,
        *,
        startup_delay: float = 60.0,
        interval: float = 0.0,
        max_cycles: int | None = None,
    ) -> list[CycleResult]:
        """Wait for sensor stabilisation, then run one cycle or loop every ``interval`` seconds."""
        if startup_delay > 0:
            LOGGER.info(
                "Waiting %.0f s for the sensor to stabilise before collecting data...",
                startup_delay,
            )
            self._sleep(startup_delay)
        results: list[CycleResult] = []
        cycles = 0
        while True:
            result = self.run_once()
            cycles += 1
            if result is not None:
                results.append(result)
            if interval <= 0 or (max_cycles is not None and cycles >= max_cycles):
                break
            self._sleep(interval)
        return results
