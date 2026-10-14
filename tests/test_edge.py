from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from crop_recommendation.config import FEATURES
from crop_recommendation.edge.sensor import (
    ModbusSensor,
    SensorError,
    SimulatedSensor,
    SoilReading,
    parse_registers,
)
from crop_recommendation.edge.service import (
    CsvLog,
    EdgeService,
    LogTelemetry,
    StaticHumidity,
    feature_frame,
)
from crop_recommendation.edge.settings import EdgeSettings, SettingsError
from crop_recommendation.edge.thingsboard import TelemetryClient, build_telemetry
from crop_recommendation.edge.weather import MONTHLY_RAINFALL, WeatherClient, monthly_rainfall
from crop_recommendation.persistence import ModelBundle

# --------------------------------------------------------------------------- sensor


def original_script_arithmetic(registers: list[int]) -> dict[str, float]:
    """Register scaling copied from the deployed prototype (``Crop_Recommendation(rasp_file).py``)."""
    return {
        "N": registers[4] * 2,
        "P": registers[5] * 2,
        "K": registers[6] * 2,
        "pH": registers[3] / 10.0,
        "Temp(°C)": registers[1] / 10.0,
        "Moisture(%)": registers[0] / 10.0,
        "EC": registers[2],
    }


@pytest.mark.parametrize(
    "registers",
    [[656, 259, 1200, 61, 110, 30, 45], [0, 0, 0, 0, 0, 0, 0], [950, 400, 2000, 70, 199, 222, 225]],
)
def test_register_parsing_matches_deployed_prototype(registers: list[int]) -> None:
    reading = parse_registers(registers)
    expected = original_script_arithmetic(registers)
    assert reading.nitrogen == expected["N"]
    assert reading.phosphorus == expected["P"]
    assert reading.potassium == expected["K"]
    assert reading.ph == expected["pH"]
    assert reading.temperature == expected["Temp(°C)"]
    assert reading.moisture == expected["Moisture(%)"]
    assert reading.ec == expected["EC"]


def test_short_register_block_is_rejected() -> None:
    with pytest.raises(SensorError, match="Expected 7 registers"):
        parse_registers([1, 2, 3])


class FakeResponse:
    def __init__(self, registers: list[int] | None = None, error: bool = False) -> None:
        self.registers = registers or []
        self._error = error

    def isError(self) -> bool:  # noqa: N802 - mirrors the pymodbus API
        return self._error


class FakeModbusClient:
    """Records calls; ``read_holding_registers`` accepts ``device_id`` (pymodbus >= 3.10)."""

    def __init__(self, response: FakeResponse, connects: bool = True) -> None:
        self.response, self.connects = response, connects
        self.closed = False
        self.calls: list[tuple[int, dict[str, Any]]] = []

    def connect(self) -> bool:
        return self.connects

    def close(self) -> None:
        self.closed = True

    def read_holding_registers(
        self, address: int, *, count: int = 1, device_id: int = 1
    ) -> FakeResponse:
        self.calls.append((address, {"count": count, "device_id": device_id}))
        return self.response


class LegacyModbusClient(FakeModbusClient):
    """pymodbus < 3.10 spelled the unit id ``slave``."""

    def read_holding_registers(self, address: int, count: int = 1, slave: int = 0) -> FakeResponse:  # type: ignore[override]
        self.calls.append((address, {"count": count, "slave": slave}))
        return self.response


@pytest.mark.parametrize(
    ("client_cls", "kwarg"), [(FakeModbusClient, "device_id"), (LegacyModbusClient, "slave")]
)
def test_modbus_sensor_reads_and_closes(client_cls: type[FakeModbusClient], kwarg: str) -> None:
    client = client_cls(FakeResponse([656, 259, 1200, 61, 110, 30, 45]))
    with ModbusSensor("/dev/ttyUSB0", client=client, unit_id=1) as sensor:
        reading = sensor.read()
    assert reading.nitrogen == 220
    assert client.calls == [(0, {"count": 7, kwarg: 1})]
    assert client.closed


def test_modbus_sensor_closes_even_when_the_body_fails() -> None:
    client = FakeModbusClient(FakeResponse([0] * 7))
    with pytest.raises(RuntimeError), ModbusSensor("p", client=client):
        raise RuntimeError("boom")
    assert client.closed


def test_modbus_sensor_connection_and_read_errors() -> None:
    with (
        pytest.raises(SensorError, match="connect"),
        ModbusSensor("p", client=FakeModbusClient(FakeResponse(), connects=False)),
    ):
        pass
    with (
        ModbusSensor("p", client=FakeModbusClient(FakeResponse(error=True))) as sensor,
        pytest.raises(SensorError, match="Modbus error"),
    ):
        sensor.read()


# --------------------------------------------------------------------------- weather


def test_rainfall_table_is_complete_and_verbatim() -> None:
    assert sorted(MONTHLY_RAINFALL) == list(range(1, 13))
    assert monthly_rainfall(7) == 819.2
    assert monthly_rainfall(1) == monthly_rainfall(12) == 9.4
    assert monthly_rainfall(13) == 0.0


class FakeSession:
    def __init__(
        self, payload: Any = None, status: int = 200, raises: Exception | None = None
    ) -> None:
        self.payload, self.status, self.raises = payload, status, raises
        self.closed = False
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeSession:
        self.requests.append({"method": "GET", "url": url, **kwargs})
        if self.raises:
            raise self.raises
        return self

    def post(self, url: str, **kwargs: Any) -> FakeSession:
        self.requests.append({"method": "POST", "url": url, **kwargs})
        if self.raises:
            raise self.raises
        return self

    @property
    def status_code(self) -> int:
        return self.status

    text = "response body"

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self) -> Any:
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_weather_client_returns_humidity_and_closes_session() -> None:
    session = FakeSession({"current": {"humidity": 78}})
    with WeatherClient("key", "Dhaka", session=session) as client:
        assert client.fetch_humidity() == 78.0
    assert session.closed
    assert session.requests[0]["params"] == {"key": "key", "q": "Dhaka"}
    assert session.requests[0]["timeout"] > 0


@pytest.mark.parametrize(
    "session",
    [
        FakeSession({"current": {}}),
        FakeSession({}, status=500),
        FakeSession(raises=ConnectionError("down")),
    ],
)
def test_weather_client_failures_yield_none(session: FakeSession) -> None:
    assert WeatherClient("key", "Dhaka", session=session).fetch_humidity() is None


# --------------------------------------------------------------------------- thingsboard


def test_telemetry_payload_keys_and_types() -> None:
    reading = SoilReading(
        moisture=65.6, temperature=25.9, ec=1200, ph=6.1, nitrogen=220, phosphorus=60, potassium=90
    )
    payload = build_telemetry(reading, humidity=71, rainfall=223.9, crop="Tomato", timestamp_ms=123)
    assert list(payload) == [
        "N",
        "P",
        "K",
        "Moisture",
        "Temperature",
        "pH",
        "Humidity",
        "Rainfall",
        "RecommendedCrop",
        "EC",
        "ts",
    ]
    assert isinstance(payload["N"], int) and isinstance(payload["Humidity"], float)
    assert payload["RecommendedCrop"] == "Tomato" and payload["ts"] == 123


def test_telemetry_client_posts_to_device_endpoint() -> None:
    session = FakeSession(status=200)
    with TelemetryClient("TOKEN", host="https://tb.example/", session=session) as client:
        assert client.send({"N": 1}) is True
    request = session.requests[0]
    assert request["url"] == "https://tb.example/api/v1/TOKEN/telemetry"
    assert request["json"] == {"N": 1}
    assert session.closed


def test_telemetry_client_reports_failures() -> None:
    assert TelemetryClient("T", session=FakeSession(status=401)).send({}) is False
    assert TelemetryClient("T", session=FakeSession(raises=TimeoutError("slow"))).send({}) is False


# --------------------------------------------------------------------------- service


class RecordingTelemetry:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def send(self, payload: dict[str, Any]) -> bool:
        self.payloads.append(payload)
        return True


class NoHumidity:
    def fetch_humidity(self) -> float | None:
        return None


class BrokenSensor(SimulatedSensor):
    def read(self) -> SoilReading:
        raise SensorError("sensor unplugged")


def make_service(
    bundle: ModelBundle, **overrides: Any
) -> tuple[EdgeService, RecordingTelemetry, list[float]]:
    telemetry, sleeps = RecordingTelemetry(), []
    arguments: dict[str, Any] = {
        "sensor": SimulatedSensor(),
        "weather": StaticHumidity(71.0),
        "clock": lambda: 1_700_000_000.0,
        "month": lambda: 7,
        "sleep": sleeps.append,
    }
    arguments.update(overrides)
    service = EdgeService(
        bundle, arguments.pop("sensor"), arguments.pop("weather"), telemetry, **arguments
    )
    return service, telemetry, sleeps


def test_feature_frame_excludes_ec_and_follows_training_order() -> None:
    frame = feature_frame(
        parse_registers(SimulatedSensor.DEFAULT_REGISTERS), humidity=71.0, rainfall=100.0
    )
    assert list(frame.columns) == list(FEATURES)
    assert frame.shape == (1, 8)


def test_run_once_publishes_prediction(synthetic_bundle: ModelBundle) -> None:
    service, telemetry, _ = make_service(synthetic_bundle)
    result = service.run_once()
    assert result is not None and result.published
    assert result.rainfall == 819.2  # July
    assert telemetry.payloads[0]["RecommendedCrop"] == result.crop
    assert telemetry.payloads[0]["ts"] == 1_700_000_000_000


def test_run_once_skips_when_humidity_or_sensor_unavailable(synthetic_bundle: ModelBundle) -> None:
    service, telemetry, _ = make_service(synthetic_bundle, weather=NoHumidity())
    assert service.run_once() is None
    service, telemetry, _ = make_service(synthetic_bundle, sensor=BrokenSensor())
    assert service.run_once() is None
    assert telemetry.payloads == []


def test_run_waits_for_stabilisation_then_loops(synthetic_bundle: ModelBundle) -> None:
    service, telemetry, sleeps = make_service(synthetic_bundle)
    results = service.run(startup_delay=60, interval=30, max_cycles=3)
    assert len(results) == len(telemetry.payloads) == 3
    assert sleeps == [60, 30, 30]


def test_single_shot_mode_exits_after_one_cycle(synthetic_bundle: ModelBundle) -> None:
    service, _, sleeps = make_service(synthetic_bundle)
    assert len(service.run(startup_delay=0, interval=0)) == 1
    assert sleeps == []


def test_csv_log_appends_with_header_once(tmp_path: Path, synthetic_bundle: ModelBundle) -> None:
    log_path = tmp_path / "logs" / "readings.csv"
    service, _, _ = make_service(synthetic_bundle, csv_log=CsvLog(log_path))
    service.run(startup_delay=0, interval=1, max_cycles=2)
    rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
    assert len(rows) == 2 and "RecommendedCrop" in rows[0]


def test_log_telemetry_sink_never_fails() -> None:
    assert LogTelemetry().send({"a": 1}) is True


# --------------------------------------------------------------------------- settings


def test_settings_validation() -> None:
    with pytest.raises(SettingsError, match="model-path"):
        EdgeSettings().validate()
    with pytest.raises(SettingsError, match="together"):
        EdgeSettings(model_path=Path("m"), scaler_path=Path("s"), dry_run=True).validate()
    with pytest.raises(SettingsError, match="Weather API key"):
        EdgeSettings(model_path=Path("m")).validate()
    with pytest.raises(SettingsError, match="ThingsBoard token"):
        EdgeSettings(model_path=Path("m"), weather_api_key="k").validate()
    EdgeSettings(model_path=Path("m"), weather_api_key="k", thingsboard_token="t").validate()
    EdgeSettings(model_path=Path("m"), dry_run=True).validate()  # no secrets needed offline
