"""Run the Raspberry Pi service: sensor + weather -> crop -> ThingsBoard."""

from __future__ import annotations

import argparse
import logging
from contextlib import ExitStack
from pathlib import Path

from crop_recommendation.edge.sensor import ModbusSensor, SensorError, SimulatedSensor, SoilSensor
from crop_recommendation.edge.service import (
    CsvLog,
    EdgeService,
    HumiditySource,
    LogTelemetry,
    StaticHumidity,
    TelemetrySink,
)
from crop_recommendation.edge.settings import (
    THINGSBOARD_TOKEN_ENV,
    WEATHER_KEY_ENV,
    EdgeSettings,
    SettingsError,
)
from crop_recommendation.edge.thingsboard import DEFAULT_HOST, TelemetryClient
from crop_recommendation.edge.weather import DEFAULT_API_URL, WeatherClient
from crop_recommendation.persistence import ModelBundle

LOGGER = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "edge",
        help="Run the Raspberry Pi sensing service.",
        description=__doc__,
        epilog=(
            f"Secrets are read from ${WEATHER_KEY_ENV} and ${THINGSBOARD_TOKEN_ENV} "
            "when not given as flags."
        ),
    )
    model = parser.add_argument_group("model")
    model.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Bundle from `crop-rec train`, or a legacy model .pkl.",
    )
    model.add_argument(
        "--scaler-path", type=Path, help="Legacy scaler .pkl (use with --encoder-path)."
    )
    model.add_argument("--encoder-path", type=Path, help="Legacy label-encoder .pkl.")

    sensor = parser.add_argument_group("sensor (RS-485 over USB, Modbus-RTU)")
    sensor.add_argument(
        "--port", default="/dev/ttyUSB0", help="Serial port (default: %(default)s)."
    )
    sensor.add_argument(
        "--baudrate", type=int, default=4800, help="Baud rate (default: %(default)s)."
    )
    sensor.add_argument(
        "--unit-id", type=int, default=1, help="Modbus slave id (default: %(default)s)."
    )

    cloud = parser.add_argument_group("weather and dashboard")
    cloud.add_argument(
        "--location", default="Dhaka", help="WeatherAPI location query (default: %(default)s)."
    )
    cloud.add_argument("--weather-url", default=DEFAULT_API_URL, help=argparse.SUPPRESS)
    cloud.add_argument("--weather-api-key", help=f"WeatherAPI key (default: ${WEATHER_KEY_ENV}).")
    cloud.add_argument(
        "--thingsboard-host", default=DEFAULT_HOST, help="ThingsBoard URL (default: %(default)s)."
    )
    cloud.add_argument(
        "--thingsboard-token", help=f"Device access token (default: ${THINGSBOARD_TOKEN_ENV})."
    )

    run = parser.add_argument_group("run control")
    run.add_argument(
        "--startup-delay",
        type=float,
        default=60.0,
        help="Seconds to let the sensor stabilise (paper: one minute; default: %(default)s).",
    )
    run.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Repeat every N seconds; 0 runs a single cycle and exits (default: %(default)s).",
    )
    run.add_argument("--csv-log", type=Path, help="Also append every cycle to this CSV file.")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate sensor and weather and log instead of uploading (no hardware or network).",
    )
    parser.set_defaults(handler=run_command)


def _settings(args: argparse.Namespace) -> EdgeSettings:
    env_weather, env_token = EdgeSettings.secrets_from_env()
    return EdgeSettings(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        encoder_path=args.encoder_path,
        serial_port=args.port,
        baudrate=args.baudrate,
        unit_id=args.unit_id,
        location=args.location,
        weather_url=args.weather_url,
        thingsboard_host=args.thingsboard_host,
        weather_api_key=args.weather_api_key or env_weather,
        thingsboard_token=args.thingsboard_token or env_token,
        startup_delay=args.startup_delay,
        interval=args.interval,
        csv_log=args.csv_log,
        dry_run=args.dry_run,
    )


def _load_bundle(settings: EdgeSettings) -> ModelBundle:
    if settings.model_path is None:
        raise SettingsError("--model-path is required")
    if settings.scaler_path and settings.encoder_path:
        return ModelBundle.from_legacy_files(
            model_path=settings.model_path,
            scaler_path=settings.scaler_path,
            encoder_path=settings.encoder_path,
        )
    return ModelBundle.load(settings.model_path)


def run_command(args: argparse.Namespace) -> int:
    settings = _settings(args)
    try:
        settings.validate()
        bundle = _load_bundle(settings)
    except (SettingsError, FileNotFoundError, TypeError) as error:
        LOGGER.error("%s", error)
        return 2

    with ExitStack() as stack:
        sensor: SoilSensor
        weather: HumiditySource
        telemetry: TelemetrySink
        try:
            if settings.dry_run:
                LOGGER.info("Dry run: simulated sensor and weather, nothing is uploaded")
                sensor = stack.enter_context(SimulatedSensor())
                weather, telemetry = StaticHumidity(), LogTelemetry()
            else:
                sensor = stack.enter_context(
                    ModbusSensor(
                        settings.serial_port,
                        baudrate=settings.baudrate,
                        unit_id=settings.unit_id,
                        timeout=settings.serial_timeout,
                    )
                )
                weather = stack.enter_context(
                    WeatherClient(
                        settings.weather_api_key or "",
                        settings.location,
                        api_url=settings.weather_url,
                    )
                )
                telemetry = stack.enter_context(
                    TelemetryClient(
                        settings.thingsboard_token or "", host=settings.thingsboard_host
                    )
                )
        except SensorError as error:
            LOGGER.error("%s", error)
            return 1

        service = EdgeService(
            bundle,
            sensor,
            weather,
            telemetry,
            csv_log=CsvLog(settings.csv_log) if settings.csv_log else None,
        )
        try:
            results = service.run(startup_delay=settings.startup_delay, interval=settings.interval)
        except KeyboardInterrupt:
            LOGGER.info("Program terminated by user.")
            return 130
    LOGGER.info("Completed %d cycle(s)", len(results))
    return 0 if results else 1
