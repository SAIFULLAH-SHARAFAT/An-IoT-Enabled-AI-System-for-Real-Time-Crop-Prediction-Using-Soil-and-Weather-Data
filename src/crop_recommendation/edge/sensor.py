"""RS-485 7-in-1 soil sensor over Modbus-RTU (paper Section 3.5, Fig. 10-11)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)

#: Holding registers 0..6: moisture, temperature, EC, pH, N, P, K.
REGISTER_COUNT = 7
START_ADDRESS = 0


class SensorError(RuntimeError):
    """Raised when the sensor cannot be reached or returns an error response."""


@dataclass(frozen=True)
class SoilReading:
    """One decoded sensor sample in the units used for training."""

    moisture: float  # %
    temperature: float  # degrees Celsius
    ec: int  # electrical conductivity, uS/cm (sent to ThingsBoard, not used by the model)
    ph: float
    nitrogen: int  # mg/kg
    phosphorus: int  # mg/kg
    potassium: int  # mg/kg


def parse_registers(registers: Sequence[int]) -> SoilReading:
    """Convert raw holding-register values into physical units.

    Scaling follows the deployed prototype: moisture, temperature and pH are stored in tenths;
    N, P and K are multiplied by two.
    """
    if len(registers) < REGISTER_COUNT:
        raise SensorError(f"Expected {REGISTER_COUNT} registers, received {len(registers)}")
    return SoilReading(
        moisture=registers[0] / 10.0,
        temperature=registers[1] / 10.0,
        ec=registers[2],
        ph=registers[3] / 10.0,
        nitrogen=registers[4] * 2,
        phosphorus=registers[5] * 2,
        potassium=registers[6] * 2,
    )


class SoilSensor(Protocol):
    """Anything that yields soil readings (real hardware or a simulator)."""

    def __enter__(self) -> SoilSensor: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...
    def read(self) -> SoilReading: ...


class ModbusSensor:
    """Context manager around a ``pymodbus`` serial client.

    The serial port is opened on entry and always closed on exit.
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 4800,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        timeout: float = 1.0,
        unit_id: int = 1,
        client: Any | None = None,
    ) -> None:
        self._unit_id = unit_id
        self._client = (
            client
            if client is not None
            else self._create_client(port, baudrate, bytesize, parity, stopbits, timeout)
        )

    @staticmethod
    def _create_client(
        port: str, baudrate: int, bytesize: int, parity: str, stopbits: int, timeout: float
    ) -> Any:
        try:
            from pymodbus.client import ModbusSerialClient
        except ImportError as error:
            raise SensorError(
                "pymodbus is required: pip install 'crop-recommendation[edge]'"
            ) from error
        return ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=timeout,
        )

    def __enter__(self) -> ModbusSensor:
        if not self._client.connect():
            raise SensorError("Failed to connect to the Modbus client")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._client.close()
        LOGGER.info("Modbus client disconnected")

    def _unit_kwarg(self) -> str:
        """pymodbus renamed ``slave`` to ``device_id`` in 3.10; support both."""
        parameters = inspect.signature(self._client.read_holding_registers).parameters
        return "device_id" if "device_id" in parameters else "slave"

    def read(self) -> SoilReading:
        result = self._client.read_holding_registers(
            START_ADDRESS, count=REGISTER_COUNT, **{self._unit_kwarg(): self._unit_id}
        )
        if result.isError():
            raise SensorError("Modbus error while reading registers")
        LOGGER.debug("Modbus registers: %s", result.registers)
        return parse_registers(result.registers)


class SimulatedSensor:
    """Fixed readings for ``--dry-run`` and tests (close to the dataset means)."""

    DEFAULT_REGISTERS = (656, 259, 1200, 61, 110, 30, 45)

    def __init__(self, registers: Sequence[int] = DEFAULT_REGISTERS) -> None:
        self._registers = tuple(registers)

    def __enter__(self) -> SimulatedSensor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> SoilReading:
        return parse_registers(self._registers)
