"""Thread-safe shared I2C bus singleton for SensorHead.

All I2C devices (MLX90640, BME688, PCA9685) share bus 1.
Acquire the lock before any read/write operation.
"""

import threading

import smbus2


class I2CBus:
    """Singleton I2C bus with threading lock for safe concurrent access."""

    _instance: "I2CBus | None" = None
    _init_lock = threading.Lock()

    def __new__(cls, bus_num: int = 1) -> "I2CBus":
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._bus = smbus2.SMBus(bus_num)
                inst._bus_lock = threading.Lock()
                inst._bus_num = bus_num
                cls._instance = inst
            return cls._instance

    @property
    def lock(self) -> threading.Lock:
        """Acquire this lock before any I2C operation."""
        return self._bus_lock

    @property
    def bus(self) -> smbus2.SMBus:
        return self._bus

    @property
    def bus_num(self) -> int:
        return self._bus_num

    def scan(self) -> list[int]:
        """Scan I2C bus for connected devices. Returns list of addresses."""
        devices = []
        with self._bus_lock:
            for addr in range(0x03, 0x78):
                try:
                    self._bus.read_byte(addr)
                    devices.append(addr)
                except OSError:
                    pass
        return devices

    def close(self) -> None:
        with self._bus_lock:
            self._bus.close()
        I2CBus._instance = None
