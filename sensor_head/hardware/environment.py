"""BME688 environmental sensor wrapper.

Reads temperature, humidity, pressure, and gas resistance.
Uses adafruit_bme680 library (compatible with BME688).
"""

import json
import logging
import time

from sensor_head.config import SensorHeadConfig
from sensor_head.hardware.i2c_bus import I2CBus

logger = logging.getLogger("sensor-head.environment")


class EnvironmentSensor:
    """BME688 environmental sensor via I2C."""

    def __init__(self, config: SensorHeadConfig) -> None:
        self._config = config
        self._sensor = None
        self._available = False

    def _init_sensor(self) -> None:
        """Lazy-initialize the BME688 sensor."""
        if self._sensor is not None:
            return

        import board
        import adafruit_bme680

        bus = I2CBus(self._config.i2c_bus)

        # Try primary address, then fallback
        for addr in [self._config.bme688_addr, self._config.bme688_addr_alt]:
            try:
                with bus.lock:
                    i2c = board.I2C()
                self._sensor = adafruit_bme680.Adafruit_BME680_I2C(
                    i2c, address=addr
                )
                # Configure gas heater for VOC measurement
                self._sensor.sea_level_pressure = 1013.25
                self._available = True
                logger.info(f"BME688 initialized at address 0x{addr:02X}")
                return
            except (ValueError, OSError) as e:
                logger.debug(f"BME688 not found at 0x{addr:02X}: {e}")
                continue

        logger.warning("BME688 not found on any address")
        self._available = False

    @property
    def available(self) -> bool:
        if self._sensor is None:
            self._init_sensor()
        return self._available

    def read(self) -> dict:
        """Read all environmental channels.

        Returns dict with temperature_c, humidity_pct, pressure_hpa,
        gas_resistance_ohm, and altitude_m.
        """
        self._init_sensor()

        if not self._available:
            return {
                "error": True,
                "sensor": "BME688",
                "status": "unavailable",
                "reason": "Sensor not detected on I2C bus",
            }

        bus = I2CBus(self._config.i2c_bus)

        try:
            with bus.lock:
                temperature = self._sensor.temperature
                humidity = self._sensor.relative_humidity
                pressure = self._sensor.pressure
                gas = self._sensor.gas
                altitude = self._sensor.altitude

            return {
                "temperature_c": round(temperature, 2),
                "humidity_pct": round(humidity, 2),
                "pressure_hpa": round(pressure, 2),
                "gas_resistance_ohm": round(gas, 0) if gas else None,
                "altitude_m": round(altitude, 1),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"BME688 read error: {e}")
            return {
                "error": True,
                "sensor": "BME688",
                "status": "read_error",
                "reason": str(e),
            }

    def read_json(self) -> str:
        """Read and return as JSON string."""
        return json.dumps(self.read(), indent=2)
