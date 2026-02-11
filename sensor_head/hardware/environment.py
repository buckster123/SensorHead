"""BME688 environmental sensor wrapper — BSEC2 v2.6.1.0.

Uses Bosch BSEC2 signal processing for full air quality intelligence:
- IAQ (Indoor Air Quality) index 0-500
- CO2 equivalent (ppm, derived from VOC correlation)
- Breath VOC equivalent (ppm, rises near people)
- Compensated temperature and humidity (corrected for self-heating)
- Gas percentage for AI smell classification
- Raw gas resistance, pressure, temperature, humidity

Falls back to raw-only mode if BSEC init fails.
"""

import json
import logging
import threading
import time

from sensor_head.config import SensorHeadConfig

logger = logging.getLogger("sensor-head.environment")

# IAQ quality bands per Bosch spec
IAQ_BANDS = [
    (0, 50, "excellent", "Clean air — fresh and pleasant"),
    (51, 100, "good", "Acceptable air quality"),
    (101, 150, "lightly_polluted", "Sensitive people may notice effects"),
    (151, 200, "moderately_polluted", "Increased discomfort likely"),
    (201, 250, "heavily_polluted", "Significant health effects possible"),
    (251, 350, "severely_polluted", "Health warnings — reduce exposure"),
    (351, 500, "extremely_polluted", "Emergency conditions"),
]

IAQ_ACCURACY_LABELS = [
    "stabilizing",   # 0 — sensor warming up, ~5 min after power-on
    "uncertain",     # 1 — some calibration data, low confidence
    "calibrating",   # 2 — actively learning baseline
    "calibrated",    # 3 — fully calibrated, readings reliable
]


def _iaq_band(iaq: float) -> dict:
    """Classify IAQ value into quality band."""
    for lo, hi, label, desc in IAQ_BANDS:
        if lo <= iaq <= hi:
            return {"quality": label, "description": desc}
    return {"quality": "unknown", "description": "Out of range"}


class EnvironmentSensor:
    """BME688 environmental sensor via BSEC2."""

    def __init__(self, config: SensorHeadConfig) -> None:
        self._config = config
        self._sensor = None
        self._available = False
        self._bsec_active = False
        self._lock = threading.Lock()
        self._last_bsec_data: dict | None = None

    def _init_sensor(self) -> None:
        """Lazy-initialize the BME688 with BSEC2."""
        if self._sensor is not None:
            return

        try:
            from bme68x import BME68X
            import bme68xConstants as cst
            import bsecConstants as bsec

            addr = self._config.bme688_addr_alt  # 0x77 is our board
            self._sensor = BME68X(addr, 1)  # 1 = enable BSEC

            version = self._sensor.get_bsec_version()
            variant = self._sensor.get_variant()
            chip_id = self._sensor.get_chip_id()

            # Set low-power sample rate (3s interval, good balance)
            self._sensor.set_sample_rate(bsec.BSEC_SAMPLE_RATE_LP)

            self._bsec_active = True
            self._available = True
            logger.info(
                f"BME688 initialized with BSEC {version} — "
                f"{variant}, chip 0x{chip_id:02X}, addr 0x{addr:02X}"
            )
        except Exception as e:
            logger.warning(f"BSEC2 init failed ({e}), trying raw mode...")
            self._try_raw_fallback()

    def _try_raw_fallback(self) -> None:
        """Fall back to adafruit_bme680 if BSEC2 fails."""
        try:
            import board
            import adafruit_bme680
            from sensor_head.hardware.i2c_bus import I2CBus

            bus = I2CBus(self._config.i2c_bus)
            for addr in [self._config.bme688_addr, self._config.bme688_addr_alt]:
                try:
                    with bus.lock:
                        i2c = board.I2C()
                    self._sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=addr)
                    self._sensor.sea_level_pressure = 1013.25
                    self._available = True
                    self._bsec_active = False
                    logger.info(f"BME688 fallback to raw mode at 0x{addr:02X}")
                    return
                except (ValueError, OSError):
                    continue
        except Exception as e:
            logger.error(f"BME688 raw fallback also failed: {e}")

        self._available = False

    @property
    def available(self) -> bool:
        if self._sensor is None:
            self._init_sensor()
        return self._available

    @property
    def bsec_active(self) -> bool:
        return self._bsec_active

    def _read_bsec(self) -> dict | None:
        """Read from BSEC2 — may return None if not ready yet."""
        try:
            data = self._sensor.get_bsec_data()
        except Exception as e:
            logger.error(f"BSEC read error: {e}")
            return None

        if data is None or data == {}:
            return None

        self._last_bsec_data = data
        return data

    def read(self) -> dict:
        """Read all environmental channels with BSEC2 intelligence.

        Returns dict with temperature, humidity, pressure, IAQ, CO2
        equivalent, breath VOC, gas resistance, and quality assessment.
        """
        self._init_sensor()

        if not self._available:
            return {
                "error": True,
                "sensor": "BME688",
                "status": "unavailable",
                "reason": "Sensor not detected on I2C bus",
            }

        if self._bsec_active:
            return self._read_bsec_full()
        else:
            return self._read_raw()

    def _read_bsec_full(self) -> dict:
        """Read with BSEC2 processing."""
        with self._lock:
            # BSEC needs a few attempts — it only returns data at its
            # configured sample rate (every ~3s in LP mode)
            data = None
            for _ in range(40):  # up to ~4 seconds of trying
                data = self._read_bsec()
                if data is not None:
                    break
                time.sleep(0.1)

            if data is None:
                # Use last known data if we have it
                if self._last_bsec_data is not None:
                    data = self._last_bsec_data
                    stale = True
                else:
                    return {
                        "error": True,
                        "sensor": "BME688",
                        "status": "warming_up",
                        "reason": "BSEC2 still stabilizing — data not yet available",
                        "bsec_version": self._sensor.get_bsec_version(),
                    }
            else:
                stale = False

            iaq = data.get("iaq", 0)
            iaq_acc = data.get("iaq_accuracy", 0)
            band = _iaq_band(iaq)

            result = {
                # Compensated (BSEC-corrected) values
                "temperature_c": round(data.get("temperature", 0), 2),
                "humidity_pct": round(data.get("humidity", 0), 2),
                "pressure_hpa": round(data.get("raw_pressure", 0) / 100, 2),

                # BSEC air quality intelligence
                "iaq": round(iaq, 1),
                "iaq_accuracy": iaq_acc,
                "iaq_accuracy_label": IAQ_ACCURACY_LABELS[min(iaq_acc, 3)],
                "air_quality": band["quality"],
                "air_quality_description": band["description"],

                # Derived gas metrics
                "co2_equivalent_ppm": round(data.get("co2_equivalent", 0), 1),
                "co2_accuracy": data.get("co2_accuracy", 0),
                "breath_voc_ppm": round(data.get("breath_voc_equivalent", 0), 4),
                "breath_voc_accuracy": data.get("breath_voc_accuracy", 0),

                # AI gas classification
                "gas_percentage": round(data.get("gas_percentage", 0), 2),
                "gas_percentage_accuracy": data.get("gas_percentage_accuracy", 0),

                # Raw sensor values
                "raw_temperature_c": round(data.get("raw_temperature", 0), 2),
                "raw_humidity_pct": round(data.get("raw_humidity", 0), 2),
                "raw_gas_resistance_ohm": round(data.get("raw_gas", 0), 1),

                # Status
                "stabilization_status": data.get("stabilization_status", 0),
                "run_in_status": data.get("run_in_status", 0),
                "bsec_version": self._sensor.get_bsec_version(),
                "stale": stale,
                "timestamp": time.time(),
            }

            return result

    def _read_raw(self) -> dict:
        """Fallback: read raw values via adafruit_bme680."""
        from sensor_head.hardware.i2c_bus import I2CBus
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
                "raw_gas_resistance_ohm": round(gas, 0) if gas else None,
                "altitude_m": round(altitude, 1),
                "bsec_version": "N/A (raw mode)",
                "iaq": None,
                "air_quality": "raw_mode",
                "air_quality_description": "BSEC unavailable — showing raw values only",
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

    def get_status(self) -> dict:
        """Return sensor status info."""
        self._init_sensor()
        status = {
            "sensor": "BME688",
            "available": self._available,
            "bsec_active": self._bsec_active,
        }
        if self._available and self._sensor and self._bsec_active:
            status["bsec_version"] = self._sensor.get_bsec_version()
            status["variant"] = self._sensor.get_variant()
            status["chip_id"] = f"0x{self._sensor.get_chip_id():02X}"
        return status
