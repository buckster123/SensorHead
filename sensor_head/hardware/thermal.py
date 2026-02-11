"""MLX90640 thermal camera wrapper.

Reads 32x24 (768 pixel) thermal array from the MLX90640 Far-Infrared sensor.
Produces raw temperature arrays and colorized heatmap JPEGs.
"""

import io
import json
import logging
import threading
import time

from PIL import Image as PILImage

from sensor_head.config import SensorHeadConfig

logger = logging.getLogger("sensor-head.thermal")

# ── Ironbow colormap (classic thermal camera look) ────────────────────
# 256-entry LUT: black → blue → magenta → red → orange → yellow → white
_IRONBOW_LUT = None


def _build_ironbow_lut() -> list[tuple[int, int, int]]:
    """Build a 256-entry ironbow colormap."""
    global _IRONBOW_LUT
    if _IRONBOW_LUT is not None:
        return _IRONBOW_LUT

    # Anchor colors for interpolation
    anchors = [
        (0, (0, 0, 0)),         # black (cold)
        (32, (0, 0, 128)),      # dark blue
        (64, (0, 0, 255)),      # blue
        (96, (128, 0, 255)),    # purple
        (128, (255, 0, 128)),   # magenta-red
        (160, (255, 0, 0)),     # red
        (192, (255, 128, 0)),   # orange
        (224, (255, 255, 0)),   # yellow
        (255, (255, 255, 255)), # white (hot)
    ]

    lut = [(0, 0, 0)] * 256
    for i in range(len(anchors) - 1):
        idx0, (r0, g0, b0) = anchors[i]
        idx1, (r1, g1, b1) = anchors[i + 1]
        span = idx1 - idx0
        for j in range(span + 1):
            t = j / span
            idx = idx0 + j
            if idx < 256:
                lut[idx] = (
                    int(r0 + (r1 - r0) * t),
                    int(g0 + (g1 - g0) * t),
                    int(b0 + (b1 - b0) * t),
                )

    _IRONBOW_LUT = lut
    return lut


class ThermalCamera:
    """MLX90640 thermal camera via I2C."""

    ROWS = 24
    COLS = 32
    PIXELS = ROWS * COLS  # 768

    def __init__(self, config: SensorHeadConfig) -> None:
        self._config = config
        self._mlx = None
        self._i2c = None
        self._available = False
        self._lock = threading.Lock()
        self._warmed_up = False

    def _init_sensor(self) -> None:
        """Lazy-initialize the MLX90640."""
        if self._mlx is not None:
            return

        try:
            import board
            import busio
            import adafruit_mlx90640

            self._i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self._mlx = adafruit_mlx90640.MLX90640(self._i2c)
            self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
            self._available = True
            logger.info(
                f"MLX90640 initialized — serial {self._mlx.serial_number}, "
                f"refresh rate 4 Hz"
            )
        except Exception as e:
            logger.warning(f"MLX90640 init failed: {e}")
            self._available = False

    def _warm_up(self) -> None:
        """Discard first 2 frames to flush stale/noisy data."""
        if self._warmed_up:
            return

        logger.info("MLX90640 warming up (discarding 2 frames)...")
        frame = [0.0] * self.PIXELS
        for _ in range(2):
            try:
                self._mlx.getFrame(frame)
            except Exception:
                pass
        self._warmed_up = True
        logger.info("MLX90640 warm-up complete")

    @property
    def available(self) -> bool:
        if self._mlx is None:
            self._init_sensor()
        return self._available

    def read_frame(self) -> list[float]:
        """Read a raw 768-element temperature array (Celsius).

        Outlier values (< -40C or > 300C) are clamped — these are
        typically dead pixels or read errors.
        """
        with self._lock:
            self._init_sensor()
            if not self._available:
                raise RuntimeError("MLX90640 not available")

            self._warm_up()

            frame = [0.0] * self.PIXELS
            self._mlx.getFrame(frame)

            # Clamp outliers (dead pixels / noise)
            for i in range(self.PIXELS):
                if frame[i] < -40.0:
                    frame[i] = -40.0
                elif frame[i] > 300.0:
                    frame[i] = 300.0

            return frame

    def read(self) -> dict:
        """Read thermal frame with stats.

        Returns dict with raw frame, min/max/avg temps, and timestamp.
        """
        if not self.available:
            return {
                "error": True,
                "sensor": "MLX90640",
                "status": "unavailable",
                "reason": "Sensor not detected on I2C bus",
            }

        try:
            t0 = time.time()
            frame = self.read_frame()
            elapsed = time.time() - t0

            return {
                "frame": [round(t, 1) for t in frame],
                "rows": self.ROWS,
                "cols": self.COLS,
                "min_c": round(min(frame), 1),
                "max_c": round(max(frame), 1),
                "avg_c": round(sum(frame) / len(frame), 1),
                "read_time_ms": round(elapsed * 1000, 1),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"MLX90640 read error: {e}")
            return {
                "error": True,
                "sensor": "MLX90640",
                "status": "read_error",
                "reason": str(e),
            }

    def read_json(self) -> str:
        """Read and return as JSON (without raw frame for brevity)."""
        data = self.read()
        if "frame" in data:
            # For JSON output, drop the raw 768-float array
            summary = {k: v for k, v in data.items() if k != "frame"}
            summary["pixels"] = self.PIXELS
            return json.dumps(summary, indent=2)
        return json.dumps(data, indent=2)

    def capture_heatmap(self, width: int = 0, height: int = 0) -> bytes:
        """Capture a thermal frame and render as an ironbow heatmap JPEG.

        Args:
            width: Output width (default from config, typically 320)
            height: Output height (default from config, typically 240)

        Returns:
            JPEG bytes of the colorized heatmap image.
        """
        if not width:
            width = self._config.thermal_upscale_size[0]
        if not height:
            height = self._config.thermal_upscale_size[1]

        frame = self.read_frame()
        lut = _build_ironbow_lut()

        # Normalize frame to 0-255 range
        t_min = min(frame)
        t_max = max(frame)
        t_range = t_max - t_min if t_max > t_min else 1.0

        # Build 32x24 RGB image
        img = PILImage.new("RGB", (self.COLS, self.ROWS))
        pixels = img.load()

        for i in range(self.PIXELS):
            row = i // self.COLS
            col = i % self.COLS
            normalized = int(((frame[i] - t_min) / t_range) * 255)
            normalized = max(0, min(255, normalized))
            pixels[col, row] = lut[normalized]

        # Upscale with nearest-neighbor to keep the thermal "pixel" look
        img = img.resize((width, height), PILImage.NEAREST)

        # Encode as JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def get_status(self) -> dict:
        """Return sensor status info."""
        self._init_sensor()
        status = {
            "sensor": "MLX90640",
            "available": self._available,
            "address": f"0x{self._config.mlx90640_addr:02X}",
            "resolution": f"{self.COLS}x{self.ROWS}",
            "pixels": self.PIXELS,
        }
        if self._available and self._mlx:
            status["serial"] = self._mlx.serial_number
            status["refresh_rate_hz"] = 4
        return status
