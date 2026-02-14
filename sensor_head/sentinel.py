"""
SensorHead Sentinel — Autonomous motion detection and surveillance.

Thermal frame differencing (MLX90640 at 2Hz) triggers AI confirmation
(IMX500 object detection), then pushes alerts through the Cloud Bridge
WebSocket with optional camera snapshots.

Architecture:
    thermal scan (0.5s) → delta threshold check → AI confirm → alert push

"The guardian that never sleeps"
"""

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("sensor-head.sentinel")


# ─── Configuration ───────────────────────────────────────────────────


@dataclass
class SentinelConfig:
    """Sentinel monitoring parameters. All fields are hot-configurable."""

    # Thermal detection
    thermal_threshold_c: float = 2.0   # Min per-pixel delta to count as "changed"
    min_changed_pixels: int = 10       # Min changed pixels to trigger
    scan_interval: float = 0.5        # Seconds between scans (2Hz default)

    # AI confirmation
    ai_confirm: bool = True            # Run IMX500 detection on trigger
    ai_confidence: float = 0.3         # Min detection confidence
    ai_labels: list = field(           # Labels that elevate from "motion" to named type
        default_factory=lambda: ["person", "cat", "dog", "bird"]
    )

    # Rate limiting
    cooldown_s: float = 30.0           # Min seconds between consecutive alerts
    max_alerts_per_hour: int = 20      # Hard cap per hour

    # Snapshots
    include_snapshot: bool = True      # Attach NoIR camera JPEG to alert
    snapshot_quality: int = 60         # JPEG quality (lower = smaller payload)
    snapshot_max_dim: int = 640        # Max pixel dimension for alert images

    # Schedule (24h "HH:MM" format, empty = always active)
    active_start: str = ""             # e.g. "22:00"
    active_end: str = ""               # e.g. "07:00" (overnight wraps correctly)


# ─── Event ───────────────────────────────────────────────────────────


@dataclass
class SentinelEvent:
    """A single sentinel detection event."""

    event_type: str                    # "motion", "person", "cat", etc.
    timestamp: float
    thermal_delta: float               # Max per-pixel temperature change
    changed_pixels: int                # Pixels exceeding threshold
    thermal_min_c: float
    thermal_max_c: float
    thermal_avg_c: float
    ai_detections: list = field(default_factory=list)
    snapshot_b64: str | None = None


# ─── Built-in Presets ────────────────────────────────────────────────


PRESETS = {
    "default": SentinelConfig(),
    "night_watch": SentinelConfig(
        thermal_threshold_c=1.5,
        min_changed_pixels=5,
        cooldown_s=20.0,
        active_start="22:00",
        active_end="07:00",
    ),
    "away_mode": SentinelConfig(
        thermal_threshold_c=1.0,
        min_changed_pixels=3,
        cooldown_s=15.0,
        ai_labels=["person"],
    ),
    "pet_watch": SentinelConfig(
        thermal_threshold_c=2.5,
        min_changed_pixels=8,
        cooldown_s=60.0,
        ai_labels=["cat", "dog", "bird"],
    ),
}


# ─── Sentinel Loop ──────────────────────────────────────────────────


class SentinelLoop:
    """Autonomous thermal motion detection with AI confirmation."""

    GRID_W = 32
    GRID_H = 24
    PIXELS = GRID_W * GRID_H  # 768

    def __init__(self, hw_config=None):
        self._config = SentinelConfig()
        self._armed = False
        self._running = False
        self._prev_frame: list[float] | None = None
        self._last_alert_time = 0.0
        self._alerts_this_hour = 0
        self._hour_start = 0.0
        self._hw_config = hw_config

        # Lazy hardware refs (same pattern as bridge.py)
        self._thermal = None
        self._inference = None
        self._cameras = None

        # Stats
        self._scan_count = 0
        self._trigger_count = 0
        self._alert_count = 0
        self._last_scan_ms = 0.0

    # ── Hardware access (lazy init) ──

    def _get_hw_config(self):
        if self._hw_config is None:
            from sensor_head.config import SensorHeadConfig
            self._hw_config = SensorHeadConfig()
        return self._hw_config

    def _get_thermal(self):
        if self._thermal is None:
            from sensor_head.hardware.thermal import ThermalCamera
            self._thermal = ThermalCamera(self._get_hw_config())
        return self._thermal

    def _get_inference(self):
        if self._inference is None:
            from sensor_head.hardware.inference import InferenceEngine
            self._inference = InferenceEngine(self._get_hw_config())
        return self._inference

    def _get_cameras(self):
        if self._cameras is None:
            from sensor_head.hardware.cameras import CameraManager
            self._cameras = CameraManager(self._get_hw_config())
        return self._cameras

    # ── Configuration ──

    def configure(self, config_dict: dict):
        """Hot-update configuration from a flat dict."""
        for key, value in config_dict.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        logger.info(f"Sentinel configured: {config_dict}")

    def load_preset(self, name: str):
        """Load a built-in preset by name."""
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
        self._config = SentinelConfig(**asdict(PRESETS[name]))
        logger.info(f"Sentinel preset loaded: {name}")

    def arm(self):
        """Arm the sentinel. Resets baseline frame."""
        self._armed = True
        self._prev_frame = None
        self._alerts_this_hour = 0
        self._hour_start = time.time()
        logger.info("Sentinel ARMED")

    def disarm(self):
        """Disarm the sentinel."""
        self._armed = False
        self._prev_frame = None
        logger.info("Sentinel DISARMED")

    def get_status(self) -> dict:
        """Full status snapshot for cloud queries."""
        return {
            "armed": self._armed,
            "running": self._running,
            "config": asdict(self._config),
            "presets": list(PRESETS.keys()),
            "stats": {
                "scan_count": self._scan_count,
                "trigger_count": self._trigger_count,
                "alert_count": self._alert_count,
                "alerts_this_hour": self._alerts_this_hour,
                "last_scan_ms": round(self._last_scan_ms, 1),
                "last_alert_time": self._last_alert_time,
            },
        }

    # ── Schedule & rate limiting ──

    def _is_in_schedule(self) -> bool:
        """Check if current time falls within the active window."""
        start = self._config.active_start
        end = self._config.active_end
        if not start or not end:
            return True

        now = time.localtime()
        current = now.tm_hour * 60 + now.tm_min

        try:
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
        except (ValueError, AttributeError):
            return True

        start_m = sh * 60 + sm
        end_m = eh * 60 + em

        if start_m <= end_m:
            return start_m <= current < end_m
        else:
            # Overnight wrap (e.g. 22:00→07:00)
            return current >= start_m or current < end_m

    def _check_rate_limit(self) -> bool:
        """Return True if an alert is allowed right now."""
        now = time.time()

        # Reset hourly counter
        if now - self._hour_start > 3600:
            self._alerts_this_hour = 0
            self._hour_start = now

        if now - self._last_alert_time < self._config.cooldown_s:
            return False

        if self._alerts_this_hour >= self._config.max_alerts_per_hour:
            return False

        return True

    # ── Detection logic ──

    def _compute_delta(
        self, current: list[float], previous: list[float]
    ) -> tuple[float, int]:
        """Per-pixel thermal delta. Returns (max_delta, changed_count)."""
        max_delta = 0.0
        changed = 0
        threshold = self._config.thermal_threshold_c

        for i in range(self.PIXELS):
            delta = abs(current[i] - previous[i])
            if delta > max_delta:
                max_delta = delta
            if delta > threshold:
                changed += 1

        return max_delta, changed

    async def _capture_snapshot(self) -> str | None:
        """Capture NoIR camera JPEG, resize for alert efficiency."""
        try:
            from PIL import Image as PILImage

            jpeg = await asyncio.to_thread(
                self._get_cameras().capture_jpeg, "noir"
            )
            img = PILImage.open(io.BytesIO(jpeg))
            w, h = img.size
            max_dim = self._config.snapshot_max_dim
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize(
                    (int(w * scale), int(h * scale)), PILImage.LANCZOS
                )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._config.snapshot_quality)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Sentinel snapshot failed: {e}")
            return None

    async def _ai_confirm(self) -> list[dict]:
        """Run IMX500 object detection for trigger classification.

        Releases the inference engine after use so the IMX500 camera
        is available for manual captures and other operations.
        """
        try:
            inference = self._get_inference()
            result = await asyncio.to_thread(
                inference.detect_objects,
                self._config.ai_confidence,
            )
            # Release IMX500 so it's available for manual captures
            await asyncio.to_thread(inference.release)
            return result.get("detections", [])
        except Exception as e:
            logger.warning(f"Sentinel AI confirm failed: {e}")
            # Ensure release even on failure
            if self._inference is not None:
                try:
                    await asyncio.to_thread(self._inference.release)
                except Exception:
                    pass
            return []

    async def scan_once(self) -> SentinelEvent | None:
        """Single scan cycle. Returns event on detection, None otherwise."""
        if not self._armed or not self._is_in_schedule():
            return None

        self._scan_count += 1
        t0 = time.time()

        # Read thermal frame (blocking → thread pool)
        try:
            frame = await asyncio.to_thread(self._get_thermal().read_frame)
        except Exception as e:
            logger.debug(f"Sentinel thermal read failed: {e}")
            return None

        self._last_scan_ms = (time.time() - t0) * 1000

        # First frame sets baseline
        if self._prev_frame is None:
            self._prev_frame = frame
            logger.debug("Sentinel baseline frame captured")
            return None

        # Compute delta
        max_delta, changed_pixels = self._compute_delta(frame, self._prev_frame)
        self._prev_frame = frame

        # Below threshold — no trigger
        if changed_pixels < self._config.min_changed_pixels:
            return None

        self._trigger_count += 1
        logger.info(
            f"Sentinel trigger: {changed_pixels} pixels, "
            f"max delta {max_delta:.1f}°C"
        )

        # Rate limit
        if not self._check_rate_limit():
            logger.debug("Sentinel alert suppressed (cooldown/rate limit)")
            return None

        # Thermal stats
        t_min = min(frame)
        t_max = max(frame)
        t_avg = sum(frame) / len(frame)

        # AI confirmation
        ai_detections = []
        event_type = "motion"
        if self._config.ai_confirm:
            ai_detections = await self._ai_confirm()
            target_labels = [l.lower() for l in self._config.ai_labels]
            for det in ai_detections:
                label = det.get("label", "").lower()
                if label in target_labels:
                    event_type = label
                    break

        # Snapshot
        snapshot = None
        if self._config.include_snapshot:
            snapshot = await self._capture_snapshot()

        # Build event
        event = SentinelEvent(
            event_type=event_type,
            timestamp=time.time(),
            thermal_delta=round(max_delta, 2),
            changed_pixels=changed_pixels,
            thermal_min_c=round(t_min, 1),
            thermal_max_c=round(t_max, 1),
            thermal_avg_c=round(t_avg, 1),
            ai_detections=ai_detections,
            snapshot_b64=snapshot,
        )

        self._alert_count += 1
        self._alerts_this_hour += 1
        self._last_alert_time = time.time()

        return event

    # ── Main loop ──

    async def run(self, alert_callback) -> None:
        """Main sentinel loop.

        Scans at configured interval, calls alert_callback(event)
        on detection. The callback should push the event through
        the bridge WebSocket.
        """
        self._running = True
        logger.info("Sentinel loop started")

        try:
            while self._running:
                if self._armed:
                    event = await self.scan_once()
                    if event:
                        try:
                            await alert_callback(event)
                        except Exception as e:
                            logger.error(f"Sentinel alert push failed: {e}")

                await asyncio.sleep(self._config.scan_interval)
        finally:
            self._running = False
            logger.info("Sentinel loop stopped")

    def stop(self):
        """Signal the loop to stop."""
        self._running = False
