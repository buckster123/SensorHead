"""Dual-camera wrapper for IMX500 and IMX708 Wide NoIR.

Manages Picamera2 lifecycle with lazy init and capture-release pattern
to avoid holding ~50MB DMA buffers per camera when idle.
"""

import io
import logging
import threading
import time

from PIL import Image

from sensor_head.config import SensorHeadConfig

logger = logging.getLogger("sensor-head.cameras")

# Sensor model strings as reported by libcamera
_IMX500_MODEL = "imx500"
_IMX708_MODEL = "imx708_wide_noir"


class CameraManager:
    """Manages both CSI cameras with capture-and-release pattern."""

    def __init__(self, config: SensorHeadConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._camera_map: dict[str, int] | None = None

    def _discover_cameras(self) -> dict[str, int]:
        """Map sensor model names to camera numbers."""
        if self._camera_map is not None:
            return self._camera_map

        from picamera2 import Picamera2

        info_list = Picamera2.global_camera_info()
        mapping = {}

        for cam_info in info_list:
            num = cam_info.get("Num")
            model = cam_info.get("Model", "")

            if _IMX500_MODEL in model:
                mapping["imx500"] = num
                logger.info(f"IMX500 AI Camera found at camera {num}")
            elif _IMX708_MODEL in model or "imx708" in model:
                mapping["noir"] = num
                logger.info(f"IMX708 Wide NoIR found at camera {num}")
            else:
                logger.info(f"Unknown camera model '{model}' at camera {num}")

        # Allow config overrides
        if self._config.imx500_camera_num is not None:
            mapping["imx500"] = self._config.imx500_camera_num
        if self._config.noir_camera_num is not None:
            mapping["noir"] = self._config.noir_camera_num

        self._camera_map = mapping
        return mapping

    def _capture(self, camera_num: int, resolution: tuple[int, int]) -> Image.Image:
        """Open camera, capture a still, close camera. Returns PIL Image."""
        from picamera2 import Picamera2
        from libcamera import Transform

        rot = self._config.camera_rotation_deg
        hflip = rot in (180, 270)
        vflip = rot in (180, 90)

        picam2 = Picamera2(camera_num)
        try:
            still_config = picam2.create_still_configuration(
                main={"size": resolution, "format": "RGB888"},
                transform=Transform(hflip=hflip, vflip=vflip),
            )
            picam2.configure(still_config)
            picam2.start()
            time.sleep(self._config.ae_settle_time)
            array = picam2.capture_array("main")
            return Image.fromarray(array)
        finally:
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.close()
            except Exception:
                pass

    def capture_visual(self) -> Image.Image:
        """Capture from the IMX500 AI Camera (left eye, daylight).

        Returns a PIL Image at the configured IMX500 resolution.
        Raises RuntimeError if the camera is not available.
        """
        with self._lock:
            mapping = self._discover_cameras()
            if "imx500" not in mapping:
                raise RuntimeError("IMX500 AI Camera not detected")
            return self._capture(
                mapping["imx500"],
                self._config.imx500_resolution,
            )

    def capture_night(self) -> Image.Image:
        """Capture from the IMX708 Wide NoIR (right eye, night vision).

        Returns a PIL Image at the configured NoIR resolution.
        Raises RuntimeError if the camera is not available.
        """
        with self._lock:
            mapping = self._discover_cameras()
            if "noir" not in mapping:
                raise RuntimeError("IMX708 Wide NoIR Camera not detected")
            return self._capture(
                mapping["noir"],
                self._config.noir_resolution,
            )

    def capture_jpeg(self, camera_key: str) -> bytes:
        """Capture and return JPEG bytes for the specified camera.

        Args:
            camera_key: "imx500" or "noir"

        Returns:
            JPEG-encoded image bytes
        """
        if camera_key == "imx500":
            img = self.capture_visual()
        elif camera_key == "noir":
            img = self.capture_night()
        else:
            raise ValueError(f"Unknown camera key: {camera_key}")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._config.jpeg_quality)
        return buf.getvalue()

    def get_status(self) -> dict:
        """Return status of both cameras."""
        from picamera2 import Picamera2

        info_list = Picamera2.global_camera_info()
        mapping = self._discover_cameras()

        return {
            "cameras_detected": len(info_list),
            "camera_info": [
                {
                    "num": c.get("Num"),
                    "model": c.get("Model"),
                    "location": c.get("Location"),
                }
                for c in info_list
            ],
            "imx500_available": "imx500" in mapping,
            "imx500_camera_num": mapping.get("imx500"),
            "noir_available": "noir" in mapping,
            "noir_camera_num": mapping.get("noir"),
            "rotation_deg": self._config.camera_rotation_deg,
        }
