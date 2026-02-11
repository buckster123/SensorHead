"""SensorHead MCP Server — Physical senses for Claude Code.

Exposes Raspberry Pi 5 sensor hardware as MCP tools:
- BME688 environmental sensor (temperature, humidity, pressure, gas)
- MLX90640 thermal camera (32x24 thermal array → heatmap)
- Arducam B0283 pan-tilt (servo control)
- RPi AI Camera IMX500 (visual + on-chip inference)
- Camera Module 3 NoIR Wide (night vision)
"""

import asyncio
import json
import logging
import time

from mcp.server.fastmcp import FastMCP, Image

from sensor_head.config import SensorHeadConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sensor-head")

# Global state — lazy-initialized on first use
_config: SensorHeadConfig | None = None
_env_sensor = None
_camera_mgr = None
_inference = None
_thermal = None


def _get_config() -> SensorHeadConfig:
    global _config
    if _config is None:
        _config = SensorHeadConfig()
    return _config


def _get_env_sensor():
    """Lazy-init the environment sensor."""
    global _env_sensor
    if _env_sensor is None:
        from sensor_head.hardware.environment import EnvironmentSensor
        _env_sensor = EnvironmentSensor(_get_config())
    return _env_sensor


def _get_camera_mgr():
    """Lazy-init the camera manager."""
    global _camera_mgr
    if _camera_mgr is None:
        from sensor_head.hardware.cameras import CameraManager
        _camera_mgr = CameraManager(_get_config())
    return _camera_mgr


def _get_inference():
    """Lazy-init the inference engine."""
    global _inference
    if _inference is None:
        from sensor_head.hardware.inference import InferenceEngine
        _inference = InferenceEngine(_get_config())
    return _inference


def _get_thermal():
    """Lazy-init the thermal camera."""
    global _thermal
    if _thermal is None:
        from sensor_head.hardware.thermal import ThermalCamera
        _thermal = ThermalCamera(_get_config())
    return _thermal


# Create the FastMCP app
mcp = FastMCP("SensorHead")


# ── Environment ──────────────────────────────────────────────────────

@mcp.tool()
async def sense_environment() -> str:
    """Read all channels from the BME688 environmental sensor (nose).

    Returns JSON with:
    - temperature_c: Ambient temperature in Celsius
    - humidity_pct: Relative humidity percentage
    - pressure_hpa: Barometric pressure in hectopascals
    - gas_resistance_ohm: Gas sensor resistance (higher = cleaner air)
    - altitude_m: Estimated altitude from pressure
    """
    sensor = _get_env_sensor()
    result = await asyncio.to_thread(sensor.read)
    return json.dumps(result, indent=2)


# ── Thermal ──────────────────────────────────────────────────────────

@mcp.tool()
async def sense_thermal() -> Image:
    """Capture a thermal heatmap from the MLX90640 Far-Infrared camera.

    Returns a colorized ironbow heatmap JPEG (320x240) showing the
    32x24 thermal array. Colors range from black/blue (cold) through
    red/orange (warm) to yellow/white (hot).

    The MLX90640 detects temperatures from -40C to 300C with ~0.1C
    resolution — great for detecting people, electronics, heat sources.
    """
    thermal = _get_thermal()
    jpeg_bytes = await asyncio.to_thread(thermal.capture_heatmap)
    return Image(data=jpeg_bytes, format="jpeg")


@mcp.tool()
async def read_thermal() -> str:
    """Read raw thermal data from the MLX90640 sensor.

    Returns JSON with temperature statistics (min, max, avg in Celsius),
    pixel count, and read timing. Use sense_thermal for the visual heatmap.
    """
    thermal = _get_thermal()
    result = await asyncio.to_thread(thermal.read_json)
    return result


# ── Vision ───────────────────────────────────────────────────────────

@mcp.tool()
async def capture_visual() -> Image:
    """Capture a photo from the IMX500 AI Camera (left eye, daylight).

    Returns a JPEG image at 2028x1520. Best for well-lit scenes.
    The IMX500 has on-chip AI for future object detection.
    """
    mgr = _get_camera_mgr()
    jpeg_bytes = await asyncio.to_thread(mgr.capture_jpeg, "imx500")
    return Image(data=jpeg_bytes, format="jpeg")


@mcp.tool()
async def capture_night() -> Image:
    """Capture a photo from the IMX708 Wide NoIR Camera (right eye, night vision).

    Returns a JPEG image at 2304x1296. Wide-angle lens with no IR filter —
    excellent in low light and dark conditions. Wider FOV than the IMX500.
    """
    mgr = _get_camera_mgr()
    jpeg_bytes = await asyncio.to_thread(mgr.capture_jpeg, "noir")
    return Image(data=jpeg_bytes, format="jpeg")


# ── AI Vision (IMX500 on-chip inference) ─────────────────────────────

@mcp.tool()
async def detect_objects(confidence: float = 0.3) -> str:
    """Detect objects in the scene using the IMX500 AI Camera's on-chip neural network.

    Runs EfficientDet Lite0 entirely on the Sony ISP — zero CPU load.
    Detects 80 COCO object classes: person, chair, laptop, cup, bottle, etc.

    Args:
        confidence: Minimum confidence threshold (0.0-1.0, default 0.3)

    Returns JSON with:
    - detections: List of {label, class_id, confidence, bbox}
    - count: Number of objects detected
    - performance: DNN and DSP timing in milliseconds
    """
    engine = _get_inference()
    result = await asyncio.to_thread(engine.detect_objects, confidence)
    return json.dumps(result, indent=2)


@mcp.tool()
async def classify_scene(top_k: int = 5) -> str:
    """Classify what the IMX500 AI Camera is looking at using on-chip MobileNetV2.

    Returns the top-K most likely ImageNet categories (1000 classes).
    Blazing fast at ~8ms per inference — runs entirely on the sensor chip.

    Args:
        top_k: Number of top predictions to return (default 5)

    Returns JSON with:
    - predictions: List of {label, class_id, confidence}
    - performance: DNN and DSP timing in milliseconds
    """
    engine = _get_inference()
    result = await asyncio.to_thread(engine.classify_scene, top_k)
    return json.dumps(result, indent=2)


@mcp.tool()
async def estimate_poses() -> str:
    """Detect human body poses using IMX500 on-chip PoseNet.

    Returns 17 body keypoints (nose, eyes, ears, shoulders, elbows,
    wrists, hips, knees, ankles) with normalized coordinates and scores.

    Returns JSON with:
    - poses: List of detected people with keypoints
    - people_detected: Number of people found
    - performance: DNN and DSP timing in milliseconds
    """
    engine = _get_inference()
    result = await asyncio.to_thread(engine.estimate_poses)
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_ai_models() -> str:
    """List all available AI models on the IMX500 sensor chip.

    Shows which models are installed, their type (detection/classification/pose),
    and which model is currently active on the chip.

    Returns JSON with model details for each available model.
    """
    engine = _get_inference()
    result = await asyncio.to_thread(engine.get_available_models)
    return json.dumps(result, indent=2)


# ── Status ───────────────────────────────────────────────────────────

@mcp.tool()
async def get_head_status() -> str:
    """Get the current status of all sensor head components.

    Returns JSON with:
    - i2c_devices: Detected devices on the I2C bus with known identities
    - environment: BME688 sensor availability and last reading
    - thermal: MLX90640 connection status
    - pan_tilt: Servo controller status
    - cameras: CSI camera detection and mapping status
    - system: Memory usage and uptime
    """
    config = _get_config()
    status = {
        "server": "SensorHead v0.4.1",
        "timestamp": time.time(),
    }

    # I2C bus scan
    try:
        from sensor_head.hardware.i2c_bus import I2CBus
        bus = I2CBus(config.i2c_bus)
        devices = await asyncio.to_thread(bus.scan)

        known = {
            0x33: "MLX90640 (thermal)",
            0x40: "PCA9685 (pan-tilt servos)",
            0x76: "BME688 (environment)",
            0x77: "BME688 (environment, alt addr)",
        }

        status["i2c_devices"] = {
            f"0x{addr:02X}": known.get(addr, "unknown")
            for addr in devices
        }
        status["i2c_device_count"] = len(devices)
    except Exception as e:
        status["i2c_error"] = str(e)

    # BME688 status
    try:
        sensor = _get_env_sensor()
        if sensor.available:
            reading = await asyncio.to_thread(sensor.read)
            status["environment"] = {
                "status": "connected",
                "reading": reading,
            }
        else:
            status["environment"] = {"status": "not_detected"}
    except Exception as e:
        status["environment"] = {"status": "error", "reason": str(e)}

    # Thermal status
    try:
        thermal = _get_thermal()
        if thermal.available:
            status["thermal"] = await asyncio.to_thread(thermal.get_status)
        else:
            status["thermal"] = {"status": "not_detected", "address": f"0x{config.mlx90640_addr:02X}"}
    except Exception as e:
        status["thermal"] = {"status": "error", "reason": str(e)}

    # Pan-tilt status
    if "i2c_devices" in status:
        pca_addr = f"0x{config.pca9685_addr:02X}"
        status["pan_tilt"] = {
            "status": "detected" if pca_addr in status["i2c_devices"] else "not_detected",
            "address": pca_addr,
        }

    # Camera status
    try:
        mgr = _get_camera_mgr()
        status["cameras"] = await asyncio.to_thread(mgr.get_status)
    except Exception as e:
        status["cameras"] = {"status": "error", "reason": str(e)}

    # System memory
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                status.setdefault("system", {})["mem_total_mb"] = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                status.setdefault("system", {})["mem_available_mb"] = int(line.split()[1]) // 1024
    except Exception:
        pass

    return json.dumps(status, indent=2)


def main():
    """Entry point for the MCP server."""
    global _config
    _config = SensorHeadConfig()
    logger.info("SensorHead MCP server starting (v0.4.1)")
    mcp.run()


if __name__ == "__main__":
    main()
