"""SensorHead Dashboard — Cyberpunk web UI for camera feeds and sensors.

Run: python -m sensor_head.dashboard [--port 8080]
"""

import asyncio
import argparse
import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sensor_head.config import SensorHeadConfig
from sensor_head.hardware.cameras import CameraManager
from sensor_head.hardware.environment import EnvironmentSensor
from sensor_head.hardware.inference import InferenceEngine
from sensor_head.hardware.thermal import ThermalCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sensor-head.dashboard")

config = SensorHeadConfig()
camera_mgr = CameraManager(config)
env_sensor = EnvironmentSensor(config)
inference = InferenceEngine(config)
thermal = ThermalCamera(config)

# Track whether the inference engine currently holds the IMX500 camera
_ai_active = False

app = FastAPI(title="SensorHead Dashboard", version="0.4.1")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/capture/visual")
async def api_capture_visual():
    """JPEG from IMX500 (left eye). Returns 409 if AI inference holds the camera."""
    if _ai_active:
        return JSONResponse(
            {"error": "IMX500 busy — AI inference active", "ai_active": True},
            status_code=409,
        )
    try:
        jpeg = await asyncio.to_thread(camera_mgr.capture_jpeg, "imx500")
        return Response(content=jpeg, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/capture/night")
async def api_capture_night():
    """JPEG from NoIR Wide (right eye)."""
    try:
        jpeg = await asyncio.to_thread(camera_mgr.capture_jpeg, "noir")
        return Response(content=jpeg, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/environment")
async def api_environment():
    """BME688 sensor readings."""
    try:
        result = await asyncio.to_thread(env_sensor.read)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/status")
async def api_status():
    """Full sensor head status."""
    from sensor_head.hardware.i2c_bus import I2CBus

    status = {"server": "SensorHead v0.4.0", "timestamp": time.time()}

    # I2C
    try:
        bus = I2CBus(config.i2c_bus)
        devices = await asyncio.to_thread(bus.scan)
        known = {
            0x33: "MLX90640 (thermal)",
            0x40: "PCA9685 (pan-tilt)",
            0x76: "BME688 (environment)",
            0x77: "BME688 (environment, alt)",
        }
        status["i2c_devices"] = {
            f"0x{a:02X}": known.get(a, "unknown") for a in devices
        }
    except Exception as e:
        status["i2c_error"] = str(e)

    # BME688
    try:
        if env_sensor.available:
            status["environment"] = await asyncio.to_thread(env_sensor.read)
        else:
            status["environment"] = {"status": "not_detected"}
    except Exception as e:
        status["environment"] = {"error": str(e)}

    # Thermal
    try:
        status["thermal"] = await asyncio.to_thread(thermal.get_status)
    except Exception as e:
        status["thermal"] = {"error": str(e)}

    # Cameras
    try:
        status["cameras"] = await asyncio.to_thread(camera_mgr.get_status)
    except Exception as e:
        status["cameras"] = {"error": str(e)}

    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    status.setdefault("system", {})["mem_total_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    status.setdefault("system", {})["mem_available_mb"] = int(line.split()[1]) // 1024
    except Exception:
        pass

    return JSONResponse(status)


@app.get("/api/thermal/heatmap")
async def api_thermal_heatmap():
    """Thermal heatmap JPEG from MLX90640."""
    try:
        jpeg = await asyncio.to_thread(thermal.capture_heatmap)
        return Response(content=jpeg, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/thermal/data")
async def api_thermal_data():
    """Raw thermal data with stats from MLX90640."""
    try:
        result = await asyncio.to_thread(thermal.read)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/environment/save-state")
async def api_save_bsec_state():
    """Manually save BSEC calibration state to disk."""
    try:
        saved = await asyncio.to_thread(env_sensor.save_state)
        if saved:
            return JSONResponse({"status": "saved", "file": str(env_sensor._state_file)})
        return JSONResponse({"status": "skipped", "reason": "BSEC not active"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.on_event("shutdown")
async def shutdown_save_state():
    """Save BSEC state before dashboard exits."""
    logger.info("Dashboard shutting down — saving BSEC state...")
    env_sensor.save_state()


@app.get("/api/detect")
async def api_detect(confidence: float = 0.3):
    """Object detection via IMX500 EfficientDet."""
    global _ai_active
    _ai_active = True
    try:
        result = await asyncio.to_thread(inference.detect_objects, confidence)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        await asyncio.to_thread(inference.release)
        _ai_active = False


@app.get("/api/classify")
async def api_classify(top_k: int = 5):
    """Scene classification via IMX500 MobileNetV2."""
    global _ai_active
    _ai_active = True
    try:
        result = await asyncio.to_thread(inference.classify_scene, top_k)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        await asyncio.to_thread(inference.release)
        _ai_active = False


@app.get("/api/pose")
async def api_pose():
    """Pose estimation via IMX500 PoseNet."""
    global _ai_active
    _ai_active = True
    try:
        result = await asyncio.to_thread(inference.estimate_poses)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        await asyncio.to_thread(inference.release)
        _ai_active = False


@app.get("/api/models")
async def api_models():
    """List available AI models."""
    try:
        result = await asyncio.to_thread(inference.get_available_models)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="SensorHead Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logger.info(f"SensorHead Dashboard starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
