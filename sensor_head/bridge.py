"""
SensorHead Cloud Bridge — Connects local sensors to ApexAurum Cloud.

Maintains a persistent WebSocket connection to the cloud backend.
Receives commands (capture, read, detect) and returns results.
Pushes telemetry (environment, thermal) at regular intervals.

Usage:
    python -m sensor_head.bridge
    # or with custom config:
    python -m sensor_head.bridge --config /path/to/bridge.json

"The bridge between worlds"
"""

import asyncio
import base64
import io
import json
import logging
import signal
import time
from pathlib import Path

logger = logging.getLogger("sensor-head.bridge")

TELEMETRY_INTERVAL = 30  # seconds
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_DELAY = 300  # 5 minutes
BRIDGE_IMAGE_MAX_DIM = 1024  # Max pixels on longest side for images sent through bridge
BRIDGE_JPEG_QUALITY = 80  # JPEG quality for resized bridge images


class CloudBridge:
    """WebSocket client connecting SensorHead to ApexAurum Cloud."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(
            config_path
            or Path.home() / ".config" / "sensorhead" / "bridge.json"
        )
        self._bridge_config = self._load_config()
        self.ws_url = self._bridge_config.get(
            "ws_url",
            self._bridge_config["cloud_url"].replace("https://", "wss://")
            + "/ws/bridge",
        )
        self.token = self._bridge_config["device_token"]
        self.running = False

        # Lazy-init hardware (same pattern as server.py)
        self._env = None
        self._cameras = None
        self._inference = None
        self._thermal = None
        self._hw_config = None

    def _load_config(self) -> dict:
        """Load bridge configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Bridge config not found: {self.config_path}\n"
                "Run 'python -m sensor_head.pairing' to pair via QR code,\n"
                "or create the file manually with: cloud_url, device_token, device_id"
            )
        data = json.loads(self.config_path.read_text())
        if "device_token" not in data:
            raise ValueError("bridge.json missing 'device_token'")
        return data

    def _get_hw_config(self):
        if self._hw_config is None:
            from sensor_head.config import SensorHeadConfig
            self._hw_config = SensorHeadConfig()
        return self._hw_config

    def _get_env(self):
        if self._env is None:
            from sensor_head.hardware.environment import EnvironmentSensor
            self._env = EnvironmentSensor(self._get_hw_config())
        return self._env

    def _get_cameras(self):
        if self._cameras is None:
            from sensor_head.hardware.cameras import CameraManager
            self._cameras = CameraManager(self._get_hw_config())
        return self._cameras

    def _get_inference(self):
        if self._inference is None:
            from sensor_head.hardware.inference import InferenceEngine
            self._inference = InferenceEngine(self._get_hw_config())
        return self._inference

    def _get_thermal(self):
        if self._thermal is None:
            from sensor_head.hardware.thermal import ThermalCamera
            self._thermal = ThermalCamera(self._get_hw_config())
        return self._thermal

    @staticmethod
    def _resize_jpeg(jpeg_bytes: bytes, max_dim: int = BRIDGE_IMAGE_MAX_DIM) -> bytes:
        """Resize JPEG to fit within max_dim on longest side for cloud transit."""
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(jpeg_bytes))
        w, h = img.size
        if max(w, h) <= max_dim:
            return jpeg_bytes  # Already small enough

        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, PILImage.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=BRIDGE_JPEG_QUALITY)
        logger.debug(f"Resized image {w}x{h} → {new_size[0]}x{new_size[1]} ({len(buf.getvalue())}B)")
        return buf.getvalue()

    async def run(self):
        """Main loop: connect, handle commands, push telemetry."""
        import websockets

        self.running = True
        delay = RECONNECT_DELAY

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._shutdown())

        logger.info(f"SensorHead Bridge starting — connecting to {self.ws_url}")

        while self.running:
            try:
                url = f"{self.ws_url}?token={self.token}"
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=30,
                    max_size=10 * 1024 * 1024,  # 10MB for images
                ) as ws:
                    delay = RECONNECT_DELAY  # Reset on successful connect
                    logger.info("Bridge connected to cloud")

                    # Run command handler and telemetry pusher concurrently
                    await asyncio.gather(
                        self._command_loop(ws),
                        self._telemetry_loop(ws),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.running:
                    break
                logger.warning(
                    f"Bridge disconnected: {e}. Reconnecting in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)

        logger.info("Bridge stopped")

    def _shutdown(self):
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received")
        self.running = False

    async def _command_loop(self, ws):
        """Receive and execute commands from the cloud."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "command":
                # Execute command in background so we can receive more
                asyncio.create_task(self._handle_command(ws, msg))
            elif msg.get("type") == "pong":
                pass

    async def _handle_command(self, ws, msg):
        """Execute a command and send the response."""
        cmd_id = msg["id"]
        action = msg["action"]
        params = msg.get("params", {})

        start = time.time()
        try:
            result, data_type = await self._execute(action, params)
            response = {
                "type": "response",
                "id": cmd_id,
                "success": True,
                "data": result,
                "data_type": data_type,
                "duration_ms": int((time.time() - start) * 1000),
            }
        except Exception as e:
            logger.error(f"Command failed: {action} — {e}")
            response = {
                "type": "response",
                "id": cmd_id,
                "success": False,
                "error": str(e),
                "duration_ms": int((time.time() - start) * 1000),
            }

        await ws.send(json.dumps(response))

    async def _execute(self, action: str, params: dict) -> tuple:
        """
        Route action to hardware and return (result, data_type).

        All hardware calls are blocking — run in thread pool.
        """
        if action == "sense_environment":
            data = await asyncio.to_thread(self._get_env().read)
            return data, "json"

        elif action == "capture_visual":
            jpeg = await asyncio.to_thread(
                self._get_cameras().capture_jpeg, "imx500"
            )
            jpeg = self._resize_jpeg(jpeg)
            return base64.b64encode(jpeg).decode(), "image_base64"

        elif action == "capture_night":
            jpeg = await asyncio.to_thread(
                self._get_cameras().capture_jpeg, "noir"
            )
            jpeg = self._resize_jpeg(jpeg)
            return base64.b64encode(jpeg).decode(), "image_base64"

        elif action == "sense_thermal":
            jpeg = await asyncio.to_thread(
                self._get_thermal().capture_heatmap
            )
            # Thermal heatmaps are already small (320x240), no resize needed
            return base64.b64encode(jpeg).decode(), "image_base64"

        elif action == "read_thermal":
            data = await asyncio.to_thread(self._get_thermal().read)
            return data, "json"

        elif action == "detect_objects":
            confidence = params.get("confidence", 0.3)
            data = await asyncio.to_thread(
                self._get_inference().detect_objects, confidence
            )
            return data, "json"

        elif action == "classify_scene":
            top_k = params.get("top_k", 5)
            data = await asyncio.to_thread(
                self._get_inference().classify_scene, top_k
            )
            return data, "json"

        elif action == "estimate_poses":
            data = await asyncio.to_thread(
                self._get_inference().estimate_poses
            )
            return data, "json"

        elif action == "get_head_status":
            status = {
                "version": "0.4.1",
                "bridge": "connected",
                "sensors": {},
            }
            try:
                env = self._get_env()
                status["sensors"]["environment"] = {
                    "available": True,
                    "bsec_active": getattr(env, "_bsec_active", False),
                }
            except Exception:
                status["sensors"]["environment"] = {"available": False}

            try:
                thermal = self._get_thermal()
                status["sensors"]["thermal"] = {
                    "available": thermal.available,
                }
            except Exception:
                status["sensors"]["thermal"] = {"available": False}

            try:
                cameras = self._get_cameras()
                status["sensors"]["cameras"] = {
                    "imx500": cameras._imx500_num is not None,
                    "noir": cameras._noir_num is not None,
                }
            except Exception:
                status["sensors"]["cameras"] = {"available": False}

            return status, "json"

        elif action == "tts_speak":
            # Voice proxy — call local VoiceServer
            text = params.get("text", "")
            voice = params.get("voice", "system")
            speed = params.get("speed", 1.0)
            result = await self._voice_speak(text, voice, speed)
            return result, "json"

        else:
            raise ValueError(f"Unknown action: {action}")

    async def _voice_speak(
        self, text: str, voice: str, speed: float
    ) -> dict:
        """Proxy TTS command to local VoiceServer."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://192.168.0.104:8765/api/tts",
                    data={"text": text, "voice": voice, "speed": str(speed)},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        return {"spoken": True, "chars": len(text), "voice": voice}
                    else:
                        body = await resp.text()
                        return {"spoken": False, "error": body}
        except Exception as e:
            return {"spoken": False, "error": str(e)}

    async def _telemetry_loop(self, ws):
        """Push environment + thermal readings every TELEMETRY_INTERVAL seconds."""
        while self.running:
            try:
                readings = {}

                # Environment
                try:
                    env_data = await asyncio.to_thread(self._get_env().read)
                    readings.update({
                        "temperature_c": env_data.get("temperature_c"),
                        "humidity_pct": env_data.get("humidity_pct"),
                        "pressure_hpa": env_data.get("pressure_hpa"),
                        "co2_ppm": env_data.get("co2_equivalent_ppm"),
                        "iaq": env_data.get("iaq"),
                        "iaq_accuracy": env_data.get("iaq_accuracy"),
                        "voc_ppm": env_data.get("breath_voc_ppm"),
                    })
                except Exception as e:
                    logger.debug(f"Environment telemetry failed: {e}")

                # Thermal summary
                try:
                    t_data = await asyncio.to_thread(self._get_thermal().read)
                    readings.update({
                        "thermal_min_c": t_data.get("min_c"),
                        "thermal_max_c": t_data.get("max_c"),
                        "thermal_avg_c": t_data.get("avg_c"),
                    })
                except Exception as e:
                    logger.debug(f"Thermal telemetry failed: {e}")

                if readings:
                    telemetry = {
                        "type": "telemetry",
                        "readings": readings,
                        "timestamp": time.time(),
                    }
                    await ws.send(json.dumps(telemetry))
                    logger.debug(
                        f"Telemetry pushed: {readings.get('temperature_c', '?')}°C, "
                        f"IAQ={readings.get('iaq', '?')}"
                    )

            except Exception as e:
                logger.debug(f"Telemetry push failed: {e}")

            await asyncio.sleep(TELEMETRY_INTERVAL)


def main():
    """Entry point for bridge service."""
    import argparse

    parser = argparse.ArgumentParser(description="SensorHead Cloud Bridge")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to bridge.json config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bridge = CloudBridge(config_path=args.config)
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
