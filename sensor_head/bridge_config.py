"""
Bridge Configuration — manages cloud credentials for SensorHead.

Config file: ~/.config/sensorhead/bridge.json
{
    "cloud_url": "https://backend-production-507c.up.railway.app",
    "ws_url": "wss://backend-production-507c.up.railway.app/ws/bridge",
    "device_token": "apex_dev_...",
    "device_id": "uuid",
    "paired_at": "2026-02-12T..."
}
"""

import json
from datetime import datetime
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "sensorhead"
CONFIG_FILE = CONFIG_DIR / "bridge.json"


def save_config(
    cloud_url: str,
    device_token: str,
    device_id: str,
    ws_url: str | None = None,
) -> Path:
    """Save bridge configuration after pairing."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if ws_url is None:
        ws_url = cloud_url.replace("https://", "wss://") + "/ws/bridge"

    config = {
        "cloud_url": cloud_url,
        "ws_url": ws_url,
        "device_token": device_token,
        "device_id": device_id,
        "api_version": "v1",
        "paired_at": datetime.utcnow().isoformat(),
    }

    # Atomic write
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2))
    tmp.rename(CONFIG_FILE)

    # Restrict permissions (token is sensitive)
    CONFIG_FILE.chmod(0o600)

    return CONFIG_FILE


def load_config() -> dict | None:
    """Load bridge configuration. Returns None if not paired."""
    if not CONFIG_FILE.exists():
        return None
    return json.loads(CONFIG_FILE.read_text())


def is_paired() -> bool:
    """Check if this SensorHead is paired with a cloud instance."""
    config = load_config()
    return config is not None and "device_token" in config
