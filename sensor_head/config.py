"""Hardware configuration for SensorHead."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> str:
    """BSEC state and other persistent files.

    Prefer SENSORHEAD_DATA_DIR. Otherwise use <repo>/data so a checkout
    running as apex1 (or anyone else) does not try to write the old
    hailo absolute path.
    """
    override = os.environ.get("SENSORHEAD_DATA_DIR")
    if override:
        return override
    return str(Path(__file__).resolve().parent.parent / "data")


@dataclass
class SensorHeadConfig:
    """All hardware addresses, pin mappings, and defaults."""

    # I2C bus
    i2c_bus: int = 1

    # MLX90640 thermal camera
    mlx90640_addr: int = 0x33

    # BME688 environmental sensor (0x76 primary, 0x77 fallback)
    bme688_addr: int = 0x76
    bme688_addr_alt: int = 0x77

    # PCA9685 servo controller (Arducam B0283)
    pca9685_addr: int = 0x40
    pan_channel: int = 0
    tilt_channel: int = 1
    pan_range: tuple[int, int] = (0, 180)
    tilt_range: tuple[int, int] = (0, 180)
    pan_default: int = 90
    tilt_default: int = 90

    # Camera CSI mapping (None = auto-detect by model name)
    imx500_camera_num: int | None = None
    noir_camera_num: int | None = None

    # Camera rotation (both mounted upside-down in sensor head bracket)
    camera_rotation_deg: int = 180

    # Per-camera capture resolutions
    imx500_resolution: tuple[int, int] = (2028, 1520)
    noir_resolution: tuple[int, int] = (2304, 1296)
    jpeg_quality: int = 85

    # Auto-exposure settle time (seconds) after camera start
    ae_settle_time: float = 1.5

    # IMX500 inference
    model_dir: str = "/usr/share/imx500-models"
    default_model: str = "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

    # Thermal heatmap
    thermal_upscale_size: tuple[int, int] = (320, 240)

    # Data directory for persistent state (BSEC calibration, etc.)
    data_dir: str = field(default_factory=default_data_dir)

    # BSEC state save interval (seconds)
    bsec_save_interval: int = 300  # 5 minutes
