"""IMX500 on-chip inference engine with model hot-swap.

Only one model can be active on the IMX500 at a time. This engine
manages model loading, switching, and inference output parsing.
All inference runs entirely on the Sony ISP — zero CPU load.
"""

import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from sensor_head.config import SensorHeadConfig

logger = logging.getLogger("sensor-head.inference")

# ── COCO 80 classes (detection models) ───────────────────────────────

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ── Pose keypoint names ──────────────────────────────────────────────

POSE_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# ── Model registry ───────────────────────────────────────────────────

class ModelType(Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    POSE = "pose"


@dataclass
class ModelSpec:
    name: str
    filename: str
    model_type: ModelType
    input_size: str  # filled at runtime


MODELS = {
    "efficientdet": ModelSpec(
        name="EfficientDet Lite0",
        filename="imx500_network_efficientdet_lite0_pp.rpk",
        model_type=ModelType.DETECTION,
        input_size="320x320",
    ),
    "ssd_mobilenet": ModelSpec(
        name="SSD MobileNetV2 FPN Lite",
        filename="imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        model_type=ModelType.DETECTION,
        input_size="320x320",
    ),
    "nanodet": ModelSpec(
        name="NanoDet Plus",
        filename="imx500_network_nanodet_plus_416x416_pp.rpk",
        model_type=ModelType.DETECTION,
        input_size="416x416",
    ),
    "mobilenet_v2": ModelSpec(
        name="MobileNetV2",
        filename="imx500_network_mobilenet_v2.rpk",
        model_type=ModelType.CLASSIFICATION,
        input_size="224x224",
    ),
    "efficientnet_b0": ModelSpec(
        name="EfficientNetV2-B0",
        filename="imx500_network_efficientnetv2_b0.rpk",
        model_type=ModelType.CLASSIFICATION,
        input_size="224x224",
    ),
    "posenet": ModelSpec(
        name="PoseNet",
        filename="imx500_network_posenet.rpk",
        model_type=ModelType.POSE,
        input_size="481x353",
    ),
    "higherhrnet": ModelSpec(
        name="HigherHRNet",
        filename="imx500_network_higherhrnet_coco.rpk",
        model_type=ModelType.POSE,
        input_size="variable",
    ),
}

# ── ImageNet labels loader ───────────────────────────────────────────

_imagenet_labels: Optional[dict[int, str]] = None


def _load_imagenet_labels() -> dict[int, str]:
    global _imagenet_labels
    if _imagenet_labels is not None:
        return _imagenet_labels

    import json
    import os

    labels_file = "/tmp/imagenet_labels.json"
    if not os.path.exists(labels_file):
        try:
            import urllib.request
            url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            urllib.request.urlretrieve(url, labels_file)
        except Exception:
            logger.warning("Could not download ImageNet labels")
            _imagenet_labels = {}
            return _imagenet_labels

    try:
        with open(labels_file) as f:
            raw = json.load(f)
        _imagenet_labels = {int(k): v[1] for k, v in raw.items()}
    except Exception:
        _imagenet_labels = {}

    return _imagenet_labels


# ── Inference Engine ─────────────────────────────────────────────────

class InferenceEngine:
    """Manages IMX500 model loading and inference with hot-swap."""

    def __init__(self, config: SensorHeadConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._active_model_key: Optional[str] = None
        self._imx500 = None
        self._picam2 = None

    def _load_model(self, model_key: str) -> None:
        """Load a model onto the IMX500 chip. Closes previous if different."""
        if self._active_model_key == model_key and self._picam2 is not None:
            return  # Already loaded

        # Tear down previous
        self._teardown()

        spec = MODELS[model_key]
        model_path = f"{self._config.model_dir}/{spec.filename}"

        from picamera2 import Picamera2
        from picamera2.devices.imx500 import IMX500

        logger.info(f"Loading model '{spec.name}' onto IMX500...")
        t0 = time.time()

        self._imx500 = IMX500(model_path)
        self._picam2 = Picamera2(self._imx500.camera_num)

        config = self._picam2.create_preview_configuration(
            controls={"FrameRate": 30},
            buffer_count=12,
        )
        self._picam2.start(config)
        self._imx500.set_auto_aspect_ratio()

        # Wait for firmware upload + pipeline warmup
        time.sleep(5)

        self._active_model_key = model_key
        logger.info(f"Model '{spec.name}' loaded in {time.time()-t0:.1f}s")

    def _teardown(self) -> None:
        """Stop and close the current camera session."""
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
            self._imx500 = None
            self._active_model_key = None
            time.sleep(0.5)

    def _get_outputs(self, max_attempts: int = 20):
        """Poll for inference outputs from the active model."""
        for _ in range(max_attempts):
            metadata = self._picam2.capture_metadata()
            outputs = self._imx500.get_outputs(metadata)
            if outputs is not None:
                kpi = self._imx500.get_kpi_info(metadata)
                return outputs, metadata, kpi
            time.sleep(0.2)
        return None, None, None

    # ── Public inference methods ─────────────────────────────────────

    def detect_objects(
        self, confidence: float = 0.3, model: str = "efficientdet"
    ) -> dict:
        """Run object detection. Returns detections with labels and boxes."""
        with self._lock:
            self._load_model(model)
            outputs, metadata, kpi = self._get_outputs()

        if outputs is None:
            return {"detections": [], "error": "No inference output"}

        boxes, scores, classes = outputs[0], outputs[1], outputs[2]
        mask = scores > confidence
        result_detections = []

        if mask.any():
            for box, score, cls in zip(
                boxes[mask], scores[mask], classes[mask].astype(int)
            ):
                label = COCO_LABELS[cls] if cls < len(COCO_LABELS) else f"class_{cls}"
                result_detections.append({
                    "label": label,
                    "class_id": int(cls),
                    "confidence": round(float(score), 3),
                    "bbox": {
                        "y_min": round(float(box[0]), 4),
                        "x_min": round(float(box[1]), 4),
                        "y_max": round(float(box[2]), 4),
                        "x_max": round(float(box[3]), 4),
                    },
                })

        result = {
            "model": MODELS[model].name,
            "detections": result_detections,
            "count": len(result_detections),
        }

        if kpi:
            result["performance"] = {
                "dnn_ms": round(kpi[0], 1),
                "dsp_ms": round(kpi[1], 1),
                "total_ms": round(kpi[0] + kpi[1], 1),
            }

        return result

    def classify_scene(
        self, top_k: int = 5, model: str = "mobilenet_v2"
    ) -> dict:
        """Run image classification. Returns top-K predictions."""
        with self._lock:
            self._load_model(model)
            outputs, metadata, kpi = self._get_outputs()

        if outputs is None:
            return {"predictions": [], "error": "No inference output"}

        probs = outputs[0]
        if probs.ndim > 1:
            probs = probs.flatten()

        labels = _load_imagenet_labels()
        top_indices = np.argsort(probs)[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            label = labels.get(int(idx), f"class_{idx}")
            predictions.append({
                "label": label.replace("_", " "),
                "class_id": int(idx),
                "confidence": round(float(probs[idx]), 4),
            })

        result = {
            "model": MODELS[model].name,
            "predictions": predictions,
        }

        if kpi:
            result["performance"] = {
                "dnn_ms": round(kpi[0], 1),
                "dsp_ms": round(kpi[1], 1),
                "total_ms": round(kpi[0] + kpi[1], 1),
            }

        return result

    def estimate_poses(self, model: str = "posenet") -> dict:
        """Run pose estimation. Returns keypoint heatmap peaks."""
        with self._lock:
            self._load_model(model)
            outputs, metadata, kpi = self._get_outputs()

        if outputs is None:
            return {"poses": [], "error": "No inference output"}

        # PoseNet: outputs[0] = heatmaps (H, W, 17), outputs[1] = offsets
        heatmaps = outputs[0]  # shape: (H, W, 17)

        # Extract peak keypoint positions from heatmaps
        poses = []
        if heatmaps.ndim == 3:
            h, w, num_kpts = heatmaps.shape
            keypoints = []

            for kpt_idx in range(min(num_kpts, 17)):
                heatmap = heatmaps[:, :, kpt_idx]
                max_val = float(heatmap.max())
                if max_val > -5.0:  # reasonable threshold for logits
                    peak_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    y_norm = round(float(peak_pos[0]) / h, 4)
                    x_norm = round(float(peak_pos[1]) / w, 4)
                    name = POSE_KEYPOINTS[kpt_idx] if kpt_idx < len(POSE_KEYPOINTS) else f"kpt_{kpt_idx}"
                    keypoints.append({
                        "name": name,
                        "y": y_norm,
                        "x": x_norm,
                        "score": round(max_val, 3),
                    })

            if keypoints:
                # Check if there's a meaningful pose (at least some keypoints with good scores)
                good_kpts = [k for k in keypoints if k["score"] > -2.0]
                poses.append({
                    "keypoints": keypoints,
                    "keypoints_detected": len(good_kpts),
                    "total_keypoints": len(keypoints),
                })

        result = {
            "model": MODELS[model].name,
            "poses": poses,
            "people_detected": len(poses),
            "tensor_shape": list(heatmaps.shape) if heatmaps is not None else None,
        }

        if kpi:
            result["performance"] = {
                "dnn_ms": round(kpi[0], 1),
                "dsp_ms": round(kpi[1], 1),
                "total_ms": round(kpi[0] + kpi[1], 1),
            }

        return result

    def get_available_models(self) -> dict:
        """List all available models and their capabilities."""
        import os
        available = {}
        for key, spec in MODELS.items():
            model_path = f"{self._config.model_dir}/{spec.filename}"
            available[key] = {
                "name": spec.name,
                "type": spec.model_type.value,
                "input_size": spec.input_size,
                "installed": os.path.exists(model_path),
                "active": key == self._active_model_key,
            }
        return available

    def release(self) -> None:
        """Release IMX500 resources."""
        with self._lock:
            self._teardown()
