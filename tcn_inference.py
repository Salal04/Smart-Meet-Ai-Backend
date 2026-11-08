"""Landmarks-only inference for the extension's exported TCN ONNX model."""
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "tcn_engagement.onnx"
CLASS_NAMES = ["Not-Engaged", "Barely-engaged", "Engaged", "Highly-Engaged"]
_session = None
_lock = threading.Lock()


def _get_session():
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def predict(landmarks):
    """Expect [16,52,3] normalized face points; same x/y normalization as local_tcn.js."""
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.shape != (16, 52, 3):
        raise ValueError(f"landmarks must have shape [16,52,3]; received {list(arr.shape)}")
    if not np.isfinite(arr).all():
        raise ValueError("landmarks must contain only finite numbers")

    xy = arr[:, :, :2]
    xy = xy - xy.mean(axis=1, keepdims=True)
    std = np.sqrt(np.mean(np.square(xy), axis=(1, 2), keepdims=True)) + 1e-6
    tensor = ((xy / std).reshape(1, 16, 104)).astype(np.float32)

    session = _get_session()
    logits = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0]).reshape(-1)
    if logits.size != 4 or not np.isfinite(logits).all():
        raise RuntimeError(f"TCN returned unexpected logits shape/values: {list(logits.shape)}")
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    index = int(probs.argmax())
    return {
        "classIndex": index,
        "label": CLASS_NAMES[index],
        "engagement_Level": index,
        "confidence": round(float(probs[index]), 4),
        "probs": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        "logits": [round(float(x), 5) for x in logits],
    }
