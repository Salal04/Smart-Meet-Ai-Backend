"""
landmark_extractor.py -- video se face landmarks khud nikalta hai (MediaPipe FaceLandmarker, Tasks API).

Use:
    from landmark_extractor import extract_landmarks, ensure_landmarks

    lm = extract_landmarks("video.mp4", logs=True)                 # np.ndarray (T, 468, 3)
    lm = extract_landmarks("video.mp4", save_path="video.npy")     # npy bhi save
    lm = extract_landmarks("video.mp4", required_only=True)        # (T, 52, 3) sirf model wale points

    path = ensure_landmarks("video.mp4")   # npy hai to wahi path, nahi to bana kar path return

Notes:
  * Dependencies (numpy, opencv-python-headless, mediapipe) na hon to pehli dafa khud pip install hoti hain.
  * Model file (face_landmarker.task, ~3.6MB) pehli dafa khud download hoti hai aur cache ho jati hai.
  * Latest mediapipe (0.10.3x) me purana `mp.solutions.face_mesh` nahi raha, isliye Tasks API use hui hai.
  * Default output (T, 468, 3) hai -- inference.py yehi format expect karta hai aur wahi se 52 points chunta hai.
    required_only=True wali array inference.py ko mat do (wo 468-wali array par index karta hai).
  * x, y normalized [0,1] hain, z mediapipe ki relative depth (MediaPipe ka native format).
"""
import importlib
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Optional

__all__ = ["extract_landmarks", "ensure_landmarks", "ensure_dependencies",
           "IMPORTANT_LANDMARKS", "LandmarkError"]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")

# inference.py wali IMPORTANT_LANDMARKS list ke bilkul barabar (52 points)
_LEFT_EYE = [33, 133, 159, 145, 153, 154, 155, 160, 161]
_RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 387, 388]
_MOUTH = [13, 14, 78, 308, 61, 291, 81, 82, 87, 312]
_NOSE = [1, 2, 98, 327, 168]
_JAW = [152, 13, 14, 234, 454, 454, 454]
_L_BROW = [70, 63, 105, 66, 107, 55]
_R_BROW = [336, 296, 334, 293, 300, 283]
IMPORTANT_LANDMARKS = _LEFT_EYE + _RIGHT_EYE + _MOUTH + _NOSE + _JAW + _L_BROW + _R_BROW

N_MESH = 468  # purane FaceMesh jaisa; Tasks API 478 deta hai (last 10 iris), unhein hata dete hain

# import name -> pip package
_REQUIRED = {"numpy": "numpy", "cv2": "opencv-python-headless", "mediapipe": "mediapipe"}


class LandmarkError(RuntimeError):
    """Landmarks nikalne me koi masla."""


def _log(msg, logs=True, level="INFO"):
    if logs:
        print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
def ensure_dependencies(auto_install: bool = True, logs: bool = False):
    """Missing packages ko pip se install karta hai (agar auto_install=True)."""
    missing = []
    for mod, pkg in _REQUIRED.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if not missing:
        return
    if not auto_install:
        raise LandmarkError(f"Ye packages install karo: pip install {' '.join(missing)}")

    _log(f"Install ho rahe hain: {missing}", logs)
    cmd = [sys.executable, "-m", "pip", "install", "-q", *missing]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        # system-managed Python (Ubuntu 24 waghera) ke liye
        try:
            subprocess.check_call(cmd + ["--break-system-packages"])
        except subprocess.CalledProcessError as e:
            raise LandmarkError(f"pip install fail hui: {' '.join(missing)}. Manually chalao: "
                                f"pip install {' '.join(missing)}") from e
    importlib.invalidate_caches()
    for mod in _REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception as e:
            raise LandmarkError(f"'{mod}' install ke baad bhi import nahi hua: {e}. "
                                f"Colab/notebook ho to runtime restart karo.") from e
    _log("Dependencies theek hain", logs)


# ----------------------------------------------------------------------------
# Model file
# ----------------------------------------------------------------------------
def _resolve_model(model_path: Optional[str], logs: bool) -> str:
    model_path = model_path or os.environ.get("LANDMARK_MODEL_PATH")
    if model_path:
        if not os.path.isfile(model_path):
            raise LandmarkError(f"Model file nahi mili: {model_path}")
        return model_path

    cache_dir = os.environ.get("LANDMARK_MODEL_DIR",
                               os.path.join(os.path.expanduser("~"), ".cache", "face_landmarker"))
    dest = os.path.join(cache_dir, "face_landmarker.task")
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        _log(f"Model cache se: {dest}", logs)
        return dest

    _log(f"Model download ho raha hai -> {dest}", logs)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = dest + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, dest)
    except Exception as e:
        raise LandmarkError(
            f"Model download nahi hua ({e}). Internet check karo, ya file khud download karke "
            f"model_path=... / env LANDMARK_MODEL_PATH me do:\n{MODEL_URL}") from e
    return dest


# ----------------------------------------------------------------------------
# Main: video -> landmarks
# ----------------------------------------------------------------------------
def extract_landmarks(
    video_path: str,
    save_path: Optional[str] = None,
    required_only: bool = False,
    fill_missing: bool = True,
    min_detect_ratio: float = 0.0,
    model_path: Optional[str] = None,
    auto_install: bool = True,
    logs: bool = False,
):
    """
    video_path       : video file
    save_path        : de to .npy save hoti hai
    required_only    : True -> (T,52,3) sirf model wale points; False -> (T,468,3) full mesh
    fill_missing     : jis frame me chehra na mile wahan pichle (ya pehle) mile hue frame ki values
                       copy hoti hain. False -> zeros.
    min_detect_ratio : itne se kam frames me chehra mile to LandmarkError (0 = sirf tab jab ek bhi na mile)
    Return           : np.ndarray float32, har video frame ke liye ek row (T = total frames)
    """
    t0 = time.time()
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video nahi mili: {video_path}")

    ensure_dependencies(auto_install=auto_install, logs=logs)
    import cv2
    import numpy as np
    import mediapipe as mp

    model_file = _resolve_model(model_path, logs)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise LandmarkError(f"Video khul nahi saki (corrupt ya unsupported codec): {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-3 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    _log(f"Video: {video_path} | fps={fps:.1f} | frames~{total}", logs)

    try:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_file),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        cap.release()
        raise LandmarkError(f"FaceLandmarker create nahi hua: {e}") from e

    rows, found = [], []
    last_ts = -1
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = max(int(i * 1000.0 / fps), last_ts + 1)  # strictly increasing
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            try:
                res = landmarker.detect_for_video(image, ts)
                face = res.face_landmarks[0] if res.face_landmarks else None
            except Exception as e:
                _log(f"Frame {i}: detect error ({e})", logs, "WARN")
                face = None

            if face is not None:
                rows.append(np.array([[p.x, p.y, p.z] for p in face[:N_MESH]], dtype=np.float32))
                found.append(True)
            else:
                rows.append(None)
                found.append(False)
            i += 1
            if logs and i % 100 == 0:
                _log(f"{i} frames process huay", logs)
    finally:
        cap.release()
        landmarker.close()

    n = len(rows)
    if n == 0:
        raise LandmarkError(f"Video me koi frame nahi parha gaya: {video_path}")
    n_found = int(sum(found))
    ratio = n_found / n
    _log(f"Chehra mila: {n_found}/{n} frames ({ratio:.0%})", logs)
    if n_found == 0:
        raise LandmarkError(f"Kisi bhi frame me chehra detect nahi hua: {video_path}")
    if ratio < min_detect_ratio:
        raise LandmarkError(f"Detection ratio {ratio:.0%} < required {min_detect_ratio:.0%}")

    zeros = np.zeros((N_MESH, 3), dtype=np.float32)
    if fill_missing:
        first = next(r for r in rows if r is not None)
        prev = first
        for k in range(n):
            if rows[k] is None:
                rows[k] = prev
            else:
                prev = rows[k]
    else:
        rows = [r if r is not None else zeros for r in rows]

    out = np.stack(rows).astype(np.float32)  # (T,468,3)
    if required_only:
        out = out[:, IMPORTANT_LANDMARKS, :]

    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            np.save(save_path, out)
            _log(f"Saved: {save_path} shape={out.shape}", logs)
        except Exception as e:
            raise LandmarkError(f"npy save nahi hui ({save_path}): {e}") from e

    _log(f"Landmarks tayyar shape={out.shape} ({time.time() - t0:.1f}s)", logs)
    return out


def ensure_landmarks(video_path: str, landmarks_path: Optional[str] = None,
                     overwrite: bool = False, logs: bool = False, **kwargs) -> str:
    """
    Landmarks .npy ka path return karta hai. File pehle se ho to wahi (dobara nahi banati),
    warna video se nikal kar save karti hai. Full (T,468,3) format me, jo inference.py ko chahiye.
    landmarks_path na do to video ke saath wala `<name>.npy`; wahan likh na sakein to temp folder.
    """
    if landmarks_path is None:
        landmarks_path = os.path.splitext(video_path)[0] + ".npy"
    if os.path.isfile(landmarks_path) and not overwrite:
        _log(f"Landmarks pehle se maujood: {landmarks_path}", logs)
        return landmarks_path
    try:
        extract_landmarks(video_path, save_path=landmarks_path, logs=logs, **kwargs)
    except OSError:
        alt = os.path.join(tempfile.gettempdir(), os.path.basename(landmarks_path))
        _log(f"{landmarks_path} par likh nahi sake, temp use: {alt}", logs, "WARN")
        extract_landmarks(video_path, save_path=alt, logs=logs, **kwargs)
        return alt
    return landmarks_path


if __name__ == "__main__":
    vid = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    arr = extract_landmarks(vid, save_path=out, logs=True)
    print(arr.shape)
