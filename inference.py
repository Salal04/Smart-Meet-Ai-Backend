"""
inference.py -- DualStreamTPSBT (RGB + Landmarks) engagement inference module.

Kisi bhi dusri file se (same folder ya SVFAP folder me rakh kar):

    from inference import add_engagement_level

    request = {"video": "video.mp4", "landmarks": "landmarks.npy",
               "meetingID": "2323", "organizationID": "3333", ...koi bhi aur fields...}

    out = add_engagement_level(request, checkpoint_path="/path/checkpoint-best.pth", logs=True)
    # out = request ki copy + "engagement_Level": 2   (baqi sab fields waise hi)

Pehli call par model load hota hai, uske baad cache se use hota hai (dobara load nahi hota).
Checkpoint path env variable se bhi de sakte ho:  ENGAGEMENT_CHECKPOINT=/path/checkpoint-best.pth
List bhi de sakte ho: add_engagement_level([req1, req2, ...]) -> list wapas.
"""
import copy
import os
import sys
import threading
import time
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# is file ka folder (aur optional SVFAP_DIR) path me daalo taake modeling_finetune
# kisi bhi jagah se import ho sake
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.environ.get("SVFAP_DIR")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import modeling_finetune  # noqa: F401,E402  (models ko timm registry me register karta hai)
from timm.models import create_model  # noqa: E402

__all__ = ["load_model", "get_model", "predict", "add_engagement_level",
           "InferenceDataset", "ENGAGENET_CLASSES"]

ENGAGENET_CLASSES = ["Not-Engaged", "Barely-engaged", "Engaged", "Highly-Engaged"]

_LEFT_EYE = [33, 133, 159, 145, 153, 154, 155, 160, 161]
_RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 387, 388]
_MOUTH = [13, 14, 78, 308, 61, 291, 81, 82, 87, 312]
_NOSE = [1, 2, 98, 327, 168]
_JAW = [152, 13, 14, 234, 454, 454, 454]
_L_BROW = [70, 63, 105, 66, 107, 55]
_R_BROW = [336, 296, 334, 293, 300, 283]
IMPORTANT_LANDMARKS = _LEFT_EYE + _RIGHT_EYE + _MOUTH + _NOSE + _JAW + _L_BROW + _R_BROW

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _log(msg: str, logs: bool = True, level: str = "INFO"):
    """logs=True ho to print karta hai, warna chup."""
    if logs:
        print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}", flush=True)


def _warn(msg: str, logs: bool = False):
    """logs=True ho to print, warna normal python warning."""
    if logs:
        _log(msg, True, "WARN")
    else:
        warnings.warn(msg)


# ----------------------------------------------------------------------------
# 1) Model load (+ cache)
# ----------------------------------------------------------------------------
def load_model(
    checkpoint_path: str,
    model_name: str = "dual_stream_base_patch16_160",
    nb_classes: int = 4,
    device: Optional[Union[str, torch.device]] = None,
    use_ema: bool = False,
    model_key: str = "model|module",
    logs: bool = False,
):
    """Model banata hai, checkpoint ke weights load karta hai, eval() mode me return karta hai."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file nahi mili: {checkpoint_path}")

    t0 = time.time()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _log(f"Model load shuru: {model_name} | device={device} | ckpt={checkpoint_path}", logs)

    try:
        model = create_model(model_name, pretrained=False, num_classes=nb_classes,
                             tubelet_size=2, use_mean_pooling=True)
    except Exception as e:
        raise RuntimeError(f"Model '{model_name}' create nahi hua: {e}") from e

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Checkpoint parhne me masla ({checkpoint_path}): {e}") from e
    state = None
    keys = ["model_ema"] if use_ema else []
    keys += model_key.split("|")
    for k in keys:
        if isinstance(ckpt, dict) and k in ckpt:
            state = ckpt[k]
            _log(f"state_dict key = '{k}'", logs)
            break
    if state is None:
        state = ckpt

    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception as e:
        raise RuntimeError(f"Weights load nahi huay (nb_classes/model_name check karo): {e}") from e
    _log(f"missing={len(missing)} unexpected={len(unexpected)}", logs)
    if missing:
        _log(f"missing (pehli 5): {missing[:5]}", logs, "WARN")
    if unexpected:
        _log(f"unexpected (pehli 5): {unexpected[:5]}", logs, "WARN")

    model = model.to(device).eval()
    _log(f"Model ready ({time.time() - t0:.1f}s)", logs)
    return model


_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()


def get_model(checkpoint_path: Optional[str] = None, logs: bool = False, **load_kwargs):
    """Model ek hi dafa load karta hai, phir cache se deta hai (thread-safe)."""
    checkpoint_path = checkpoint_path or os.environ.get("ENGAGEMENT_CHECKPOINT")
    if not checkpoint_path:
        raise ValueError("checkpoint_path do ya ENGAGEMENT_CHECKPOINT env variable set karo.")
    key = (os.path.abspath(checkpoint_path), tuple(sorted(load_kwargs.items())))
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = load_model(checkpoint_path, logs=logs, **load_kwargs)
        else:
            _log("Model cache se use ho raha hai", logs)
        return _MODEL_CACHE[key]


# ----------------------------------------------------------------------------
# 2) Dataset
# ----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    """
    Item: (video_path, landmark_npy_path) | {"video","landmarks","name"} |
          (rgb_tensor[C,T,H,W], landmark_tensor[T,N,3])
    """

    def __init__(self, items: Sequence, num_frames: int = 16, sampling_rate: int = 4,
                 input_size: int = 160, short_side_size: int = 160, logs: bool = False):
        self.logs = logs
        self.items = list(items)
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.short_side_size = short_side_size

    def __len__(self):
        return len(self.items)

    def _indices(self, n_total: int) -> np.ndarray:
        converted = self.num_frames * self.sampling_rate
        if n_total <= converted:
            idx = np.linspace(0, n_total, num=max(n_total // self.sampling_rate, 1))
            pad = np.ones(max(self.num_frames - len(idx), 0)) * n_total
            idx = np.clip(np.concatenate([idx, pad]), 0, n_total - 1).astype(np.int64)
        else:
            start = (n_total - converted) // 2
            end = start + converted
            idx = np.linspace(start, end, num=self.num_frames)
            idx = np.clip(idx, start, end - 1).astype(np.int64)
        return idx[: self.num_frames]

    def _load_video(self, path: str):
        from decord import VideoReader, cpu
        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        idx = self._indices(len(vr))
        frames = vr.get_batch(idx.tolist()).asnumpy()
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        h, w = x.shape[-2:]
        s = self.short_side_size / min(h, w)
        x = F.interpolate(x, size=(max(int(round(h * s)), self.input_size),
                                   max(int(round(w * s)), self.input_size)),
                          mode="bilinear", align_corners=False, antialias=True)
        h, w = x.shape[-2:]
        top, left = (h - self.input_size) // 2, (w - self.input_size) // 2
        x = x[:, :, top:top + self.input_size, left:left + self.input_size]
        x = (x - MEAN) / STD
        return x.permute(1, 0, 2, 3).contiguous(), idx

    def _load_landmarks(self, path: Optional[str], idx: np.ndarray) -> torch.Tensor:
        n = len(IMPORTANT_LANDMARKS)
        if path is None or not os.path.exists(path):
            _warn(f"Landmark file nahi mili: {path}. Zeros use ho rahe hain.", self.logs)
            return torch.zeros(len(idx), n, 3)
        lm = np.load(path)
        if lm.ndim == 4 and lm.shape[1] == 1:
            lm = lm.squeeze(1)
        lm = lm[np.clip(idx, 0, lm.shape[0] - 1)][:, IMPORTANT_LANDMARKS, :]
        return torch.from_numpy(lm).float()

    def __getitem__(self, i):
        it = self.items[i]
        if isinstance(it, dict):
            vid, lm, name = it["video"], it.get("landmarks"), it.get("name")
        elif isinstance(it, (tuple, list)) and isinstance(it[0], torch.Tensor):
            return it[0].float(), it[1].float(), f"item_{i}"
        elif isinstance(it, (tuple, list)):
            vid, lm, name = it[0], (it[1] if len(it) > 1 else None), None
        else:
            vid, lm, name = it, None, None

        rgb, idx = self._load_video(vid)
        lms = self._load_landmarks(lm, idx)
        return rgb, lms, name or os.path.splitext(os.path.basename(vid))[0]


# ----------------------------------------------------------------------------
# 3) Predict
# ----------------------------------------------------------------------------
def _as_dataset(inputs, **ds_kwargs) -> Dataset:
    if isinstance(inputs, Dataset):
        return inputs
    is_single = isinstance(inputs, (dict, str)) or (
        isinstance(inputs, (tuple, list)) and len(inputs) == 2
        and (isinstance(inputs[0], torch.Tensor) or
             (isinstance(inputs[0], str) and (inputs[1] is None or isinstance(inputs[1], str))))
    )
    return InferenceDataset([inputs] if is_single else inputs, **ds_kwargs)


@torch.no_grad()
def predict(model, inputs, batch_size: int = 4, class_names: Optional[List[str]] = None,
            num_workers: int = 0, **ds_kwargs) -> List[dict]:
    """Returns: [{"name", "pred", "label", "probs"}, ...]"""
    class_names = class_names or ENGAGENET_CLASSES
    device = next(model.parameters()).device
    model.eval()

    ds = _as_dataset(inputs, **ds_kwargs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    results = []
    for batch in loader:
        rgb, lms = batch[0].to(device), batch[1].to(device)
        names = batch[3] if len(batch) >= 4 else batch[2]
        if isinstance(names, torch.Tensor):
            names = names.tolist()
        probs = F.softmax(model(rgb, lms).float(), dim=-1).cpu()
        for n, p in zip(names, probs):
            k = int(p.argmax())
            results.append({"name": n, "pred": k,
                            "label": class_names[k] if k < len(class_names) else str(k),
                            "probs": p.tolist()})
    return results


# ----------------------------------------------------------------------------
# 4) Main API: request in -> same request + engagement_Level out
# ----------------------------------------------------------------------------
def _get(meta, key, default=None):
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _set(meta, key, value):
    if isinstance(meta, dict):
        meta[key] = value
    else:
        setattr(meta, key, value)


def _clone(r):
    try:
        return copy.deepcopy(r)
    except Exception:
        return copy.copy(r)


def add_engagement_level(
    requests,
    model=None,
    checkpoint_path: Optional[str] = None,
    video_key: str = "video",
    landmarks_key: str = "landmarks",
    level_key: str = "engagement_Level",
    batch_size: int = 4,
    logs: bool = False,
    raise_errors: bool = False,
    **ds_kwargs,
):
    """
    requests : ek request (dict ya object) ya un ki list. Extra fields kitni bhi ho sakti hain,
               unhein chheda nahi jata.
    model    : load_model() ka output (optional). Na do to checkpoint_path / ENGAGEMENT_CHECKPOINT
               se ek dafa load hoke cache ho jata hai.
    logs     : True ho to har step (model load, har request, errors, timing) print hota hai.
    raise_errors : False (default) -> kharab request me engagement_Level=None, baqi chalti rehti hain.
                   True -> pehla error raise ho jata hai.
    Return   : request(s) ki copy -- saari original fields + `engagement_Level` (int):
               0=Not-Engaged, 1=Barely-engaged, 2=Engaged, 3=Highly-Engaged.
               Original request change nahi hoti.
    """
    t_start = time.time()
    single = not isinstance(requests, (list, tuple))
    reqs = [requests] if single else list(requests)
    _log(f"{len(reqs)} request(s) mili", logs)

    outs = []
    for i, r in enumerate(reqs):
        try:
            o = _clone(r)
            _set(o, level_key, None)
        except Exception as e:
            _log(f"Request {i}: copy/set nahi ho saki: {e}", logs, "ERROR")
            if raise_errors:
                raise
            o = r  # aakhri raasta
        outs.append(o)

    # validation
    valid = []
    for i, r in enumerate(reqs):
        try:
            vid = _get(r, video_key)
        except Exception as e:
            vid = None
            _log(f"Request {i}: '{video_key}' parhne me error: {e}", logs, "ERROR")
        if not vid:
            _log(f"Request {i}: '{video_key}' field khali/missing -> skip", logs, "ERROR")
            if raise_errors:
                raise ValueError(f"Request {i}: '{video_key}' missing")
        elif not os.path.exists(vid):
            _log(f"Request {i}: video file nahi mili ({vid}) -> skip", logs, "ERROR")
            if raise_errors:
                raise FileNotFoundError(vid)
        else:
            valid.append(i)
            _log(f"Request {i}: OK video={vid} landmarks={_get(r, landmarks_key)}", logs)

    if not valid:
        _log("Koi valid request nahi, model run nahi hua", logs, "WARN")
        return outs[0] if single else outs

    if model is None:
        model = get_model(checkpoint_path, logs=logs)  # load fail ho to raise (model ke bina kaam nahi)

    items = [{"video": _get(reqs[i], video_key),
              "landmarks": _get(reqs[i], landmarks_key),
              "name": str(i)} for i in valid]

    def _run(sub_items, bs):
        return predict(model, InferenceDataset(sub_items, logs=logs, **ds_kwargs), batch_size=bs)

    try:
        t0 = time.time()
        results = _run(items, batch_size)
        for i, r in zip(valid, results):
            _set(outs[i], level_key, r["pred"])
            _log(f"Request {i}: {level_key}={r['pred']} ({r['label']}) probs={[round(x, 3) for x in r['probs']]}", logs)
        _log(f"Batch inference done ({time.time() - t0:.2f}s)", logs)
    except Exception as e:
        _log(f"Batch fail hua: {type(e).__name__}: {e} -> ek ek request try ho rahi hai", logs, "WARN")
        if raise_errors:
            raise
        for i, it in zip(valid, items):
            try:
                r = _run([it], 1)[0]
                _set(outs[i], level_key, r["pred"])
                _log(f"Request {i}: {level_key}={r['pred']} ({r['label']})", logs)
            except Exception as e2:
                _log(f"Request {i} fail: {type(e2).__name__}: {e2} -> {level_key}=None", logs, "ERROR")

    done = sum(1 for o in outs if _get(o, level_key) is not None)
    _log(f"Khatam: {done}/{len(reqs)} successful, total {time.time() - t_start:.2f}s", logs)
    return outs[0] if single else outs


if __name__ == "__main__":
    ckpt, vid, lm = sys.argv[1:4]
    print(add_engagement_level({"video": vid, "landmarks": lm}, checkpoint_path=ckpt, logs=True))
