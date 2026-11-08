"""
engine.py -- model ka saara kaam (FastAPI se alag): preprocessing, MediaPipe landmarks,
Dataset, buffer + batching thread, model inference.

Saari settings config.py se aati hain.

API file (api.py) sirf ye use karti hai:
    engine.start(on_batch=None) / engine.stop()   model + MediaPipe load / band
    engine.prep_video / prep_video_lm / prep_frames   input -> (rgb uint8 tensor, landmarks)
    await engine.prepare(prep, *args, meta=, notify=)  input validate + Job banao
    engine.submit(job)                         job buffer me
    await engine.infer(prep, *args)            sync mode: result ka intezar
    engine.BadInput, engine.BufferFull         errors

on_batch(entries): har DataLoader batch ke baad batcher thread se call hota hai (sirf notify=True jobs ke liye).
    entries = [ {...meta, "job_id", "engagement_Level", "confidence"}  # ya "error" ]
"""
import asyncio
import io
import os
import queue
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, InvalidStateError
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config  # sab se pehle: .env load karta hai
import applog
import inference as inf
import landmark_extractor as lmx

BATCH_SIZE = config.BATCH_SIZE
NUM_WORKERS = config.NUM_WORKERS
COLLECT_SIZE = config.COLLECT_SIZE
BATCH_WAIT_MS = config.BATCH_WAIT_MS
MAX_BUFFER = config.MAX_BUFFER
FULL_BATCH_ONLY = config.FULL_BATCH_ONLY
FULL_BATCH_TIMEOUT = config.FULL_BATCH_TIMEOUT
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
CHECKPOINT = config.ENGAGEMENT_CHECKPOINT

NUM_FRAMES = config.NUM_FRAMES
SAMPLING_RATE = config.SAMPLING_RATE
INPUT_SIZE = config.INPUT_SIZE
N_LM = len(inf.IMPORTANT_LANDMARKS)  # 52
log = applog.get_logger("engine")


_MEAN = inf.MEAN.view(3, 1, 1, 1)
_STD = inf.STD.view(3, 1, 1, 1)
# frame sampling bilkul inference.py wali (training jaisi)
_SAMPLER = inf.InferenceDataset([], num_frames=NUM_FRAMES, sampling_rate=SAMPLING_RATE)


class BadInput(ValueError):
    """User ka input kharab -> HTTP 422."""


# ----------------------------------------------------------------------------
# MediaPipe landmarker (ek dafa load, lock ke saath use)
# ----------------------------------------------------------------------------
class LandmarkService:
    def __init__(self, model_path: Optional[str] = None, logs: bool = True):
        lmx.ensure_dependencies(auto_install=True, logs=logs)
        import mediapipe as mp
        self._mp = mp
        model_file = lmx._resolve_model(model_path, logs)  # download + cache
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_file),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,  # stateless: timestamps ka masla nahi
            num_faces=1,
        )
        self._lm = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()  # landmarker thread-safe nahi hai

    def extract(self, frames: np.ndarray) -> np.ndarray:
        """frames (T,H,W,3) uint8 RGB -> (T,52,3) float32"""
        mp = self._mp
        rows = []
        with self._lock:
            for f in frames:
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f))
                res = self._lm.detect(img)
                if res.face_landmarks:
                    rows.append(np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0][:lmx.N_MESH]],
                                         dtype=np.float32))
                else:
                    rows.append(None)
        n_found = sum(r is not None for r in rows)
        log.info("MediaPipe: chehra mila %d/%d frames", n_found, len(rows))
        first = next((r for r in rows if r is not None), None)
        if first is None:
            raise BadInput("Kisi bhi frame me chehra detect nahi hua.")
        prev = first  # missing frames: pichla (ya pehla) mila hua chehra copy
        for k, r in enumerate(rows):
            if r is None:
                rows[k] = prev
            else:
                prev = r
        return np.stack(rows)[:, inf.IMPORTANT_LANDMARKS, :]

    def close(self):
        self._lm.close()


# ----------------------------------------------------------------------------
# Input -> tensors
# ----------------------------------------------------------------------------
def read_video(data: bytes, filename: str):
    """video bytes -> (frames (16,H,W,3) uint8 RGB, idx)"""
    from decord import VideoReader, cpu
    suffix = os.path.splitext(filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        try:
            vr = VideoReader(tmp.name, num_threads=1, ctx=cpu(0))
            n = len(vr)
            if n == 0:
                raise BadInput("Video me koi frame nahi.")
            idx = _SAMPLER._indices(n)
            log.info("Video: file=%r size=%.2fMB total_frames=%d | sampled idx=%s",
                     filename, len(data) / 1e6, n, idx.tolist())
            frames = vr.get_batch(idx.tolist()).asnumpy()
        except BadInput:
            raise
        except Exception as e:
            raise BadInput(f"Video khul nahi saki (corrupt/unsupported): {e}")
        return frames, idx
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def decode_frames(blobs: List[bytes]) -> np.ndarray:
    if len(blobs) != NUM_FRAMES:
        raise BadInput(f"Bilkul {NUM_FRAMES} frames chahiye, {len(blobs)} mili.")
    out = []
    for i, b in enumerate(blobs):
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise BadInput(f"Frame {i} decode nahi hui (image file nahi?).")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if out and img.shape != out[0].shape:
            img = cv2.resize(img, (out[0].shape[1], out[0].shape[0]))
        out.append(img)
    log.info("Frames decode: %d images, shape=%s", len(out), out[0].shape)
    return np.stack(out)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (3,T,160,160) uint8 (resize short side + center crop).
    Normalize dataset me hota hai taake buffer me memory kam lage."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    s = INPUT_SIZE / min(h, w)
    x = F.interpolate(x, size=(max(int(round(h * s)), INPUT_SIZE), max(int(round(w * s)), INPUT_SIZE)),
                      mode="bilinear", align_corners=False, antialias=True)
    h, w = x.shape[-2:]
    top, left = (h - INPUT_SIZE) // 2, (w - INPUT_SIZE) // 2
    x = x[:, :, top:top + INPUT_SIZE, left:left + INPUT_SIZE]
    return x.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()


def parse_landmarks(data: bytes, idx: Optional[np.ndarray]) -> np.ndarray:
    """.npy bytes -> (16,52,3). Accepts (T,468|478,3), (T,1,468,3) ya (T,52,3).
    idx diya ho to un frames ke landmarks chune jate hain, warna file me exactly 16 rows chahiye."""
    try:
        lm = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as e:
        raise BadInput(f"landmarks .npy parh nahi sake: {e}")
    if lm.ndim == 4 and lm.shape[1] == 1:
        lm = lm.squeeze(1)
    if lm.ndim != 3 or lm.shape[-1] != 3:
        raise BadInput(f"landmarks shape galat: {lm.shape} (T,468,3) ya (T,52,3) chahiye.")
    if lm.shape[1] >= 468:
        lm = lm[:, inf.IMPORTANT_LANDMARKS, :]
    elif lm.shape[1] != N_LM:
        raise BadInput(f"landmarks me {lm.shape[1]} points hain; 468/478 ya {N_LM} chahiye.")
    if idx is not None:
        lm = lm[np.clip(idx, 0, len(lm) - 1)]
    elif len(lm) != NUM_FRAMES:
        raise BadInput(f"landmarks me {NUM_FRAMES} rows chahiye, {len(lm)} mili.")
    return lm.astype(np.float32)


# ----------------------------------------------------------------------------
# Dataset + Batcher
# ----------------------------------------------------------------------------
class BufferDataset(Dataset):
    """Buffer se uthaye hue items. Item = (rgb uint8 [3,T,H,W], landmarks float [T,52,3])."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rgb, lms = self.items[i]
        rgb = (rgb.float() / 255.0 - _MEAN) / _STD
        return rgb, lms, i


class Job:
    """Ek request. meta + job_id result ke saath wapas jate hain; notify=True ho to result batch
    callback (on_batch) se bheja jata hai, warna sirf future resolve hota hai (sync mode)."""
    __slots__ = ("rgb", "lms", "future", "meta", "job_id", "notify", "t_submit", "prep_ms")

    def __init__(self, rgb, lms, meta=None, job_id=None, notify=False):
        self.rgb, self.lms = rgb, lms
        self.meta = meta or {}
        self.job_id = job_id or uuid.uuid4().hex
        self.notify = notify
        self.t_submit = None   # buffer me daalne ka waqt (perf_counter)
        self.prep_ms = 0.0     # preprocessing me lagne wala waqt
        self.future: Future = Future()


def _resolve(fut: Future, result=None, error=None):
    try:
        if error is not None:
            fut.set_exception(error)
        else:
            fut.set_result(result)
    except InvalidStateError:  # client timeout se cancel ho chuka
        pass


def _entry(job: Job, result=None, error=None) -> dict:
    """Batch callback ki list ka ek item: meta ki saari fields + job_id + result (ya error)."""
    e = {**job.meta, config.JOB_ID_KEY: job.job_id}
    if error is not None:
        e["error"] = f"{type(error).__name__}: {error}"
    else:
        e[config.ENGAGEMENT_KEY] = result[0]
        e[config.CONFIDENCE_KEY] = round(result[1], 4)
    return e


class Batcher(threading.Thread):
    """Buffer (queue) se items utha kar Dataset -> DataLoader(batch_size, num_workers) -> model."""

    def __init__(self, model, on_batch: Optional[Callable[[List[dict]], None]] = None):
        super().__init__(daemon=True, name="batcher")
        self.model = model
        self.device = next(model.parameters()).device
        self.on_batch = on_batch
        self.buffer: "queue.Queue[Job]" = queue.Queue(maxsize=MAX_BUFFER)
        self._halt = threading.Event()
        self._carry: List[Job] = []
        self._round = 0

    def submit(self, job: Job):
        self.buffer.put_nowait(job)  # full ho to queue.Full

    def shutdown(self):
        self._halt.set()

    def _report(self, entries: List[dict]):
        """Ek batch ka result backend ke liye (api.py ka callback). Fail ho to batcher na ruke."""
        if not entries or self.on_batch is None:
            return
        try:
            self.on_batch(entries)
        except Exception as e:
            log.error("on_batch callback fail: %s: %s", type(e).__name__, e, exc_info=True)

    def _collect(self) -> List[Job]:
        """Default: pehla item + BATCH_WAIT_MS tak aur items (max COLLECT_SIZE). Pura batch ka intezar nahi."""
        try:
            jobs = [self.buffer.get(timeout=0.5)]
        except queue.Empty:
            return []
        deadline = time.monotonic() + BATCH_WAIT_MS / 1000.0
        while len(jobs) < COLLECT_SIZE:
            try:
                remaining = deadline - time.monotonic()
                jobs.append(self.buffer.get(timeout=remaining) if remaining > 0
                            else self.buffer.get_nowait())
            except queue.Empty:
                break
        return jobs

    def _collect_full(self) -> List[Job]:
        """FULL_BATCH_ONLY: jab tak BATCH_SIZE items na hon intezar. Phir jo buffer me pehle se hai
        (max COLLECT_SIZE) utha lo, lekin sirf BATCH_SIZE ke multiple; baqi agle round ke liye carry."""
        jobs, self._carry = self._carry, []
        first_t = time.monotonic() if jobs else None
        while len(jobs) < BATCH_SIZE and not self._halt.is_set():
            try:
                jobs.append(self.buffer.get(timeout=0.5))
                if first_t is None:
                    first_t = time.monotonic()
            except queue.Empty:
                if (FULL_BATCH_TIMEOUT > 0 and jobs and
                        time.monotonic() - first_t >= FULL_BATCH_TIMEOUT):
                    return jobs  # timeout: adhoora batch hi sahi
        if len(jobs) < BATCH_SIZE:  # shutdown
            self._carry = jobs
            return []
        limit = max(BATCH_SIZE, COLLECT_SIZE)
        while len(jobs) < limit:
            try:
                jobs.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        n = len(jobs) // BATCH_SIZE * BATCH_SIZE
        self._carry = jobs[n:]
        return jobs[:n]

    def run(self):
        while not self._halt.is_set():
            jobs = self._collect_full() if FULL_BATCH_ONLY else self._collect()
            if not jobs:
                continue
            log.info("COLLECT: %d job(s) buffer se uthay | mode=%s | buffer_baqi=%d | carry=%d",
                     len(jobs), "full_batch" if FULL_BATCH_ONLY else "normal",
                     self.buffer.qsize(), len(self._carry))
            try:
                self._process(jobs)
            except Exception as e:
                log.error("ROUND fail: %s: %s | jobs=%s", type(e).__name__, e,
                          [applog.short(j.job_id) for j in jobs], exc_info=True)
                failed = [j for j in jobs if not j.future.done()]  # jin ka result pehle nahi gaya
                for j in failed:
                    _resolve(j.future, error=e)
                self._report([_entry(j, error=e) for j in failed if j.notify])
        for j in self._carry:  # shutdown par bache hue jobs fail
            _resolve(j.future, error=RuntimeError("Server band ho raha hai."))

    @torch.inference_mode()
    def _process(self, jobs: List[Job]):
        self._round += 1
        rnd = self._round
        t_round = time.perf_counter()
        ds = BufferDataset([(j.rgb, j.lms) for j in jobs])  # futures dataset me nahi jate
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        n_batches = (len(jobs) + BATCH_SIZE - 1) // BATCH_SIZE
        log.info("ROUND #%d start | jobs=%d -> %d batch(es) of <=%d | device=%s",
                 rnd, len(jobs), n_batches, BATCH_SIZE, self.device)
        for b, (rgb, lms, idx) in enumerate(loader, 1):  # ek iteration = ek batch
            ids = idx.tolist()
            t_b = time.perf_counter()
            log.info("BATCH #%d.%d | size=%d | jobs=%s | rgb=%s lms=%s", rnd, b, len(ids),
                     [applog.short(jobs[i].job_id) for i in ids], tuple(rgb.shape), tuple(lms.shape))
            probs = F.softmax(self.model(rgb.to(self.device), lms.to(self.device)).float(), dim=-1).cpu()
            infer_ms = (time.perf_counter() - t_b) * 1000
            log.info("BATCH #%d.%d model forward ok | %.0fms (%.0fms/sample)", rnd, b, infer_ms,
                     infer_ms / max(len(ids), 1))
            entries = []
            for i, p in zip(ids, probs):
                conf, k = p.max(0)
                res = (int(k), float(conf))
                job = jobs[i]
                queue_ms = (t_b - job.t_submit) * 1000 if job.t_submit else 0.0
                total_ms = job.prep_ms + ((time.perf_counter() - job.t_submit) * 1000 if job.t_submit else 0.0)
                log.info("JOB DONE job=%s | engagement=%d conf=%.4f | prep=%.0fms queue_wait=%.0fms "
                         "infer=%.0fms | total_in_server=%.0fms",
                         applog.short(job.job_id), res[0], res[1], job.prep_ms, queue_ms, infer_ms, total_ms)
                _resolve(job.future, result=res)
                if job.notify:
                    entries.append(_entry(job, result=res))
            if entries:
                log.info("BATCH #%d.%d result backend ko bhejna: %d item(s)", rnd, b, len(entries))
            self._report(entries)  # is batch ka result (list + meta) backend 2 ko
        log.info("ROUND #%d done | %.0fms", rnd, (time.perf_counter() - t_round) * 1000)


# ----------------------------------------------------------------------------
# Public interface (api.py ye use karti hai)
# ----------------------------------------------------------------------------
landmarker: Optional[LandmarkService] = None
batcher: Optional[Batcher] = None


class BufferFull(RuntimeError):
    pass


def start(on_batch: Optional[Callable[[List[dict]], None]] = None):
    """Model + MediaPipe load karo aur batching thread chalao."""
    global landmarker, batcher
    log.info("STARTUP: model load ho raha hai | checkpoint=%s", CHECKPOINT)
    with applog.step(log, "STARTUP model load"):
        model = inf.get_model(CHECKPOINT, logs=True)  # ENGAGEMENT_CHECKPOINT na ho to yahin error
    with applog.step(log, "STARTUP MediaPipe landmarker load"):
        landmarker = LandmarkService()
    batcher = Batcher(model, on_batch)
    batcher.start()
    log.info("READY | device=%s batch_size=%d num_workers=%d collect=%d wait_ms=%d max_buffer=%d "
             "full_batch_only=%s full_batch_timeout=%ds frames=%d sampling_rate=%d input=%d batch_callback=%s",
             batcher.device, BATCH_SIZE, NUM_WORKERS, COLLECT_SIZE, BATCH_WAIT_MS, MAX_BUFFER,
             bool(FULL_BATCH_ONLY), FULL_BATCH_TIMEOUT, NUM_FRAMES, SAMPLING_RATE, INPUT_SIZE,
             on_batch is not None)


def stop():
    log.info("SHUTDOWN: batcher + landmarker band ho rahe hain")
    if batcher:
        batcher.shutdown()
    if landmarker:
        landmarker.close()


def buffer_size() -> Optional[int]:
    return batcher.buffer.qsize() if batcher else None


def prep_video(data, filename):
    """Sirf video: 16 frames + landmarks MediaPipe se."""
    with applog.step(log, "PREP 1/3 video decode + frame sampling (decord)"):
        frames, _ = read_video(data, filename)
    with applog.step(log, "PREP 2/3 resize + center-crop -> tensor"):
        rgb = frames_to_tensor(frames)
    with applog.step(log, "PREP 3/3 landmarks (MediaPipe)"):
        lms = landmarker.extract(frames)
    return rgb, lms


def prep_video_lm(data, filename, lm_data):
    """Video + landmarks .npy: 16 frames video se, landmarks file se (wahi indices)."""
    with applog.step(log, "PREP 1/3 video decode + frame sampling (decord)"):
        frames, idx = read_video(data, filename)
    with applog.step(log, "PREP 2/3 resize + center-crop -> tensor"):
        rgb = frames_to_tensor(frames)
    with applog.step(log, "PREP 3/3 landmarks .npy parse (file se)"):
        lms = parse_landmarks(lm_data, idx)
    return rgb, lms


def prep_frames(blobs, lm_data):
    """16 frames (+ optional landmarks .npy); landmarks na hon to MediaPipe se."""
    with applog.step(log, "PREP 1/3 frames decode"):
        frames = decode_frames(blobs)
    with applog.step(log, "PREP 2/3 resize + center-crop -> tensor"):
        rgb = frames_to_tensor(frames)
    if lm_data is None:
        with applog.step(log, "PREP 3/3 landmarks (MediaPipe)"):
            lms = landmarker.extract(frames)
    else:
        with applog.step(log, "PREP 3/3 landmarks .npy parse (file se)"):
            lms = parse_landmarks(lm_data, None)
    return rgb, lms


async def prepare(prep, *args, meta: Optional[dict] = None, notify: bool = False,
                  job_id: Optional[str] = None) -> Job:
    """Input validate + preprocess (blocking kaam thread me). Kharab input par BadInput.
    Kamyab ho to Job milti hai -- iska matlab request VALID hai."""
    sw = applog.Stopwatch()
    log.info("PREPROCESS start | prep=%s", getattr(prep, "__name__", prep))
    rgb, lms = await asyncio.to_thread(prep, *args)
    job = Job(rgb, torch.from_numpy(lms).float(), meta=meta, job_id=job_id, notify=notify)
    job.prep_ms = sw.ms()
    log.info("PREPROCESS ok | job=%s rgb=%s lms=%s | total=%.0fms",
             applog.short(job.job_id), tuple(rgb.shape), tuple(lms.shape), job.prep_ms)
    return job


def submit(job: Job):
    """Job ko buffer me daalo. Buffer full ho to BufferFull."""
    job.t_submit = time.perf_counter()
    try:
        batcher.submit(job)
    except queue.Full:
        log.warning("BUFFER FULL | job=%s reject (max=%d)", applog.short(job.job_id), MAX_BUFFER)
        raise BufferFull("Buffer full hai.")
    log.info("SUBMIT job=%s buffer me gaya | buffer=%d/%d", applog.short(job.job_id),
             batcher.buffer.qsize(), MAX_BUFFER)


async def infer(prep, *args, job_id: Optional[str] = None):
    """Sync mode: prep -> buffer -> (level, confidence) ka intezar."""
    job = await prepare(prep, *args, job_id=job_id)
    submit(job)
    sw = applog.Stopwatch()
    log.info("WAIT job=%s result ka intezar (timeout=%ds)", applog.short(job.job_id), REQUEST_TIMEOUT)
    try:
        res = await asyncio.wait_for(asyncio.wrap_future(job.future), REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        log.error("TIMEOUT job=%s %.0fms baad bhi result nahi aaya", applog.short(job.job_id), sw.ms())
        raise
    log.info("WAIT job=%s result mil gaya | waited=%.0fms", applog.short(job.job_id), sw.ms())
    return res
