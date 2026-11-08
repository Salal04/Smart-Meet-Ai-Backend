"""
api.py -- FastAPI endpoints. Model/preprocessing/batching ka kaam engine.py me hai,
settings config.py me, secrets .env me.

Run (sirf EK process -- --workers>1 mat karna):
    python api.py            # ya: uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints (sab POST, multipart/form-data, aur sab me optional form field `meta` = JSON object):
    1) /predict/video            video                          -> landmarks MediaPipe se
    2) /predict/video-landmarks  video + landmarks(.npy)
    3) /predict/frames           frames(16 images) [+ landmarks(.npy)]
    GET /health

Flow (RESULT_URL set ho -- async mode):
    1. Request aati hai -> input validate + preprocess.
       Input kharab -> client ko 422, backend ko kuch nahi jata.
    2. Valid nikli -> Backend 1 (RECEIVED_URL) ko "received" POST + client ko foran 202 {"job_id", "status"}.
    3. Job buffer me jata hai. Jab batch ka model result aa jaye -> Backend 2 (RESULT_URL) ko POST:
       JSON list, har item = [ {...meta ki saari fields, "job_id", "engagement_Level", "confidence"} ]
       (batch me fail hone par item me "confidence" ki jagah "error" aata hai).
RESULT_URL khali ho to purana sync mode: result usi response me.
"""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import config
import applog
import engine
import tcn_inference

log = applog.get_logger("api")


def _safe_url(url: str) -> str:
    """log ke liye URL (query string / credentials ke baghair)."""
    u = urlsplit(url)
    return f"{u.scheme}://{u.hostname}{':' + str(u.port) if u.port else ''}{u.path}"


def _mb(n: int) -> str:
    return f"{n / 1e6:.2f}MB"


def _fail(code: int, msg: str) -> HTTPException:
    """client ko error dene se pehle log bhi karo."""
    log.warning("REJECT %d | %s", code, msg)
    return HTTPException(code, msg)


# --- backend ko JSON POST (retry ke saath) ----------------------------------
_TASKS = set()  # background tasks/futures ka reference (garbage collect na hon)


def _keep(task):
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def _post_json(url: str, token: Optional[str], headers: dict, body, label: str):
    headers = dict(headers)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    target = _safe_url(url)
    t0 = time.perf_counter()
    log.info("CALLBACK start | %s -> %s", label, target)
    async with httpx.AsyncClient(timeout=config.CALLBACK_TIMEOUT) as client:
        for attempt in range(1, config.CALLBACK_RETRIES + 1):
            t_try = time.perf_counter()
            try:
                r = await client.post(url, json=body, headers=headers)
                took = (time.perf_counter() - t_try) * 1000
                if r.status_code < 400:
                    log.info("CALLBACK ok | %s | status=%d try=%d | %.0fms (total %.0fms)",
                             label, r.status_code, attempt, took, (time.perf_counter() - t0) * 1000)
                    return
                log.warning("CALLBACK %s status=%d try=%d/%d | %.0fms | response=%s", label,
                            r.status_code, attempt, config.CALLBACK_RETRIES, took, r.text[:200])
                if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    log.error("CALLBACK %s backend ne reject kiya (%d), retry nahi hoga", label, r.status_code)
                    return  # backend ne reject kiya, retry ka faida nahi
            except httpx.HTTPError as e:
                log.warning("CALLBACK %s %s: %s try=%d/%d | %.0fms", label, type(e).__name__, e, attempt,
                            config.CALLBACK_RETRIES, (time.perf_counter() - t_try) * 1000)
            if attempt < config.CALLBACK_RETRIES:
                wait = config.CALLBACK_BACKOFF_BASE ** attempt
                log.info("CALLBACK %s retry %.1fs baad", label, wait)
                await asyncio.sleep(wait)
    log.error("CALLBACK FAILED | %s | %d koshishon ke baad bhi nahi gaya (%.0fms)", label,
              config.CALLBACK_RETRIES, (time.perf_counter() - t0) * 1000)


def _make_batch_callback(loop: asyncio.AbstractEventLoop):
    """engine ke batcher thread se call hota hai: ek batch ki list Backend 2 ko bhejo."""
    def cb(entries: List[dict]):
        batch_id = uuid.uuid4().hex
        log.info("RESULT batch=%s | %d item(s) Backend 2 ko bhejne ke liye queue | jobs=%s", batch_id[:8],
                 len(entries), [applog.short(e.get(config.JOB_ID_KEY)) for e in entries])
        coro = _post_json(config.RESULT_URL, config.RESULT_TOKEN, {"X-Batch-Id": batch_id}, entries,
                          f"result batch={batch_id[:8]} items={len(entries)}")
        _keep(asyncio.run_coroutine_threadsafe(coro, loop))
    return cb


@asynccontextmanager
async def lifespan(app: FastAPI):
    applog.attach_uvicorn()
    t0 = time.perf_counter()
    log.info("=== SERVER START | mode=%s | received_url=%s | result_url=%s | logs: level=%s file=%s ===",
             "async" if config.RESULT_URL else "sync",
             _safe_url(config.RECEIVED_URL) if config.RECEIVED_URL else None,
             _safe_url(config.RESULT_URL) if config.RESULT_URL else None,
             config.LOG_LEVEL, config.LOG_TO_FILE)
    for name in ("RECEIVED_URL", "RESULT_URL"):
        url = getattr(config, name)
        if url and not url.startswith(("http://", "https://")):
            raise RuntimeError(f"{name} http(s) URL honi chahiye (.env check karo).")
    if config.RESULT_URL and not config.RECEIVED_URL:
        log.warning("RECEIVED_URL set nahi -- 'received' notification nahi jayegi.")
    on_batch = _make_batch_callback(asyncio.get_running_loop()) if config.RESULT_URL else None
    engine.start(on_batch=on_batch)
    log.info("=== SERVER READY | startup %.1fs ===", time.perf_counter() - t0)
    yield
    log.info("=== SERVER STOP ===")
    engine.stop()


app = FastAPI(title="Engagement API", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Har request: id banao, 'API HIT' + 'API DONE' (status + total time) log karo."""
    rid = uuid.uuid4().hex  # yehi job_id banegi
    token = applog.set_rid(rid)
    quiet = request.url.path == "/health"  # health-check ka spam INFO me nahi
    emit = log.debug if quiet else log.info
    client = request.client.host if request.client else "?"
    t0 = time.perf_counter()
    emit("API HIT  | %s %s | client=%s | content-length=%s | rid=%s", request.method, request.url.path,
         client, request.headers.get("content-length", "?"), rid)
    try:
        response = await call_next(request)
    except Exception as e:
        log.error("API CRASH | %s %s | %s: %s | %.0fms", request.method, request.url.path,
                  type(e).__name__, e, (time.perf_counter() - t0) * 1000, exc_info=True)
        raise
    finally:
        applog.reset_rid(token)
    emit("API DONE | %s %s -> %d | %.0fms (client ko response tak)", request.method, request.url.path,
         response.status_code, (time.perf_counter() - t0) * 1000)
    return response


def parse_meta(meta: str) -> dict:
    try:
        m = json.loads(meta) if meta else {}
    except json.JSONDecodeError:
        raise _fail(422, "meta valid JSON nahi hai.")
    if not isinstance(m, dict):
        raise _fail(422, "meta JSON object honi chahiye.")
    log.info("STEP 1 meta parse ok | keys=%s", list(m.keys()))
    return m


def _result(meta: dict, k: int, conf: float):
    return [{**meta, config.ENGAGEMENT_KEY: k, config.CONFIDENCE_KEY: round(conf, 4)}]


def _err_text(e: Exception) -> str:
    if isinstance(e, asyncio.TimeoutError):
        return "Inference timeout."
    return f"{type(e).__name__}: {e}"


# --- sync mode (RESULT_URL khali) -------------------------------------------
async def _run_sync(prep, meta, *args):
    log.info("MODE sync | preprocess -> buffer -> model, result isi response me")
    sw = applog.Stopwatch()
    try:
        k, conf = await engine.infer(prep, *args, job_id=applog.get_rid())
    except engine.BadInput as e:
        raise _fail(422, str(e))
    except engine.BufferFull:
        raise _fail(503, "Buffer full hai, thori der baad dobara try karo.")
    except asyncio.TimeoutError:
        raise _fail(504, "Inference timeout.")
    except Exception as e:
        log.error("Inference fail: %s", _err_text(e), exc_info=True)
        raise HTTPException(500, f"Inference fail: {_err_text(e)}")
    log.info("RESPONSE sync | engagement=%d confidence=%.4f | pipeline total=%.0fms", k, conf, sw.ms())
    return _result(meta, k, conf)


# --- async mode (RESULT_URL set) --------------------------------------------
async def _run_async(prep, meta, *args):
    log.info("MODE async | validate+preprocess -> 202 -> buffer -> batch -> Backend 2")
    sw = applog.Stopwatch()
    try:
        job = await engine.prepare(prep, *args, meta=meta, notify=True,
                                   job_id=applog.get_rid())  # yahan validation hoti hai
    except engine.BadInput as e:
        raise _fail(422, str(e))
    except Exception as e:
        log.error("Preprocess fail: %s", _err_text(e), exc_info=True)
        raise HTTPException(500, f"Preprocess fail: {_err_text(e)}")
    log.info("STEP 3 input VALID | job=%s | preprocess %.0fms", applog.short(job.job_id), sw.ms())
    try:
        engine.submit(job)
    except engine.BufferFull:
        raise _fail(503, "Buffer full hai, thori der baad dobara try karo.")

    # request valid + buffer me -> Backend 1 ko "received"
    if config.RECEIVED_URL:
        body = [{**meta, config.JOB_ID_KEY: job.job_id, "status": "received"}]
        log.info("STEP 5 'received' notification Backend 1 ko schedule | job=%s", applog.short(job.job_id))
        _keep(asyncio.create_task(_post_json(
            config.RECEIVED_URL, config.RECEIVED_TOKEN, {"X-Job-Id": job.job_id}, body,
            f"received job={applog.short(job.job_id)}")))
    log.info("STEP 6 client ko 202 accepted | job=%s | api time %.0fms (model result baad me Backend 2 ko jayega)",
             applog.short(job.job_id), sw.ms())
    return JSONResponse({config.JOB_ID_KEY: job.job_id, "status": "accepted"}, status_code=202)


async def _dispatch(prep, meta: dict, *args):
    if config.RESULT_URL:
        return await _run_async(prep, meta, *args)
    return await _run_sync(prep, meta, *args)


# --- 1) sirf video: landmarks MediaPipe se ---------------------------------
@app.post("/predict/video")
async def predict_video(video: UploadFile = File(...), meta: str = Form("{}")):
    log.info("ENDPOINT /predict/video | landmarks=MediaPipe")
    m = parse_meta(meta)
    data = await video.read()
    log.info("STEP 2 upload read | video=%r size=%s", video.filename, _mb(len(data)))
    return await _dispatch(engine.prep_video, m, data, video.filename)


# --- 2) video + landmarks: 16 frames video se, landmarks file se -----------
@app.post("/predict/video-landmarks")
async def predict_video_landmarks(video: UploadFile = File(...), landmarks: UploadFile = File(...),
                                  meta: str = Form("{}")):
    log.info("ENDPOINT /predict/video-landmarks | landmarks=uploaded .npy")
    m = parse_meta(meta)
    data = await video.read()
    lm = await landmarks.read()
    log.info("STEP 2 upload read | video=%r size=%s | landmarks=%r size=%s", video.filename,
             _mb(len(data)), landmarks.filename, _mb(len(lm)))
    return await _dispatch(engine.prep_video_lm, m, data, video.filename, lm)


# --- 3) 16 frames (+ optional landmarks) ----------------------------------
@app.post("/predict/frames")
async def predict_frames(frames: List[UploadFile] = File(...), landmarks: Optional[UploadFile] = File(None),
                         meta: str = Form("{}")):
    log.info("ENDPOINT /predict/frames | frames=%d landmarks=%s", len(frames),
             "uploaded .npy" if landmarks is not None else "MediaPipe")
    m = parse_meta(meta)
    blobs = [await f.read() for f in frames]
    lm_data = await landmarks.read() if landmarks is not None else None
    log.info("STEP 2 upload read | frames=%d total=%s | landmarks=%s", len(blobs),
             _mb(sum(map(len, blobs))), _mb(len(lm_data)) if lm_data is not None else "-")
    return await _dispatch(engine.prep_frames, m, blobs, lm_data)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "async" if config.RESULT_URL else "sync",
            "buffer": engine.buffer_size(), "batch_size": engine.BATCH_SIZE,
            "num_workers": engine.NUM_WORKERS}


@app.post("/predict/tcn-landmarks")
async def predict_tcn_landmarks(request: Request):
    """TCN endpoint: JSON body contains only landmarks shaped [16][52][3]."""
    try:
        body = await request.json()
    except Exception:
        raise _fail(422, "Request body valid JSON honi chahiye.")
    if not isinstance(body, dict) or set(body) != {"landmarks"}:
        raise _fail(422, 'JSON body me sirf "landmarks" field honi chahiye.')
    try:
        return await asyncio.to_thread(tcn_inference.predict, body["landmarks"])
    except (ValueError, TypeError) as e:
        raise _fail(422, str(e))
    except Exception as e:
        log.error("TCN inference fail: %s: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(500, f"TCN inference fail: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
