"""
applog.py -- poore project ka logging (console + file, timing ke saath).

Settings config.py ke LOG_* se aati hain:
    LOGS_ENABLED=0   -> har tarah ka log band (console bhi, file bhi)
    LOG_LEVEL        -> DEBUG / INFO / WARNING / ERROR
    LOG_CONSOLE      -> terminal par dikhao
    LOG_TO_FILE      -> file me save karo (logs/engagement.log, rotate hoti hai)

Use:
    import applog
    log = applog.get_logger("engine")
    log.info("hello")
    with applog.step(log, "resize"):      # khud time naap kar "resize ok | 12ms" likhta hai
        ...
    sw = applog.Stopwatch(); ...; sw.ms()

Har request ko ek `rid` (= job_id) milti hai (api.py ka middleware set karta hai). Wo har log line me
(pehle 8 chars) aati hai, to ek request ka poora flow `grep <rid>` se mil jata hai.
Preprocessing thread me bhi rid khud chali jati hai (contextvars).
"""
import contextvars
import logging
import os
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

import config

ROOT = "engagement"

_rid = contextvars.ContextVar("rid", default="-")
_configured = False
_file_handler = None


# --- request id --------------------------------------------------------------
def set_rid(rid: str):
    return _rid.set(rid)


def reset_rid(token):
    _rid.reset(token)


def get_rid() -> str:
    return _rid.get()


def short(job_id) -> str:
    """log me dikhane ke liye job_id ke pehle 8 chars."""
    return str(job_id)[:8]


class _RidFilter(logging.Filter):
    def filter(self, record):
        record.rid = short(_rid.get())
        return True


_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(threadName)-10.10s | %(rid)-8s | %(name)-14s | %(message)s"


# --- setup -------------------------------------------------------------------
def setup():
    """Ek hi dafa chalta hai (get_logger khud bula leta hai)."""
    global _configured, _file_handler
    if _configured:
        return
    _configured = True

    root = logging.getLogger(ROOT)
    root.propagate = False
    root.handlers.clear()

    if not config.LOGS_ENABLED:
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.CRITICAL + 1)
        return

    root.setLevel(getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO))
    fmt = logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S")
    flt = _RidFilter()

    if config.LOG_CONSOLE:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.addFilter(flt)
        root.addHandler(sh)

    if config.LOG_TO_FILE:
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            path = os.path.join(config.LOG_DIR, config.LOG_FILE_NAME)
            _file_handler = RotatingFileHandler(
                path, maxBytes=config.LOG_MAX_MB * 1024 * 1024,
                backupCount=config.LOG_BACKUPS, encoding="utf-8")
            _file_handler.setFormatter(fmt)
            _file_handler.addFilter(flt)
            root.addHandler(_file_handler)
            root.info("Log file: %s (max %dMB x %d backups)", path, config.LOG_MAX_MB, config.LOG_BACKUPS)
        except OSError as e:
            print(f"[WARN] log file nahi ban saki ({e}); sirf console par log hoga.", flush=True)

    if not root.handlers:
        root.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"{ROOT}.{name}")


def attach_uvicorn():
    """uvicorn ke apne logs (access log, errors) bhi usi file me. uvicorn start hone ke BAAD bulao."""
    if not (config.LOGS_ENABLED and config.LOG_UVICORN and _file_handler):
        return
    for n in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(n)
        if _file_handler not in lg.handlers:
            lg.addHandler(_file_handler)


# --- timing helpers ----------------------------------------------------------
class Stopwatch:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


@contextmanager
def step(log: logging.Logger, label: str):
    """with step(log, "kaam"): ...  ->  "kaam ok | 12ms"  (fail par "kaam FAILED | 12ms | Error: ...")."""
    sw = Stopwatch()
    log.debug("%s ...", label)
    try:
        yield sw
    except Exception as e:
        log.warning("%s FAILED | %.0fms | %s: %s", label, sw.ms(), type(e).__name__, e)
        raise
    log.info("%s ok | %.0fms", label, sw.ms())
