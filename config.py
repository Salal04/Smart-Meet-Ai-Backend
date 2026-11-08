"""
config.py -- poore project ki settings ek jagah.

* Secrets (URLs + tokens) .env se aate hain; baqi settings yahan defaults ke saath hain.
* Har setting ko environment variable se override kiya ja sakta hai (docker/systemd me kaam aata hai).
* Is file ko sab se pehle import karo (engine.py aur api.py karte hain), taake .env load ho jaye --
  landmark_extractor.py wale LANDMARK_MODEL_PATH / LANDMARK_MODEL_DIR bhi .env se aa jate hain.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# .env isi folder se load hoti hai. override=False: asli environment variable .env par ghalib rehta hai.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# =============================================================================
# SECRETS (.env se) -- values .env me rakho, yahan nahi
# =============================================================================
# Backend 1: valid request milne par "received" notification yahan jati hai.
RECEIVED_URL = _str("RECEIVED_URL")
RECEIVED_TOKEN = _str("RECEIVED_TOKEN")   # set ho to header "Authorization: Bearer <token>" jata hai

# Backend 2: batch ka result yahan jata hai. Set ho to API async mode me chalti hai
# (client ko foran 202 milta hai); khali ho to result usi request ke response me aata hai.
RESULT_URL = _str("RESULT_URL")
RESULT_TOKEN = _str("RESULT_TOKEN")       # set ho to header "Authorization: Bearer <token>" jata hai

# =============================================================================
# MODEL
# =============================================================================
# Trained checkpoint ka path (zaroori). Server start par yahin se model load hota hai.
ENGAGEMENT_CHECKPOINT = _str("ENGAGEMENT_CHECKPOINT")

# =============================================================================
# PREPROCESSING -- ye values training se match karni chahiye, bina retrain badalna mat
# =============================================================================
NUM_FRAMES = _int("NUM_FRAMES", 16)        # model ko ek sample me kitne frames chahiye
SAMPLING_RATE = _int("SAMPLING_RATE", 4)   # video se frames kitne gap par uthte hain (inference.py wala sampler)
INPUT_SIZE = _int("INPUT_SIZE", 160)       # frame ka final size (short side resize + center crop) = INPUT_SIZE x INPUT_SIZE

# =============================================================================
# BATCHING (buffer -> DataLoader -> model)
# =============================================================================
BATCH_SIZE = _int("BATCH_SIZE", 4)                   # DataLoader ek forward pass me kitne samples dalta hai
NUM_WORKERS = _int("NUM_WORKERS", 0)                 # DataLoader worker processes (0 = main thread; tensors chhote hain, 0 theek hai)
COLLECT_SIZE = _int("COLLECT_SIZE", 16)              # ek round me buffer se zyada se zyada kitne jobs uthne hain
BATCH_WAIT_MS = _int("BATCH_WAIT_MS", 50)            # pehla job aane ke baad aur jobs ka itne ms intezar
MAX_BUFFER = _int("MAX_BUFFER", 256)                 # buffer me max jobs; us se zyada par client ko 503
FULL_BATCH_ONLY = _int("FULL_BATCH_ONLY", 0)         # 1 = model tab chalta hai jab pure BATCH_SIZE jobs jama hon
FULL_BATCH_TIMEOUT = _int("FULL_BATCH_TIMEOUT", 0)   # sec; FULL_BATCH_ONLY=1 ho to itni der baad adhoora batch bhi chala do (0 = hamesha pura intezar)
REQUEST_TIMEOUT = _int("REQUEST_TIMEOUT", 120)       # sec; sync mode me result ke intezar ki hadd

# =============================================================================
# BACKEND CALLBACKS (received + batch result dono par lagu)
# =============================================================================
CALLBACK_RETRIES = _int("CALLBACK_RETRIES", 3)       # POST fail ho to kitni dafa koshish
CALLBACK_TIMEOUT = _int("CALLBACK_TIMEOUT", 10)      # sec; ek POST ki hadd
CALLBACK_BACKOFF_BASE = _float("CALLBACK_BACKOFF_BASE", 2.0)  # retry ke beech intezar = BASE ** attempt sec

# =============================================================================
# SERVER
# =============================================================================
HOST = _str("HOST", "0.0.0.0")   # kis interface par sunna hai
PORT = _int("PORT", 8000)        # port (sirf EK process chalao, --workers > 1 nahi)

# =============================================================================
# LOGGING (applog.py use karta hai)
# =============================================================================
LOGS_ENABLED = _bool("LOGS_ENABLED", True)      # master switch: 0 = koi log nahi (na console na file)
LOG_LEVEL = _str("LOG_LEVEL", "INFO")           # DEBUG (sab kuch) / INFO (har major step) / WARNING / ERROR
LOG_CONSOLE = _bool("LOG_CONSOLE", True)        # terminal par log dikhao
LOG_TO_FILE = _bool("LOG_TO_FILE", True)        # file me save karo
LOG_DIR = _str("LOG_DIR", str(Path(__file__).resolve().parent / "logs"))  # log folder
LOG_FILE_NAME = _str("LOG_FILE_NAME", "engagement.log")
LOG_MAX_MB = _int("LOG_MAX_MB", 10)             # ek file itne MB ki ho to nayi shuru (purani rotate)
LOG_BACKUPS = _int("LOG_BACKUPS", 5)            # kitni purani log files rakhni hain
LOG_UVICORN = _bool("LOG_UVICORN", True)        # uvicorn ke access/error logs bhi isi file me

# =============================================================================
# OUTPUT JSON KEYS
# =============================================================================
ENGAGEMENT_KEY = "engagement_Level"   # 0=Not-Engaged 1=Barely 2=Engaged 3=Highly
CONFIDENCE_KEY = "confidence"         # winning class ki softmax probability (4 decimals)
JOB_ID_KEY = "job_id"                 # har request ki unique id; received aur result dono me aati hai
