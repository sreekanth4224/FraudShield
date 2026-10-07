import threading
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CALIBRATION_FILE = MODELS_DIR / "calibration.json"

def torch_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def wait_gpu(torch, device):
    if device != "cuda":
        return
    done = torch.cuda.Event()
    done.record()
    while not done.query():
        time.sleep(0.001)

LOAD_LOCK = threading.RLock()
