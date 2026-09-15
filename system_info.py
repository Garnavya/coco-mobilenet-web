import json
import platform
import psutil
import torch
from pathlib import Path
from datetime import datetime, timezone
from config import config

def get_static_system_info() -> dict:
    info = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine()
        },
        "cpu": {
            "model": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "python": {
            "version": platform.python_version()
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
        "gpu": None
    }

    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(config.GPU_INDEX)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu"] = {
                "name": pynvml.nvmlDeviceGetName(handle),
                "index": config.GPU_INDEX,
                "driver_version": pynvml.nvmlSystemGetDriverVersion(),
                "total_vram_mb": round(mem_info.total / (1024**2), 2)
            }
            pynvml.nvmlShutdown()
        except Exception as e:
            info["gpu"] = {"error": f"Failed to get NVML info: {str(e)}"}

    return info

def save_system_info(output_path: Path):
    info = get_static_system_info()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4)