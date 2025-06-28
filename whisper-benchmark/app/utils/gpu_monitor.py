import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pynvml
import torch


logger = logging.getLogger("whisper-benchmark")

pynvml.nvmlInit()
HAS_PYNVML = True


class GPUMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.stats: Dict[str, List[float]] = {"memory_used": [], "utilization": []}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.device_count = 0
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
        elif HAS_PYNVML:
            self.device_count = pynvml.nvmlDeviceGetCount()

        if self.device_count == 0:
            logger.warning("GPU не обнаружены или недоступны")

    def start(self):
        if self.device_count == 0:
            return

        self.stats = {"memory_used": [], "utilization": []}
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> Dict[str, float]:
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            return {"max_memory_used_mb": 0.0, "avg_utilization": 0.0}

        self._stop_event.set()
        self._monitor_thread.join()

        if not self.stats["memory_used"]:
            return {"max_memory_used_mb": 0.0, "avg_utilization": 0.0}

        max_memory_used = max(self.stats["memory_used"])
        avg_utilization = (
            float(np.mean(self.stats["utilization"]))
            if self.stats["utilization"]
            else 0.0
        )

        return {
            "max_memory_used_mb": round(max_memory_used / (1024 * 1024), 2),
            "avg_utilization": round(avg_utilization, 2),
        }

    def _monitor(self):
        while not self._stop_event.is_set():
            self._collect_stats()
            time.sleep(self.interval)

    def _collect_stats(self):
        if torch.cuda.is_available():
            memory_used = 0.0

            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used += float(memory_info.used)

            self.stats["memory_used"].append(memory_used)

            utilization_sum = 0.0
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_sum += float(utilization.gpu)

            avg_utilization = utilization_sum / self.device_count
            self.stats["utilization"].append(avg_utilization)
        elif HAS_PYNVML:
            memory_used = 0.0
            utilization_sum = 0.0

            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used += float(memory_info.used)

                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_sum += float(utilization.gpu)

            self.stats["memory_used"].append(memory_used)
            avg_utilization = (
                utilization_sum / self.device_count if self.device_count > 0 else 0.0
            )
            self.stats["utilization"].append(avg_utilization)
