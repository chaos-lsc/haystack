"""Sample whole-host RAM/swap and this process; distinguish Windows from acceptance."""

import platform
import threading
import time

import psutil


class ResourceMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.result = {
            "platform": platform.platform(),
            "physical_total": psutil.virtual_memory().total,
            "swap_total": psutil.swap_memory().total,
            "process_rss_peak": 0,
            "host_used_peak": 0,
            "host_available_min": psutil.virtual_memory().total,
            "swap_used_peak": 0,
            "samples": 0,
            "started_at": time.time(),
            "acceptance_claim": False,
        }
        initial_swap = psutil.swap_memory()
        self.swap_start = (initial_swap.sin, initial_swap.sout)

    def _sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            ram, swap = psutil.virtual_memory(), psutil.swap_memory()
            self.result["process_rss_peak"] = max(self.result["process_rss_peak"], process.memory_info().rss)
            self.result["host_used_peak"] = max(self.result["host_used_peak"], ram.total - ram.available)
            self.result["host_available_min"] = min(self.result["host_available_min"], ram.available)
            self.result["swap_used_peak"] = max(self.result["swap_used_peak"], swap.used)
            self.result["swap_sin"] = swap.sin
            self.result["swap_sout"] = swap.sout
            self.result["swap_read_bytes_during_run"] = max(0, swap.sin - self.swap_start[0])
            self.result["swap_write_bytes_during_run"] = max(0, swap.sout - self.swap_start[1])
            self.result["samples"] += 1
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.result["seconds"] = time.time() - self.result["started_at"]
