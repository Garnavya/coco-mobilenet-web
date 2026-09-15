import csv
import time
import psutil
import threading
from datetime import datetime, timezone
from pathlib import Path

class SystemMonitor:
    def __init__(self, output_csv: Path, interval_seconds: float, gpu_index: int, 
                 pause_event: threading.Event, abort_event: threading.Event):
        self.output_csv = output_csv
        self.interval = interval_seconds
        self.gpu_index = gpu_index
        
        self.pause_event = pause_event
        self.abort_event = abort_event
        self.stop_event = threading.Event()
        
        self.thread = None
        self.nvml_initialized = False
        self.gpu_handle = None
        self.start_time = None
        
        self.HEADERS = [
            "timestamp", "elapsed_seconds", "cpu_percent", 
            "ram_used_gb", "ram_available_gb", "ram_percent",
            "gpu_util_percent", "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w",
            "gpu_vram_used_mb", "gpu_vram_total_mb", "gpu_vram_percent",
            "gpu_clock_mhz", "gpu_memory_clock_mhz",
            "battery_percent", "is_ac_plugged"
        ]

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.nvml_initialized = True
        except Exception as e:
            print(f"Monitor warning: NVML initialization failed: {e}")

    def _shutdown_nvml(self):
        if self.nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def start(self):
        self._init_nvml()
        # Daemon=True ensures if main thread crashes horribly, this thread doesn't hang the process
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.start_time = time.monotonic()
        # Initialize psutil CPU percent (first call always returns 0.0)
        psutil.cpu_percent() 
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self._shutdown_nvml()

    def _monitor_loop(self):
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            
            while not self.stop_event.is_set():
                try:
                    row = self._gather_metrics()
                    writer.writerow(row)
                    
                    # Disk flush logic: 10s is infrequent. We prioritize crash-safety over microscopic IO savings.
                    # This ensures no telemetry is lost if the laptop abruptly dies.
                    f.flush() 
                    
                except Exception as e:
                    print(f"Monitor loop error (ignored): {e}")
                
                # Use event wait instead of sleep so stop() returns instantly
                self.stop_event.wait(self.interval)

    def _gather_metrics(self) -> list:
        now = datetime.now(timezone.utc).astimezone()
        elapsed = round(time.monotonic() - self.start_time, 2)
        
        # CPU & RAM
        cpu_pct = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_used_gb = round(ram.used / (1024**3), 2)
        ram_avail_gb = round(ram.available / (1024**3), 2)
        
        # Battery / AC Logic
        battery = psutil.sensors_battery()
        bat_pct, is_ac = "", ""
        
        if battery:
            bat_pct = round(battery.percent, 1)
            is_ac = bool(battery.power_plugged)
            
            # --- FAIL SAFE LOGIC TRIGGERS ---
            if not is_ac:
                if bat_pct < 40:
                    self.abort_event.set()
                    self.pause_event.clear() # Unpause to allow immediate shutdown
                else:
                    self.pause_event.set()
            else:
                self.pause_event.clear() # AC restored, clear pause
        
        # GPU Metrics
        g_util = g_temp = g_pow = g_pow_lim = g_mem_use = g_mem_tot = g_mem_pct = g_clk = g_mclk = ""
        
        if self.nvml_initialized and self.gpu_handle:
            import pynvml
            try:
                utils = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                g_util = utils.gpu
            except pynvml.NVMLError: pass
            
            try:
                g_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError: pass

            try:
                # Milliwatts to Watts
                g_pow = round(pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0, 2)
            except pynvml.NVMLError: pass

            try:
                g_pow_lim = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(self.gpu_handle) / 1000.0, 2)
            except pynvml.NVMLError: pass

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                g_mem_use = round(mem.used / (1024**2), 2)
                g_mem_tot = round(mem.total / (1024**2), 2)
                g_mem_pct = round((mem.used / mem.total) * 100, 1) if mem.total > 0 else 0
            except pynvml.NVMLError: pass
            
            try:
                g_clk = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
            except pynvml.NVMLError: pass

            try:
                g_mclk = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError: pass

        return [
            now.isoformat(), elapsed, cpu_pct, ram_used_gb, ram_avail_gb, ram.percent,
            g_util, g_temp, g_pow, g_pow_lim, g_mem_use, g_mem_tot, g_mem_pct, g_clk, g_mclk,
            bat_pct, is_ac
        ]