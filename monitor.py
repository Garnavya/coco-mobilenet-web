import csv
import time
import psutil
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress deprecation warning from pynvml/nvidia-ml-py
warnings.filterwarnings('ignore', category=FutureWarning)

def decode_throttle_reasons(mask: int) -> str:
    """
    Decodes NVML throttle reasons bitmask into human-readable pipe-delimited flags.
    Distinguishes OEM software power capping from thermal throttling and hardware slowdowns.
    """
    if not mask or mask == 1:
        return "None"
    mapping = {
        0x0000000000000002: "GpuIdle",
        0x0000000000000004: "SwPowerCap",
        0x0000000000000008: "HwSlowdown",
        0x0000000000000010: "SyncBoost",
        0x0000000000000020: "SwThermal",
        0x0000000000000040: "HwThermal",
        0x0000000000000080: "HwPowerBrake",
        0x0000000000000100: "DisplayClockSetting",
    }
    reasons = [name for bit, name in mapping.items() if (mask & bit)]
    return "|".join(reasons) if reasons else f"0x{mask:x}"

class SystemMonitor:
    def __init__(self, output_csv: Path, interval_seconds: float = 10.0, gpu_index: int = 0, 
                 pause_event: threading.Event = None, abort_event: threading.Event = None,
                 battery_abort_threshold: float = 40.0):
        self.output_csv = Path(output_csv)
        self.interval = interval_seconds
        self.gpu_index = gpu_index
        
        self.pause_event = pause_event if pause_event is not None else threading.Event()
        self.abort_event = abort_event if abort_event is not None else threading.Event()
        self.stop_event = threading.Event()
        self.battery_abort_threshold = battery_abort_threshold
        
        self.thread = None
        self.nvml_initialized = False
        self.gpu_handle = None
        self.start_time = None
        
        # State tracking for rate metrics
        self.prev_disk_io = None
        self.prev_disk_time = None
        self.prev_bat_pct = None
        self.prev_bat_time = None
        
        self.HEADERS = [
            "timestamp", "elapsed_seconds", 
            # CPU & System Memory
            "cpu_percent", "ram_used_gb", "ram_available_gb", "ram_percent",
            "swap_used_gb", "swap_percent",
            # Disk I/O Throughput
            "disk_read_mbs", "disk_write_mbs",
            # GPU Utilization & Thermals
            "gpu_util_percent", "gpu_mem_util_percent", "gpu_temp_c",
            # GPU Power & Clocks
            "gpu_power_w", "gpu_power_limit_w", "gpu_clock_mhz", "gpu_memory_clock_mhz",
            # GPU VRAM & PCIe Bus Throughput
            "gpu_vram_used_mb", "gpu_vram_total_mb", "gpu_vram_percent",
            "pcie_tx_kbps", "pcie_rx_kbps",
            # Operational Stability & Throttle Reasons
            "gpu_throttle_reasons",
            # Battery & AC Power
            "battery_percent", "battery_rate_pct_hr", "is_ac_plugged"
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
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.start_time = time.monotonic()
        # Initialize psutil CPU percent (first call always returns 0.0)
        psutil.cpu_percent()
        try:
            self.prev_disk_io = psutil.disk_io_counters()
            self.prev_disk_time = 0.0
        except Exception:
            pass
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self._shutdown_nvml()

    def _monitor_loop(self):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)
            
            while not self.stop_event.is_set():
                try:
                    row = self._gather_metrics()
                    writer.writerow(row)
                    f.flush()
                except Exception as e:
                    print(f"Monitor loop error (ignored): {e}")
                
                self.stop_event.wait(self.interval)

    def _gather_metrics(self) -> list:
        now = datetime.now(timezone.utc).astimezone()
        elapsed = round(time.monotonic() - self.start_time, 2)
        
        # 1. CPU & System RAM & Swap
        cpu_pct = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_used_gb = round(ram.used / (1024**3), 2)
        ram_avail_gb = round(ram.available / (1024**3), 2)
        ram_pct = round(ram.percent, 1)
        
        swap_used_gb = ""
        swap_pct = ""
        try:
            swap = psutil.swap_memory()
            swap_used_gb = round(swap.used / (1024**3), 2)
            swap_pct = round(swap.percent, 1)
        except Exception:
            pass

        # 2. Disk I/O Throughput (MB/s)
        disk_read_mbs = ""
        disk_write_mbs = ""
        try:
            disk = psutil.disk_io_counters()
            if disk and self.prev_disk_io is not None and self.prev_disk_time is not None:
                dt_disk = max(0.001, elapsed - self.prev_disk_time)
                disk_read_mbs = round((disk.read_bytes - self.prev_disk_io.read_bytes) / (1024 * 1024 * dt_disk), 2)
                disk_write_mbs = round((disk.write_bytes - self.prev_disk_io.write_bytes) / (1024 * 1024 * dt_disk), 2)
            else:
                disk_read_mbs = 0.0
                disk_write_mbs = 0.0
            self.prev_disk_io = disk
            self.prev_disk_time = elapsed
        except Exception:
            pass

        # 3. Battery & AC Power Logic
        battery = psutil.sensors_battery()
        bat_pct, is_ac, bat_rate = "", "", ""
        
        if battery:
            bat_pct = round(battery.percent, 1)
            is_ac = bool(battery.power_plugged)
            
            # Estimate discharge rate (% / hour) when on DC battery
            if not is_ac and self.prev_bat_pct is not None and self.prev_bat_time is not None:
                dt_bat = elapsed - self.prev_bat_time
                if dt_bat >= 10.0:  # suppress high-frequency noise
                    rate = ((self.prev_bat_pct - bat_pct) / dt_bat) * 3600.0
                    bat_rate = round(max(0.0, rate), 2)
            elif is_ac:
                bat_rate = 0.0
                
            self.prev_bat_pct = bat_pct
            self.prev_bat_time = elapsed
            
            # --- FAIL SAFE LOGIC TRIGGERS ---
            if not is_ac:
                if bat_pct < self.battery_abort_threshold:
                    self.abort_event.set()
                    self.pause_event.clear()  # Unpause to allow immediate graceful shutdown
                else:
                    self.pause_event.set()
            else:
                self.pause_event.clear()  # AC restored, clear pause

        # 4. GPU Metrics (NVML)
        g_util = g_mem_util = g_temp = g_pow = g_pow_lim = ""
        g_mem_use = g_mem_tot = g_mem_pct = g_clk = g_mclk = ""
        pcie_tx = pcie_rx = ""
        g_throttle = ""
        
        if self.nvml_initialized and self.gpu_handle:
            import pynvml
            try:
                utils = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                g_util = utils.gpu
                g_mem_util = utils.memory
            except Exception: pass
            
            try:
                g_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception: pass

            try:
                g_pow = round(pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0, 2)
            except Exception: pass

            try:
                g_pow_lim = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(self.gpu_handle) / 1000.0, 2)
            except Exception: pass

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                g_mem_use = round(mem.used / (1024**2), 2)
                g_mem_tot = round(mem.total / (1024**2), 2)
                g_mem_pct = round((mem.used / mem.total) * 100, 1) if mem.total > 0 else 0
            except Exception: pass
            
            try:
                g_clk = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
            except Exception: pass

            try:
                g_mclk = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
            except Exception: pass

            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(self.gpu_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(self.gpu_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            except Exception: pass

            try:
                mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.gpu_handle)
                g_throttle = decode_throttle_reasons(mask)
            except Exception: pass

        return [
            now.isoformat(), elapsed,
            cpu_pct, ram_used_gb, ram_avail_gb, ram_pct,
            swap_used_gb, swap_pct,
            disk_read_mbs, disk_write_mbs,
            g_util, g_mem_util, g_temp,
            g_pow, g_pow_lim, g_clk, g_mclk,
            g_mem_use, g_mem_tot, g_mem_pct,
            pcie_tx, pcie_rx,
            g_throttle,
            bat_pct, bat_rate, is_ac
        ]
