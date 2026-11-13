import csv
import json
from pathlib import Path

def parse_float(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def generate_final_report(metrics_csv: Path, report_json: Path, run_metadata: dict):
    metrics_csv = Path(metrics_csv)
    report_json = Path(report_json)
    if not metrics_csv.exists():
        return
    
    stats = {
        "cpu_pct": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "ram_gb": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "ram_pct": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "swap_gb": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "swap_pct": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "disk_read": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "disk_write": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_util": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_mem_util": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_temp": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_power": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_clk": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_vram": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "pcie_tx": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "pcie_rx": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "bat_pct": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "bat_rate": {"sum": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
    }
    
    throttle_reasons_set = set()
    throttle_sample_count = 0
    total_samples = 0
    total_duration_sec = 0.0
    
    # Energy calculation variables (trapezoidal integration of power over time)
    energy_joules = 0.0
    prev_time = None
    prev_power = None
    
    # Battery state tracking
    ac_disconnect_count = 0
    prev_is_ac = None

    with open(metrics_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_samples += 1
            
            def update_stat(key, row_key):
                val = parse_float(row.get(row_key))
                if val is not None:
                    stats[key]["sum"] += val
                    stats[key]["count"] += 1
                    stats[key]["min"] = min(stats[key]["min"], val)
                    stats[key]["max"] = max(stats[key]["max"], val)
                    return val
                return None

            update_stat("cpu_pct", "cpu_percent")
            update_stat("ram_gb", "ram_used_gb")
            update_stat("ram_pct", "ram_percent")
            update_stat("swap_gb", "swap_used_gb")
            update_stat("swap_pct", "swap_percent")
            update_stat("disk_read", "disk_read_mbs")
            update_stat("disk_write", "disk_write_mbs")
            
            update_stat("gpu_util", "gpu_util_percent")
            update_stat("gpu_mem_util", "gpu_mem_util_percent")
            update_stat("gpu_temp", "gpu_temp_c")
            current_power = update_stat("gpu_power", "gpu_power_w")
            update_stat("gpu_clk", "gpu_clock_mhz")
            update_stat("gpu_vram", "gpu_vram_used_mb")
            update_stat("pcie_tx", "pcie_tx_kbps")
            update_stat("pcie_rx", "pcie_rx_kbps")
            
            update_stat("bat_pct", "battery_percent")
            update_stat("bat_rate", "battery_rate_pct_hr")
            
            # AC status & transitions
            is_ac_val = row.get("is_ac_plugged")
            if is_ac_val is not None and is_ac_val != "":
                is_ac = (str(is_ac_val).lower() == "true")
                if prev_is_ac is True and not is_ac:
                    ac_disconnect_count += 1
                prev_is_ac = is_ac

            # Throttle reasons tracking
            throttle_str = row.get("gpu_throttle_reasons", "")
            if throttle_str and throttle_str.strip() not in ("None", ""):
                throttle_sample_count += 1
                for reason in throttle_str.split("|"):
                    cleaned = reason.strip()
                    if cleaned and cleaned != "None":
                        throttle_reasons_set.add(cleaned)

            # Energy trapezoidal integration
            current_time = parse_float(row.get("elapsed_seconds"))
            if current_time is not None:
                total_duration_sec = max(total_duration_sec, current_time)
                if current_power is not None:
                    if prev_time is not None and prev_power is not None:
                        dt = max(0.0, current_time - prev_time)
                        energy_joules += ((prev_power + current_power) / 2.0) * dt
                    prev_time = current_time
                    prev_power = current_power

    def get_avg(key):
        return round(stats[key]["sum"] / stats[key]["count"], 2) if stats[key]["count"] > 0 else None

    def get_min(key):
        val = stats[key]["min"]
        return round(val, 2) if val != float('inf') else None

    def get_max(key):
        val = stats[key]["max"]
        return round(val, 2) if val != float('-inf') else None

    # Calculate Energy-per-Sample / Energy-per-Frame
    total_work_units = run_metadata.get("total_samples") or run_metadata.get("total_frames")
    energy_per_sample_j = None
    energy_per_sample_mj = None
    if total_work_units and total_work_units > 0 and energy_joules > 0:
        energy_per_sample_j = round(energy_joules / total_work_units, 4)
        energy_per_sample_mj = round((energy_joules / total_work_units) * 1000.0, 2)

    report = {
        "run_status": run_metadata.get("status", "unknown"),
        "run_information": run_metadata,
        "cpu": {
            "average_usage_percent": get_avg("cpu_pct"),
            "minimum_usage_percent": get_min("cpu_pct"),
            "maximum_usage_percent": get_max("cpu_pct")
        },
        "ram": {
            "average_used_gb": get_avg("ram_gb"),
            "maximum_used_gb": get_max("ram_gb"),
            "maximum_percent": get_max("ram_pct"),
            "maximum_swap_used_gb": get_max("swap_gb"),
            "maximum_swap_percent": get_max("swap_pct")
        },
        "disk_io": {
            "average_read_mbs": get_avg("disk_read"),
            "peak_read_mbs": get_max("disk_read"),
            "average_write_mbs": get_avg("disk_write"),
            "peak_write_mbs": get_max("disk_write")
        },
        "gpu": {
            "compute_core": {
                "average_utilization_percent": get_avg("gpu_util"),
                "minimum_utilization_percent": get_min("gpu_util"),
                "maximum_utilization_percent": get_max("gpu_util")
            },
            "memory_controller": {
                "average_utilization_percent": get_avg("gpu_mem_util"),
                "maximum_utilization_percent": get_max("gpu_mem_util"),
                "architecture_note": "High memory controller utilization relative to compute core utilization indicates memory-bandwidth boundedness (typical of depthwise-separable MobileNet convolutions)."
            },
            "thermals": {
                "average_temperature_c": get_avg("gpu_temp"),
                "minimum_temperature_c": get_min("gpu_temp"),
                "maximum_temperature_c": get_max("gpu_temp")
            },
            "power_and_energy": {
                "average_power_w": get_avg("gpu_power"),
                "minimum_power_w": get_min("gpu_power"),
                "maximum_power_w": get_max("gpu_power"),
                "estimated_energy_joules": round(energy_joules, 2) if energy_joules > 0 else None,
                "estimated_energy_wh": round(energy_joules / 3600.0, 4) if energy_joules > 0 else None,
                "energy_per_sample_joules": energy_per_sample_j,
                "energy_per_sample_millijoules": energy_per_sample_mj
            },
            "vram": {
                "maximum_used_mb": get_max("gpu_vram")
            },
            "clocks": {
                "average_graphics_clock_mhz": get_avg("gpu_clk"),
                "peak_graphics_clock_mhz": get_max("gpu_clk")
            },
            "pcie_interconnect": {
                "peak_pcie_tx_kbps": get_max("pcie_tx"),
                "peak_pcie_rx_kbps": get_max("pcie_rx"),
                "average_pcie_tx_kbps": get_avg("pcie_tx"),
                "average_pcie_rx_kbps": get_avg("pcie_rx")
            },
            "stability_and_throttling": {
                "throttled_sample_count": throttle_sample_count,
                "throttle_ratio_percent": round((throttle_sample_count / total_samples) * 100.0, 1) if total_samples > 0 else 0.0,
                "observed_throttle_reasons": sorted(list(throttle_reasons_set)),
                "is_throttled": (throttle_sample_count > 0)
            }
        },
        "battery_and_ac": {
            "minimum_battery_percent": get_min("bat_pct"),
            "maximum_battery_percent": get_max("bat_pct"),
            "average_battery_percent": get_avg("bat_pct"),
            "max_discharge_rate_percent_per_hour": get_max("bat_rate"),
            "ac_disconnect_events": ac_disconnect_count
        },
        "monitoring": {
            "total_samples": total_samples,
            "total_monitored_seconds": round(total_duration_sec, 2)
        }
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
