import csv
import json
from pathlib import Path

def parse_float(val, default=None):
    try: return float(val)
    except (ValueError, TypeError): return default

def generate_final_report(metrics_csv: Path, report_json: Path, run_metadata: dict):
    if not metrics_csv.exists():
        return
    
    stats = {
        "cpu_pct": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "ram_gb": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_util": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_temp": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
        "gpu_power": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
    }
    
    max_vram_mb = 0
    max_ram_pct = 0
    
    # Energy calculation variables
    energy_joules = 0.0
    prev_time = None
    prev_power = None
    
    rows_processed = 0

    # Stream the CSV line-by-line (O(1) memory)
    with open(metrics_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_processed += 1
            
            # Helper to update aggregates
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
            update_stat("gpu_util", "gpu_util_percent")
            update_stat("gpu_temp", "gpu_temp_c")
            current_power = update_stat("gpu_power", "gpu_power_w")
            
            # Track peak maximums
            vram = parse_float(row.get("gpu_vram_used_mb"))
            if vram and vram > max_vram_mb: max_vram_mb = vram
                
            ram_pct = parse_float(row.get("ram_percent"))
            if ram_pct and ram_pct > max_ram_pct: max_ram_pct = ram_pct

            # Calculate Energy (Trapezoidal integration)
            current_time = parse_float(row.get("elapsed_seconds"))
            if current_time is not None and current_power is not None:
                if prev_time is not None and prev_power is not None:
                    dt = current_time - prev_time
                    # (P1 + P2) / 2 * dt (Watts * seconds = Joules)
                    energy_joules += ((prev_power + current_power) / 2.0) * dt
                
                prev_time = current_time
                prev_power = current_power

    # Build final structure
    def get_avg(key):
        return round(stats[key]["sum"] / stats[key]["count"], 2) if stats[key]["count"] > 0 else None

    def get_min_max(key, stat_type):
        val = stats[key][stat_type]
        return round(val, 2) if val not in (float('inf'), float('-inf')) else None

    report = {
        "run_status": run_metadata.get("status", "unknown"),
        "run_information": run_metadata,
        "cpu": {
            "average_usage_percent": get_avg("cpu_pct"),
            "minimum_usage_percent": get_min_max("cpu_pct", "min"),
            "maximum_usage_percent": get_min_max("cpu_pct", "max")
        },
        "ram": {
            "average_used_gb": get_avg("ram_gb"),
            "maximum_used_gb": get_min_max("ram_gb", "max"),
            "maximum_percent": max_ram_pct
        },
        "gpu": {
            "average_utilization_percent": get_avg("gpu_util"),
            "minimum_utilization_percent": get_min_max("gpu_util", "min"),
            "maximum_utilization_percent": get_min_max("gpu_util", "max"),
            "average_temperature_c": get_avg("gpu_temp"),
            "minimum_temperature_c": get_min_max("gpu_temp", "min"),
            "maximum_temperature_c": get_min_max("gpu_temp", "max"),
            "average_power_w": get_avg("gpu_power"),
            "minimum_power_w": get_min_max("gpu_power", "min"),
            "maximum_power_w": get_min_max("gpu_power", "max"),
            "estimated_energy_wh": round(energy_joules / 3600.0, 4) if energy_joules > 0 else None,
            "maximum_vram_used_mb": max_vram_mb
        },
        "monitoring": {
            "total_samples": rows_processed
        }
    }

    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)