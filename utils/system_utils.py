"""System utility functions for BitNet.

Provides helpers for detecting hardware capabilities, available memory,
and determining optimal inference settings for the current system.
"""

import os
import platform
import subprocess
import sys
from typing import Dict, Optional, Tuple


def get_cpu_info() -> Dict[str, str]:
    """Retrieve basic CPU information.

    Returns:
        dict with keys: 'name', 'arch', 'cores', 'threads'
    """
    info = {
        "name": "Unknown",
        "arch": platform.machine(),
        "cores": str(os.cpu_count() or 1),
        "threads": str(os.cpu_count() or 1),
    }

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["name"] = line.split(":", 1)[1].strip()
                        break
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["name"] = result.stdout.strip()
        elif system == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name", "/value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if line.startswith("Name="):
                    info["name"] = line.split("=", 1)[1].strip()
                    break
    except Exception:
        pass  # Fall back to defaults

    return info


def get_available_memory_gb() -> float:
    """Return available system RAM in gigabytes."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        elif system == "Darwin":
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5
            )
            pages_free = 0
            page_size = 4096
            for line in result.stdout.splitlines():
                if "Pages free" in line:
                    pages_free = int(line.split(":")[1].strip().rstrip("."))
                elif "page size of" in line:
                    page_size = int(line.split("page size of")[1].split()[0])
            return (pages_free * page_size) / (1024 ** 3)
    except Exception:
        pass

    return 4.0  # Conservative fallback


def get_total_memory_gb() -> float:
    """Return total system RAM in gigabytes."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass

    return 8.0  # Conservative fallback


def get_optimal_thread_count(reserved_cores: int = 1) -> int:
    """Suggest an optimal thread count for inference.

    Args:
        reserved_cores: Number of CPU cores to reserve for the OS/other tasks.

    Returns:
        Recommended thread count (at least 1).
    """
    total = os.cpu_count() or 1
    return max(1, total - reserved_cores)


def can_fit_model(model_size_gb: float, safety_margin: float = 0.15) -> bool:
    """Check whether the model can fit in available system memory.

    Args:
        model_size_gb: Estimated size of the model in GB.
        safety_margin: Fraction of total memory to keep free (default 15%).

    Returns:
        True if the model is likely to fit in memory.
    """
    available = get_available_memory_gb()
    total = get_total_memory_gb()
    headroom = total * safety_margin
    return model_size_gb <= (available - headroom)


def get_system_summary() -> Dict[str, str]:
    """Return a human-readable summary of the current system."""
    cpu = get_cpu_info()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cpu_name": cpu["name"],
        "cpu_arch": cpu["arch"],
        "cpu_cores": cpu["cores"],
        "total_memory_gb": f"{get_total_memory_gb():.1f}",
        "available_memory_gb": f"{get_available_memory_gb():.1f}",
        "optimal_threads": str(get_optimal_thread_count()),
    }
