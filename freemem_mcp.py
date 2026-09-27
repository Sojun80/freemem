#!/usr/bin/env python3
"""MCP server for local freemem, gpuwatch, and guarded GPU controls.

The server returns structured values so an AI host can answer questions about
the machine without parsing the human-oriented shell output. It uses
nvidia-smi for NVIDIA telemetry, nvidia-settings for overclock offsets, and
/proc plus ps for the freemem view. GPU writes are delegated to gpuoc, which
keeps the CLI and MCP safety rules in one place.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server import MCPServer


mcp = MCPServer(
    "freemem-gpuwatch",
    version="0.3.0",
    instructions=(
        "Use host_stats for a quick read-only CPU, RAM, root-disk, and GPU snapshot. "
        "Use gpu_stats for detailed GPU temperatures, power, VRAM, clocks, fans, "
        "P-state, and overclock offsets. Use gpu_apply_overclock or "
        "gpu_reset_overclock for guarded NVIDIA changes; both require confirm=true "
        "and never infer confirmation from surrounding conversation. Use memory_stats "
        "for RAM and top RSS processes. Values are live snapshots; null means the "
        "host or driver does not expose a field."
    ),
)

GPUOC_PATH = Path(__file__).with_name("gpuoc")

GPU_QUERY_FIELDS = [
    "index",
    "name",
    "pci.bus_id",
    "temperature.gpu",
    "temperature.gpu.tlimit",
    "temperature.memory",
    "power.draw",
    "power.limit",
    "enforced.power.limit",
    "power.default_limit",
    "power.min_limit",
    "power.max_limit",
    "memory.used",
    "memory.total",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.graphics",
    "clocks.max.sm",
    "clocks.max.memory",
    "fan.speed",
    "pstate",
    "clocks_event_reasons.active",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(command: list[str], timeout: float = 4.0) -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return "", f"{command[0]} is not installed", 127
    except subprocess.TimeoutExpired:
        return "", f"{command[0]} timed out", 124
    except OSError as exc:
        return "", str(exc), 1
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def _run_gpuoc(arguments: list[str], timeout: float = 12.0) -> tuple[str, str, int]:
    """Run the local guarded GPU control CLI with a small, fixed argument list."""

    if not GPUOC_PATH.is_file() or not os.access(GPUOC_PATH, os.X_OK):
        return "", f"{GPUOC_PATH} is missing or not executable", 127
    return _run([str(GPUOC_PATH), *arguments], timeout=timeout)


def _control_number(value: Any, name: str, integer: bool = False) -> int | float | None:
    """Validate an MCP numeric control and return it in a shell-safe form."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        expected = "an integer" if integer else "a number"
        raise ValueError(f"{name} must be {expected}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if integer and not float(value).is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else value


def _control_text(value: int | float) -> str:
    if isinstance(value, float):
        return format(value, ".12f").rstrip("0").rstrip(".")
    return str(value)


def _number(value: str | None, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() in {"N/A", "NA", "NOT SUPPORTED", "[NOT SUPPORTED]"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if integer or parsed.is_integer():
        return int(parsed)
    return parsed


def _short_model(name: str) -> str:
    model = name.strip()
    if model.startswith("NVIDIA "):
        model = model[len("NVIDIA ") :]
    if model.startswith("GeForce "):
        model = model[len("GeForce ") :]
    return model


def _pci_bus_number(bus_id: str) -> int | None:
    match = re.search(r":([0-9a-fA-F]{2}):", bus_id)
    return int(match.group(1), 16) if match else None


def _settings_value(settings_gpu: int, attribute: str) -> int | float | None:
    output, _, status = _run(
        ["nvidia-settings", "-q", f"[gpu:{settings_gpu}]/{attribute}", "-t"],
        timeout=2,
    )
    if status != 0:
        return None
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    return _number(first_line)


@lru_cache(maxsize=1)
def _settings_gpu_by_bus() -> dict[int, int]:
    """Map nvidia-smi PCI bus numbers to nvidia-settings GPU indices."""

    if shutil.which("nvidia-settings") is None:
        return {}

    mapping: dict[int, int] = {}
    for settings_gpu in range(32):
        output, _, status = _run(
            ["nvidia-settings", "-q", f"[gpu:{settings_gpu}]/PCIBus", "-t"],
            timeout=2,
        )
        bus = _number(output.splitlines()[0] if output.splitlines() else None, integer=True)
        if status != 0 or bus is None:
            if settings_gpu > 0:
                break
            continue
        mapping[int(bus)] = settings_gpu
    return mapping


def _overclock_settings(pci_bus_id: str) -> dict[str, int | float | None]:
    bus = _pci_bus_number(pci_bus_id)
    settings_gpu = _settings_gpu_by_bus().get(bus) if bus is not None else None
    if settings_gpu is None:
        return {
            "available": False,
            "core_offset_mhz": None,
            "memory_transfer_offset_mhz": None,
            "vtt_offset_mv": None,
        }

    voltage_offset_uv = _settings_value(settings_gpu, "GPUOverVoltageOffset")
    return {
        "available": True,
        "core_offset_mhz": _settings_value(settings_gpu, "GPUGraphicsClockOffset"),
        "memory_transfer_offset_mhz": _settings_value(
            settings_gpu, "GPUMemoryTransferRateOffset"
        ),
        "vtt_offset_mv": (
            round(voltage_offset_uv / 1000, 1)
            if isinstance(voltage_offset_uv, (int, float))
            else None
        ),
    }


def _gpu_snapshot(gpu_index: int | None = None) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {
            "timestamp": _now(),
            "available": False,
            "gpus": [],
            "error": "nvidia-smi is not installed",
        }

    output, error, status = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if status != 0:
        return {
            "timestamp": _now(),
            "available": False,
            "gpus": [],
            "error": error or "nvidia-smi could not read GPU telemetry",
        }

    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    gpus: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < len(GPU_QUERY_FIELDS):
            continue
        values = {field: value.strip() for field, value in zip(GPU_QUERY_FIELDS, row)}
        index = _number(values["index"], integer=True)
        if index is None or (gpu_index is not None and index != gpu_index):
            continue

        memory_used = _number(values["memory.used"], integer=True)
        memory_total = _number(values["memory.total"], integer=True)
        used_percent = None
        if isinstance(memory_used, (int, float)) and isinstance(memory_total, (int, float)) and memory_total:
            used_percent = round(memory_used / memory_total * 100, 1)

        gpus.append(
            {
                "index": index,
                "model": _short_model(values["name"]),
                "name": values["name"],
                "pci_bus_id": values["pci.bus_id"],
                "temperature_c": {
                    "gpu": _number(values["temperature.gpu"]),
                    "memory": _number(values["temperature.memory"]),
                    "gpu_limit": _number(values["temperature.gpu.tlimit"]),
                },
                "power_w": {
                    "draw": _number(values["power.draw"]),
                    "limit": _number(values["power.limit"]),
                    "enforced_limit": _number(values["enforced.power.limit"]),
                    "default_limit": _number(values["power.default_limit"]),
                    "min_limit": _number(values["power.min_limit"]),
                    "max_limit": _number(values["power.max_limit"]),
                },
                "vram_mib": {
                    "used": memory_used,
                    "total": memory_total,
                    "free": _number(values["memory.free"], integer=True),
                    "used_percent": used_percent,
                },
                "utilization_percent": {
                    "gpu": _number(values["utilization.gpu"]),
                    "memory": _number(values["utilization.memory"]),
                },
                "clocks_mhz": {
                    "current_graphics": _number(values["clocks.current.graphics"]),
                    "current_sm": _number(values["clocks.current.sm"]),
                    "current_memory": _number(values["clocks.current.memory"]),
                    "max_graphics": _number(values["clocks.max.graphics"]),
                    "max_sm": _number(values["clocks.max.sm"]),
                    "max_memory": _number(values["clocks.max.memory"]),
                },
                "overclock": _overclock_settings(values["pci.bus_id"]),
                "fan_percent": _number(values["fan.speed"]),
                "pstate": values["pstate"] if values["pstate"] not in {"", "N/A"} else None,
                "throttle_reasons": values["clocks_event_reasons.active"] or None,
            }
        )

    result: dict[str, Any] = {
        "timestamp": _now(),
        "available": True,
        "gpu_count": len(gpus),
        "gpus": gpus,
    }
    if gpu_index is not None and not gpus:
        result["error"] = f"GPU index {gpu_index} was not reported by nvidia-smi"
    return result


def _memory_snapshot(process_limit: int) -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, remainder = line.partition(":")
            match = re.search(r"(\d+)", remainder)
            if match:
                meminfo[key] = int(match.group(1))
    except OSError as exc:
        return {"timestamp": _now(), "available": False, "error": str(exc)}

    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable")
    if available is None:
        available = (
            meminfo.get("MemFree", 0)
            + meminfo.get("Buffers", 0)
            + meminfo.get("Cached", 0)
            + meminfo.get("SReclaimable", 0)
            - meminfo.get("Shmem", 0)
        )
    available = max(0, min(total, available))
    used = max(0, total - available)

    processes: list[dict[str, Any]] = []
    ps_output, _, ps_status = _run(["ps", "-eo", "pid=,user=,rss=,args="], timeout=3)
    if ps_status == 0:
        for line in ps_output.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            pid, user, rss, command = parts
            try:
                processes.append(
                    {
                        "pid": int(pid),
                        "user": user,
                        "rss_kib": int(rss),
                        "command": command,
                    }
                )
            except ValueError:
                continue
        processes.sort(key=lambda process: process["rss_kib"], reverse=True)
        processes = processes[:process_limit]

    return {
        "timestamp": _now(),
        "available": True,
        "ram_kib": {
            "total": total,
            "used": used,
            "available": available,
            "used_percent": round(used / total * 100, 1) if total else None,
            "available_percent": round(available / total * 100, 1) if total else None,
        },
        "top_processes_by_rss": processes,
        "process_limit": process_limit,
    }


def _cpu_snapshot() -> dict[str, Any]:
    """Read load averages and a short /proc/stat utilization sample."""

    result: dict[str, Any] = {
        "logical_cpus": os.cpu_count(),
        "load_average": {"1m": None, "5m": None, "15m": None},
        "usage_percent": None,
    }
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        result["load_average"] = {
            "1m": round(load_1m, 2),
            "5m": round(load_5m, 2),
            "15m": round(load_15m, 2),
        }
    except OSError:
        pass

    def read_cpu_counters() -> tuple[int, int] | None:
        try:
            first_line = Path("/proc/stat").read_text().splitlines()[0]
            fields = first_line.split()
            if fields[0] != "cpu" or len(fields) < 5:
                return None
            counters = [int(value) for value in fields[1:]]
            idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
            return sum(counters), idle
        except (IndexError, OSError, ValueError):
            return None

    first = read_cpu_counters()
    if first is not None:
        time.sleep(0.05)
        second = read_cpu_counters()
        if second is not None:
            total_delta = second[0] - first[0]
            idle_delta = second[1] - first[1]
            if total_delta > 0:
                result["usage_percent"] = round(
                    max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100)),
                    1,
                )
    return result


def _disk_snapshot() -> dict[str, Any]:
    """Read usage for the host root filesystem without walking mount points."""

    try:
        usage = shutil.disk_usage("/")
    except OSError as exc:
        return {"available": False, "path": "/", "error": str(exc)}
    return {
        "available": True,
        "path": "/",
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_percent": round(usage.used / usage.total * 100, 1) if usage.total else None,
    }


def _host_snapshot(process_limit: int) -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "cpu": _cpu_snapshot(),
        "memory": _memory_snapshot(process_limit),
        "disk": _disk_snapshot(),
        "gpu": _gpu_snapshot(),
    }


@mcp.tool()
def gpu_stats(gpu_index: int | None = None) -> dict[str, Any]:
    """Read current GPU temperatures, power, VRAM, clocks, fans, and overclock offsets.

    Values are numeric where supported and null when the driver or hardware
    does not expose a sensor. GPU utilization is activity; vram_mib.used_percent
    is capacity consumed. Pass gpu_index to inspect one nvidia-smi GPU.
    """

    if gpu_index is not None and gpu_index < 0:
        raise ValueError("gpu_index must be non-negative")
    return _gpu_snapshot(gpu_index)


@mcp.tool()
def gpu_apply_overclock(
    gpu_index: int,
    core_offset_mhz: int | None = None,
    memory_transfer_offset_mhz: int | None = None,
    power_limit_w: int | float | None = None,
    fan_percent: int | None = None,
    vtt_offset_mv: int | None = None,
    confirm: bool = False,
    unsafe: bool = False,
) -> dict[str, Any]:
    """Apply guarded NVIDIA GPU controls through the local gpuoc backend.

    Pass only the controls that should change; omitted controls are preserved.
    This tool never writes unless confirm is explicitly true. Core offsets are
    capped at +/-500 MHz, memory offsets at +/-3000 MHz, and VTT/GPU voltage
    offsets at +/-50 mV unless unsafe is explicitly true. The NVIDIA driver
    still enforces its own supported range, and power_limit_w is checked against
    each GPU's reported power range by the backend. VTT maps to NVIDIA's
    GPUOverVoltageOffset control and may be unavailable on newer GeForce
    drivers. Use gpu_stats first when the current state is not known.
    """

    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    core = _control_number(core_offset_mhz, "core_offset_mhz", integer=True)
    memory = _control_number(
        memory_transfer_offset_mhz,
        "memory_transfer_offset_mhz",
        integer=True,
    )
    power = _control_number(power_limit_w, "power_limit_w")
    fan = _control_number(fan_percent, "fan_percent", integer=True)
    vtt = _control_number(vtt_offset_mv, "vtt_offset_mv", integer=True)
    if core is None and memory is None and power is None and fan is None and vtt is None:
        raise ValueError(
            "provide at least one of core_offset_mhz, "
            "memory_transfer_offset_mhz, power_limit_w, fan_percent, or vtt_offset_mv"
        )
    if core is not None and not unsafe and abs(core) > 500:
        raise ValueError("core_offset_mhz is capped at +/-500 unless unsafe=true")
    if memory is not None and not unsafe and abs(memory) > 3000:
        raise ValueError(
            "memory_transfer_offset_mhz is capped at +/-3000 unless unsafe=true"
        )
    if vtt is not None and not unsafe and abs(vtt) > 50:
        raise ValueError("vtt_offset_mv is capped at +/-50 unless unsafe=true")
    if fan is not None and not 20 <= fan <= 100:
        raise ValueError("fan_percent must be between 20 and 100")

    requested = {
        "gpu_index": gpu_index,
        "core_offset_mhz": core,
        "memory_transfer_offset_mhz": memory,
        "power_limit_w": power,
        "fan_percent": fan,
        "vtt_offset_mv": vtt,
        "unsafe": unsafe,
    }
    if confirm is not True:
        return {
            "timestamp": _now(),
            "available": True,
            "applied": False,
            "confirmation_required": True,
            "requested": requested,
            "message": "Set confirm=true to authorize the GPU write.",
        }

    arguments = ["apply", "--gpu", str(gpu_index)]
    if core is not None:
        arguments += ["--core", str(core)]
    if memory is not None:
        arguments += ["--memory", str(memory)]
    if power is not None:
        arguments += ["--power-limit", _control_text(power)]
    if fan is not None:
        arguments += ["--fan", str(fan)]
    if vtt is not None:
        arguments += ["--vtt", str(vtt)]
    if unsafe:
        arguments.append("--unsafe")
    arguments.append("--yes")

    output, error, status = _run_gpuoc(arguments)
    result: dict[str, Any] = {
        "timestamp": _now(),
        "available": status != 127,
        "applied": status == 0,
        "requested": requested,
        "output": output or None,
    }
    if status != 0:
        result["error"] = error or output or "gpuoc failed to apply the requested controls"
        return result
    result["snapshot"] = _gpu_snapshot(gpu_index)
    return result


@mcp.tool()
def gpu_reset_overclock(gpu_index: int, confirm: bool = False) -> dict[str, Any]:
    """Reset one NVIDIA GPU to zero offsets, automatic fan, and default power.

    This tool never writes unless confirm is explicitly true. It affects only
    the GPU identified by gpu_index and delegates the reset to gpuoc.
    """

    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    requested = {"gpu_index": gpu_index}
    if confirm is not True:
        return {
            "timestamp": _now(),
            "available": True,
            "applied": False,
            "confirmation_required": True,
            "requested": requested,
            "message": "Set confirm=true to authorize the GPU reset.",
        }

    output, error, status = _run_gpuoc(
        ["reset", "--gpu", str(gpu_index), "--yes"]
    )
    result: dict[str, Any] = {
        "timestamp": _now(),
        "available": status != 127,
        "applied": status == 0,
        "requested": requested,
        "output": output or None,
    }
    if status != 0:
        result["error"] = error or output or "gpuoc failed to reset the GPU"
        return result
    result["snapshot"] = _gpu_snapshot(gpu_index)
    return result


@mcp.tool()
def memory_stats(process_limit: int = 15) -> dict[str, Any]:
    """Read RAM totals and the largest processes by resident memory (RSS)."""

    if process_limit < 0 or process_limit > 100:
        raise ValueError("process_limit must be between 0 and 100")
    return _memory_snapshot(process_limit)


@mcp.tool()
def system_snapshot(process_limit: int = 15) -> dict[str, Any]:
    """Read a fresh combined CPU, RAM, root-disk, GPU, and top-process snapshot."""

    if process_limit < 0 or process_limit > 100:
        raise ValueError("process_limit must be between 0 and 100")
    return _host_snapshot(process_limit)


@mcp.tool()
def host_stats(process_limit: int = 5) -> dict[str, Any]:
    """Read one quick, read-only host snapshot: CPU load, RAM, root disk, and GPU stats.

    This is the preferred tool for a general machine-health check. It includes
    top RAM-consuming processes, current GPU temperatures, power, VRAM, clocks,
    fan speed, P-state, and overclock offsets. Values are numeric where
    supported and null when the host does not expose them.
    """

    if process_limit < 0 or process_limit > 100:
        raise ValueError("process_limit must be between 0 and 100")
    return _host_snapshot(process_limit)


if __name__ == "__main__":
    # MCP owns stdout for the stdio protocol; do not print anything here.
    mcp.run()
