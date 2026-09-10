#!/usr/bin/env python3
"""Read-only MCP server for the local freemem and gpuwatch telemetry.

The server deliberately returns structured values so an AI host can answer
questions about the machine without parsing the human-oriented shell output.
It uses nvidia-smi for NVIDIA telemetry, nvidia-settings for overclock offsets,
and /proc plus ps for the freemem view.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server import MCPServer


mcp = MCPServer("freemem-gpuwatch")

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
        }

    return {
        "available": True,
        "core_offset_mhz": _settings_value(settings_gpu, "GPUGraphicsClockOffset"),
        "memory_transfer_offset_mhz": _settings_value(
            settings_gpu, "GPUMemoryTransferRateOffset"
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
def memory_stats(process_limit: int = 15) -> dict[str, Any]:
    """Read RAM totals and the largest processes by resident memory (RSS)."""

    if process_limit < 0 or process_limit > 100:
        raise ValueError("process_limit must be between 0 and 100")
    return _memory_snapshot(process_limit)


@mcp.tool()
def system_snapshot(process_limit: int = 15) -> dict[str, Any]:
    """Read a fresh combined GPU, VRAM, RAM, and top-process snapshot."""

    if process_limit < 0 or process_limit > 100:
        raise ValueError("process_limit must be between 0 and 100")
    return {
        "timestamp": _now(),
        "gpu": _gpu_snapshot(),
        "memory": _memory_snapshot(process_limit),
    }


if __name__ == "__main__":
    # MCP owns stdout for the stdio protocol; do not print anything here.
    mcp.run()
