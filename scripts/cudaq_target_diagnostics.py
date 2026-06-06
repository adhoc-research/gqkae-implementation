#!/usr/bin/env python
"""Discover locally usable CUDA-Q CPU/GPU targets for H4 feasibility work."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _try_import_cudaq():
    try:
        import cudaq  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, str(exc)
    return cudaq, None


def _nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "error": "nvidia-smi not found"}
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mib": parts[3],
                }
            )
    return {"available": bool(gpus), "gpus": gpus, "raw": out.strip()}


def _bell_kernel(cudaq):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.mz(q)
    return kernel


def _resource_dict(resources) -> dict[str, Any]:
    if hasattr(resources, "to_dict"):
        try:
            return dict(resources.to_dict())
        except Exception:
            pass
    out: dict[str, Any] = {}
    for attr in ("depth", "multi_qubit_gate_count", "num_qubits", "num_used_qubits"):
        if hasattr(resources, attr):
            value = getattr(resources, attr)
            try:
                out[attr] = value() if callable(value) else value
            except Exception as exc:
                out[f"{attr}_error"] = str(exc)
    return out


def _candidate_targets() -> list[dict[str, str | None | bool]]:
    # CUDA-Q target names/options are version-dependent. Try conservative/common names first.
    return [
        {"label": "qpp-cpu", "target": "qpp-cpu", "option": None, "gpu_candidate": False},
        {"label": "nvidia", "target": "nvidia", "option": None, "gpu_candidate": True},
        {"label": "nvidia-fp64", "target": "nvidia", "option": "fp64", "gpu_candidate": True},
        {"label": "nvidia-fp32", "target": "nvidia", "option": "fp32", "gpu_candidate": True},
        {"label": "qpp-cuda", "target": "qpp-cuda", "option": None, "gpu_candidate": True},
        {"label": "custatevec", "target": "custatevec", "option": None, "gpu_candidate": True},
    ]


def probe_targets(shots: int = 100) -> dict[str, Any]:
    cudaq, import_error = _try_import_cudaq()
    payload: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cudaq_import_error": import_error,
        "cudaq_version": None,
        "has_set_random_seed": False,
        "nvidia_smi": _nvidia_smi(),
        "candidates": [],
        "selected_gpu": None,
    }
    if cudaq is None:
        return payload

    payload["cudaq_version"] = getattr(cudaq, "__version__", None)
    payload["has_set_random_seed"] = bool(hasattr(cudaq, "set_random_seed"))

    for cand in _candidate_targets():
        row: dict[str, Any] = dict(cand)
        t0 = time.perf_counter()
        try:
            if cand["option"]:
                cudaq.set_target(str(cand["target"]), option=str(cand["option"]))
            else:
                cudaq.set_target(str(cand["target"]))
            if hasattr(cudaq, "set_random_seed"):
                cudaq.set_random_seed(1234)
            kernel = _bell_kernel(cudaq)
            sample = cudaq.sample(kernel, shots_count=int(shots))
            counts = {str(k): int(v) for k, v in sample.items()}
            try:
                resources = _resource_dict(cudaq.estimate_resources(kernel))
            except Exception as exc:
                resources = {"error": str(exc)}
            row.update(
                {
                    "usable": True,
                    "elapsed_s": time.perf_counter() - t0,
                    "counts": counts,
                    "resources": resources,
                }
            )
        except Exception as exc:
            row.update({"usable": False, "elapsed_s": time.perf_counter() - t0, "error": str(exc)})
        payload["candidates"].append(row)

    for row in payload["candidates"]:
        if row.get("usable") and row.get("gpu_candidate"):
            payload["selected_gpu"] = {
                "label": row["label"],
                "target": row["target"],
                "option": row.get("option"),
            }
            break
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--output", default="runs/cudaq_target_diagnostics.json")
    args = p.parse_args()

    payload = probe_targets(shots=args.shots)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"CUDA-Q version: {payload.get('cudaq_version')}")
    print(f"nvidia-smi available: {payload.get('nvidia_smi', {}).get('available')}")
    for row in payload.get("candidates", []):
        status = "OK" if row.get("usable") else "FAIL"
        opt = f" option={row.get('option')}" if row.get("option") else ""
        print(f"[{status}] {row.get('label')} target={row.get('target')}{opt} elapsed={row.get('elapsed_s'):.3f}s")
        if not row.get("usable"):
            print(f"      {row.get('error')}")
    print(f"selected_gpu: {payload.get('selected_gpu')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
