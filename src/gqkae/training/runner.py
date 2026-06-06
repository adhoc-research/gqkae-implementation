"""End-to-end H4 MVP training loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..chemistry import build_h4_problem
from ..circuits import sample_sequence_circuit
from ..config import Config, apply_overrides, as_dict, load_config
from ..fermion_mapping import excitation_pauli_terms
from ..operator_pool import build_operator_pool
from ..qsci import qsci_energy_from_counts


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("PyTorch is required for training. Install with `uv sync`.") from exc
    return torch


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def _sequence_composition(sequence: list[int] | None, pool) -> dict[str, int]:
    if sequence is None:
        return {}
    out = {"noop": 0, "single": 0, "double": 0, "pauli_evolution": 0, "other": 0}
    for token in sequence:
        op = pool[int(token)]
        if op.kind == "pauli_evolution":
            out["pauli_evolution"] += 1
        elif op.is_noop:
            out["noop"] += 1
        elif op.rank == 1:
            out["single"] += 1
        elif op.rank == 2:
            out["double"] += 1
        else:
            out["other"] += 1
    return out


def _paper_table_count_view(gate_count: dict[str, Any]) -> dict[str, float]:
    """Return the cited Table-I count view, excluding HF-reference X gates."""
    x_count = float(gate_count.get("x", 0) or 0)
    total = float(gate_count.get("total", 0) or 0)
    return {
        "two_qubit": float(gate_count.get("two_qubit", gate_count.get("cx", 0)) or 0),
        "total_excluding_hf_x": total - x_count,
        "total_including_hf_x": total,
        "hf_x": x_count,
    }


def _paper_table_delta(gate_count: dict[str, Any]) -> dict[str, float]:
    view = _paper_table_count_view(gate_count)
    return {
        "two_qubit_vs_h4_gqkae_mean": float(view["two_qubit"] - 100.0),
        "total_excluding_hf_x_vs_h4_gqkae_mean": float(view["total_excluding_hf_x"] - 314.0),
        "total_including_hf_x_vs_h4_gqkae_mean": float(view["total_including_hf_x"] - 314.0),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _pool_manifest(pool, n_qubits: int) -> dict[str, Any]:
    operators: list[dict[str, Any]] = []
    for op in pool.operators:
        terms = [
            {"coefficient": float(term.coefficient), "pauli_word": "".join(term.word)}
            for term in excitation_pauli_terms(op, n_qubits)
        ]
        operators.append(
            {
                "token": int(op.token),
                "name": op.name,
                "kind": op.kind,
                "angle": float(op.angle),
                "annihilate": list(op.annihilate),
                "create": list(op.create),
                "parent": op.parent,
                "pauli_terms": terms,
            }
        )
    canonical = json.dumps(operators, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "operators": operators,
    }


def _timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "total_s": 0.0, "mean_s": 0.0, "std_s": 0.0, "min_s": 0.0, "max_s": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": float(len(values)),
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=0)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def train(config: Config) -> dict[str, Any]:
    torch = _require_torch()
    from ..models.hqkansformer import HQKANsformerPolicy, count_parameters, parameter_memory_mb
    from .grpo import grpo_loss, normalize_advantages

    train_wall_start = time.perf_counter()
    detailed_timing = os.environ.get("GQKAE_DETAILED_TIMING", "").lower() in {"1", "true", "yes", "on"}
    rng = set_seed(config.experiment.seed)
    out_dir = Path(config.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_start = time.perf_counter()
    problem = build_h4_problem(config.molecule)
    pool = build_operator_pool(problem, config.operator_pool)
    setup_elapsed_s = time.perf_counter() - setup_start
    device = torch.device(config.training.device)
    model = HQKANsformerPolicy(pool.vocab_size, pool.sequence_length, config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best = {
        "energy": float("inf"),
        "error": float("inf"),
        "sequence": None,
        "iteration": -1,
        "gate_count": {"two_qubit": 0, "total": 0},
    }

    for iteration in range(config.training.iterations):
        iteration_start = time.perf_counter()
        model.eval()
        generation_start = time.perf_counter()
        with torch.no_grad():
            sequences_t, old_log_probs_t = model.sample(
                batch_size=config.training.batch_circuits,
                temperature=config.training.temperature,
                repetition_penalty=config.training.repetition_penalty,
                device=device,
            )
        sequences = sequences_t.detach().cpu().numpy()
        generation_elapsed_s = time.perf_counter() - generation_start

        energies: list[float] = []
        subdims: list[int] = []
        gate_counts: list[dict[str, int]] = []
        circuit_eval_times_s: list[float] = []
        circuit_sample_times_s: list[float] = []
        circuit_qsci_times_s: list[float] = []
        detailed_circuit_timings: list[dict[str, float | int]] = []
        evaluation_start = time.perf_counter()
        for circuit_index, seq in enumerate(sequences):
            circuit_start = time.perf_counter()
            sample_start = time.perf_counter()
            sample = sample_sequence_circuit(
                seq,
                pool,
                problem.basis,
                shots=config.qsci.shots,
                rng=rng,
                backend=config.qsci.backend,
                cudaq_target=config.qsci.cudaq_target,
                cudaq_option=config.qsci.cudaq_option,
                cudaq_seed=config.experiment.seed + iteration,
                cudaq_reverse_bitstrings=config.qsci.cudaq_reverse_bitstrings,
            )
            sample_elapsed_s = time.perf_counter() - sample_start
            qsci_start = time.perf_counter()
            result = qsci_energy_from_counts(
                sample.counts,
                problem,
                dmax=config.qsci.dmax,
                add_hf_det=config.qsci.add_hf_det,
                enlarge_method=config.qsci.enlarge_method,
            )
            qsci_elapsed_s = time.perf_counter() - qsci_start
            circuit_elapsed_s = time.perf_counter() - circuit_start
            circuit_eval_times_s.append(circuit_elapsed_s)
            circuit_sample_times_s.append(sample_elapsed_s)
            circuit_qsci_times_s.append(qsci_elapsed_s)
            if detailed_timing:
                detailed_circuit_timings.append(
                    {
                        "circuit_index": int(circuit_index),
                        "sample_elapsed_s": float(sample_elapsed_s),
                        "qsci_elapsed_s": float(qsci_elapsed_s),
                        "total_elapsed_s": float(circuit_elapsed_s),
                    }
                )
            energies.append(result.energy)
            subdims.append(result.subspace_dimension)
            gate_counts.append(sample.gate_count)
            error = result.energy - problem.casci_energy
            if result.energy < best["energy"]:
                best = {
                    "energy": float(result.energy),
                    "error": float(error),
                    "sequence": [int(x) for x in seq.tolist()],
                    "iteration": iteration,
                    "gate_count": sample.gate_count,
                }

        evaluation_elapsed_s = time.perf_counter() - evaluation_start
        rewards_t = torch.tensor([-e for e in energies], dtype=torch.float32, device=device)
        advantages_t = normalize_advantages(rewards_t)

        model.train()
        losses: list[float] = []
        optimization_start = time.perf_counter()
        for _ in range(config.training.policy_updates):
            optimizer.zero_grad(set_to_none=True)
            new_log_probs = model.log_probs_for_sequences(sequences_t)
            loss = grpo_loss(
                new_log_probs,
                old_log_probs_t,
                advantages_t,
                clip_epsilon=config.training.grpo_clip,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        optimization_elapsed_s = time.perf_counter() - optimization_start
        iteration_elapsed_s = time.perf_counter() - iteration_start

        entry = {
            "iteration": iteration,
            "energies": [float(e) for e in energies],
            "mean_energy": float(np.mean(energies)),
            "best_batch_energy": float(np.min(energies)),
            "best_so_far_energy": float(best["energy"]),
            "best_so_far_error": float(best["error"]),
            "mean_subspace_dimension": float(np.mean(subdims)),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_total_gates": float(np.mean([g.get("total", 0) for g in gate_counts])),
            "mean_two_qubit_gates": float(np.mean([g.get("two_qubit", 0) for g in gate_counts])),
            "timing_s": {
                "generation": float(generation_elapsed_s),
                "evaluation": float(evaluation_elapsed_s),
                "optimization": float(optimization_elapsed_s),
                "iteration_total": float(iteration_elapsed_s),
                "per_circuit_evaluation": _timing_stats(circuit_eval_times_s),
                "per_circuit_sample": _timing_stats(circuit_sample_times_s),
                "per_circuit_qsci": _timing_stats(circuit_qsci_times_s),
            },
        }
        if detailed_timing:
            entry["circuit_timings_s"] = detailed_circuit_timings
        history.append(entry)
        print(
            f"iter={iteration:03d} mean_E={entry['mean_energy']:.10f} "
            f"best_err={entry['best_so_far_error']:.6e} loss={entry['mean_loss']:.4e} "
            f"elapsed={iteration_elapsed_s:.2f}s eval={evaluation_elapsed_s:.2f}s opt={optimization_elapsed_s:.2f}s",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        (out_dir / "best.json").write_text(json.dumps(best, indent=2))

    final_entry = history[-1] if history else {}
    train_elapsed_s = time.perf_counter() - train_wall_start
    pool_manifest = _pool_manifest(pool, problem.n_qubits)
    summary = {
        "config": as_dict(config),
        "metadata": {
            "git_commit": _git_commit(),
            "method_variant": config.operator_pool.spec,
            "pool_sha256": pool_manifest["sha256"],
            "gate_count_table_i_convention": "paper/cited Pauli-evolution counts excluding HF-reference x gates; total_including_hf_x also reported",
            "detailed_timing": detailed_timing,
        },
        "references": {
            "hf_energy": problem.hf_energy,
            "casci_energy": problem.casci_energy,
            "ccsd_energy": problem.ccsd_energy,
            "n_qubits": problem.n_qubits,
            "n_determinants": problem.n_determinants,
        },
        "operator_pool": {
            "vocab_size": pool.vocab_size,
            "sequence_length": pool.sequence_length,
            "spec": pool.spec,
            "config": as_dict(config.operator_pool),
            "sha256": pool_manifest["sha256"],
            "operators": pool_manifest["operators"],
        },
        "model": {
            "parameters": count_parameters(model),
            "parameter_memory_mb": parameter_memory_mb(model),
        },
        "best": {
            **best,
            "sequence_composition": _sequence_composition(best.get("sequence"), pool),
            "paper_table_i_count_view": _paper_table_count_view(best.get("gate_count", {}) or {}),
            "paper_table_i_delta": _paper_table_delta(best.get("gate_count", {}) or {}),
        },
        "timing_s": {
            "setup": float(setup_elapsed_s),
            "training_total": float(train_elapsed_s),
            "iteration_total": _timing_stats([
                float(entry.get("timing_s", {}).get("iteration_total", 0.0)) for entry in history
            ]),
            "generation_total": _timing_stats([
                float(entry.get("timing_s", {}).get("generation", 0.0)) for entry in history
            ]),
            "evaluation_total": _timing_stats([
                float(entry.get("timing_s", {}).get("evaluation", 0.0)) for entry in history
            ]),
            "optimization_total": _timing_stats([
                float(entry.get("timing_s", {}).get("optimization", 0.0)) for entry in history
            ]),
        },
        "final": {
            "iteration": final_entry.get("iteration", -1),
            "mean_energy": final_entry.get("mean_energy"),
            "best_batch_energy": final_entry.get("best_batch_energy"),
            "best_so_far_error": final_entry.get("best_so_far_error"),
            "final_batch_best_error": (
                None
                if not final_entry
                else float(final_entry.get("best_batch_energy") - problem.casci_energy)
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the H4 GQKAE MVP")
    parser.add_argument("--config", default="configs/h4_mvp.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. training.iterations=2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args.override)
    summary = train(config)
    print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
