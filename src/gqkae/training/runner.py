"""End-to-end H4 MVP training loop."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ..chemistry import build_h4_problem
from ..circuits import sample_sequence_circuit
from ..config import Config, apply_overrides, as_dict, load_config
from ..operator_pool import build_h4_uccsd_pool
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


def train(config: Config) -> dict[str, Any]:
    torch = _require_torch()
    from ..models.hqkansformer import HQKANsformerPolicy, count_parameters, parameter_memory_mb
    from .grpo import grpo_loss, normalize_advantages

    rng = set_seed(config.experiment.seed)
    out_dir = Path(config.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = build_h4_problem(config.molecule)
    pool = build_h4_uccsd_pool(problem.basis, config.operator_pool)
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
        model.eval()
        with torch.no_grad():
            sequences_t, old_log_probs_t = model.sample(
                batch_size=config.training.batch_circuits,
                temperature=config.training.temperature,
                repetition_penalty=config.training.repetition_penalty,
                device=device,
            )
        sequences = sequences_t.detach().cpu().numpy()

        energies: list[float] = []
        subdims: list[int] = []
        gate_counts: list[dict[str, int]] = []
        for seq in sequences:
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
            result = qsci_energy_from_counts(
                sample.counts,
                problem,
                dmax=config.qsci.dmax,
                add_hf_det=config.qsci.add_hf_det,
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

        rewards_t = torch.tensor([-e for e in energies], dtype=torch.float32, device=device)
        advantages_t = normalize_advantages(rewards_t)

        model.train()
        losses: list[float] = []
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
        }
        history.append(entry)
        print(
            f"iter={iteration:03d} mean_E={entry['mean_energy']:.10f} "
            f"best_err={entry['best_so_far_error']:.6e} loss={entry['mean_loss']:.4e}",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        (out_dir / "best.json").write_text(json.dumps(best, indent=2))

    final_entry = history[-1] if history else {}
    summary = {
        "config": as_dict(config),
        "references": {
            "hf_energy": problem.hf_energy,
            "casci_energy": problem.casci_energy,
            "n_qubits": problem.n_qubits,
            "n_determinants": problem.n_determinants,
        },
        "operator_pool": {
            "vocab_size": pool.vocab_size,
            "sequence_length": pool.sequence_length,
        },
        "model": {
            "parameters": count_parameters(model),
            "parameter_memory_mb": parameter_memory_mb(model),
        },
        "best": best,
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
