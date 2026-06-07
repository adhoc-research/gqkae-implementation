#!/usr/bin/env python
"""Benchmark H4 policy model-only costs for paper-scale ETA estimates."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch

from gqkae.chemistry import build_h4_problem
from gqkae.config import apply_overrides, as_dict, load_config
from gqkae.models.gpt_policy import GPTPolicy
from gqkae.models.hqkansformer import HQKANsformerPolicy, count_parameters, parameter_memory_mb
from gqkae.operator_pool import build_operator_pool
from gqkae.training.grpo import grpo_loss, normalize_advantages
from gqkae.training.runner import set_seed


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes. This repository is usually run on Linux.
    return float(usage / 1024.0 if usage > 10_000 else usage / (1024.0 * 1024.0))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device) -> tuple[Any, float]:
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - start


def _build_model(variant: str, vocab_size: int, sequence_length: int, config):
    variant = variant.lower()
    if variant in {"hqkan", "gqkae", "hqkansformer"}:
        return HQKANsformerPolicy(vocab_size, sequence_length, config), "gqkae"
    if variant in {"gpt", "gqe", "transformer"}:
        return GPTPolicy(vocab_size, sequence_length, config), "gqe"
    raise ValueError(f"unknown model variant {variant!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--sample-repeats", type=int, default=1)
    p.add_argument("--opt-updates", type=int, default=None, help="Defaults to config training.policy_updates")
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg.experiment.seed)
    device = torch.device(cfg.training.device)

    setup_start = time.perf_counter()
    problem = build_h4_problem(cfg.molecule)
    pool = build_operator_pool(problem, cfg.operator_pool)
    setup_s = time.perf_counter() - setup_start

    (model, model_family), instantiation_s = _time_call(
        lambda: _build_model(cfg.model.variant, pool.vocab_size, pool.sequence_length, cfg.model),
        device,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    batch = int(cfg.training.batch_circuits)
    length = int(pool.sequence_length)
    dummy_inputs = torch.randint(0, pool.vocab_size, (batch, length), device=device)
    dummy_sequences = torch.randint(0, pool.vocab_size, (batch, length), device=device)
    old_log_probs = torch.zeros((batch, length), dtype=torch.float32, device=device)
    rewards = torch.randn(batch, dtype=torch.float32, device=device)
    advantages = normalize_advantages(rewards)

    model.eval()
    _, forward_s = _time_call(lambda: model(dummy_inputs), device)
    sample_times = []
    for _ in range(args.sample_repeats):
        _, elapsed = _time_call(
            lambda: model.sample(batch, cfg.training.temperature, cfg.training.repetition_penalty, device),
            device,
        )
        sample_times.append(elapsed)

    model.train()
    updates = int(args.opt_updates if args.opt_updates is not None else cfg.training.policy_updates)
    losses = []
    def one_update_loop():
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            new_log_probs = model.log_probs_for_sequences(dummy_sequences)
            loss = grpo_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=cfg.training.grpo_clip)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    _, opt_loop_s = _time_call(one_update_loop, device)

    payload = {
        "config_path": args.config,
        "config": as_dict(cfg),
        "model_family": model_family,
        "vocab_size": pool.vocab_size,
        "sequence_length": pool.sequence_length,
        "parameters": count_parameters(model),
        "parameter_memory_mb": parameter_memory_mb(model),
        "rss_peak_mb": _rss_mb(),
        "timing_s": {
            "chemistry_pool_setup": setup_s,
            "model_instantiation": instantiation_s,
            "forward_batch": forward_s,
            "sample_batch_mean": sum(sample_times) / len(sample_times),
            "sample_batch_total": sum(sample_times),
            "opt_updates": updates,
            "opt_loop_total": opt_loop_s,
            "opt_update_mean": opt_loop_s / max(updates, 1),
        },
    }
    if args.output is None:
        stem = Path(args.config).stem
        out = Path("runs") / f"{stem}_model_scale_benchmark.json"
    else:
        out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
