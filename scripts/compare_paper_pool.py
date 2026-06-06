#!/usr/bin/env python
"""Compare the local Tequila-backed paper pool against a JSON reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from gqkae.chemistry import build_h4_problem
from gqkae.config import Config, MoleculeConfig, OperatorPoolConfig, load_config
from gqkae.gate_counting import paper_style_pauli_evolution_gate_count
from gqkae.operator_pool import build_operator_pool


def _default_config() -> Config:
    cfg = Config()
    cfg.molecule = MoleculeConfig(spin_orbital_ordering="interleaved")
    cfg.operator_pool = OperatorPoolConfig(
        spec="paper_pauli_evolution",
        sequence_length=20,
        params=None,
        ccsd_threshold=1e-6,
        remove_z_ladder=True,
        only_use_first_pauli=True,
        dedupe_pauli_words=True,
        include_noop=True,
    )
    return cfg


def _local_tokens(config: Config) -> list[dict[str, Any]]:
    problem = build_h4_problem(config.molecule)
    pool = build_operator_pool(problem, config.operator_pool)
    out: list[dict[str, Any]] = []
    for op in pool.operators:
        if op.is_noop:
            word = "I" * problem.n_qubits
            coeff = 0.0
        else:
            assert len(op.pauli_terms) == 1
            coeff, word_tuple = op.pauli_terms[0]
            word = "".join(word_tuple)
        out.append({
            "token": op.token,
            "pauli_word": word,
            "coefficient": float(coeff),
            "gate_count": dict(paper_style_pauli_evolution_gate_count(word)),
        })
    return out


def compare(reference: dict[str, Any], local_tokens: list[dict[str, Any]], coeff_tol: float) -> list[str]:
    errors: list[str] = []
    ref_tokens = reference["tokens"]
    if len(ref_tokens) != len(local_tokens):
        errors.append(f"vocab size mismatch: reference={len(ref_tokens)} local={len(local_tokens)}")
    for i, (ref, loc) in enumerate(zip(ref_tokens, local_tokens, strict=False)):
        if ref.get("token") != loc.get("token"):
            errors.append(f"token index mismatch at row {i}: reference={ref.get('token')} local={loc.get('token')}")
        if ref.get("pauli_word") != loc.get("pauli_word"):
            errors.append(
                f"pauli word mismatch at token {i}: reference={ref.get('pauli_word')} local={loc.get('pauli_word')}"
            )
        if not math.isclose(float(ref.get("coefficient", 0.0)), float(loc.get("coefficient", 0.0)), abs_tol=coeff_tol, rel_tol=0.0):
            errors.append(
                f"coefficient mismatch at token {i}: reference={ref.get('coefficient')} local={loc.get('coefficient')}"
            )
        ref_count = {k: int(v) for k, v in ref.get("gate_count", {}).items() if k in {"h", "s", "sdg", "rz", "cx", "two_qubit", "total"}}
        loc_count = {k: int(v) for k, v in loc.get("gate_count", {}).items() if k in {"h", "s", "sdg", "rz", "cx", "two_qubit", "total"}}
        if ref_count != loc_count:
            errors.append(f"gate count mismatch at token {i}: reference={ref_count} local={loc_count}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional config; defaults to H4 paper-fidelity pool settings.")
    parser.add_argument("--reference", type=Path, default=Path("data/h4_cited_pauli_evolution_pool.json"))
    parser.add_argument("--coeff-tol", type=float, default=1e-10)
    parser.add_argument("--write-local", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else _default_config()
    reference = json.loads(args.reference.read_text())
    local = _local_tokens(cfg)
    if args.write_local:
        args.write_local.parent.mkdir(parents=True, exist_ok=True)
        args.write_local.write_text(json.dumps({"tokens": local}, indent=2, sort_keys=True) + "\n")
    errors = compare(reference, local, coeff_tol=args.coeff_tol)
    summary = {
        "reference": str(args.reference),
        "reference_vocab_size": len(reference["tokens"]),
        "local_vocab_size": len(local),
        "match": not errors,
        "n_errors": len(errors),
        "errors": errors[:20],
    }
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
