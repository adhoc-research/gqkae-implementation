"""Quantum-selected configuration interaction post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .chemistry import ActiveSpaceProblem


@dataclass(frozen=True)
class QSCIResult:
    energy: float
    determinant_indices: tuple[int, ...]
    determinant_bitstrings: tuple[str, ...]
    subspace_dimension: int

    @property
    def valid(self) -> bool:
        return self.subspace_dimension > 0 and np.isfinite(self.energy)


def select_determinants(
    counts: Mapping[str, int],
    problem: ActiveSpaceProblem,
    dmax: int,
    add_hf_det: bool = True,
) -> tuple[int, ...]:
    """Select the most frequent valid determinants for QSCI."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    selected: list[int] = []
    seen: set[int] = set()

    if add_hf_det:
        hf_idx = problem.basis.index_by_bitstring[problem.basis.hf_bitstring]
        selected.append(hf_idx)
        seen.add(hf_idx)

    for bitstring, _count in ranked:
        idx = problem.basis.index_by_bitstring.get(bitstring)
        if idx is None or idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= dmax:
            break
    return tuple(selected[:dmax])


def qsci_energy_from_counts(
    counts: Mapping[str, int],
    problem: ActiveSpaceProblem,
    dmax: int,
    add_hf_det: bool = True,
) -> QSCIResult:
    """Diagonalize the active-space Hamiltonian in the sampled determinant subspace."""
    indices = select_determinants(counts, problem, dmax=dmax, add_hf_det=add_hf_det)
    if not indices:
        return QSCIResult(float("inf"), (), (), 0)

    sub_h = problem.hamiltonian[np.ix_(indices, indices)]
    eigvals = np.linalg.eigvalsh(sub_h)
    bitstrings = tuple(problem.basis.bitstrings[i] for i in indices)
    return QSCIResult(
        energy=float(eigvals[0]),
        determinant_indices=indices,
        determinant_bitstrings=bitstrings,
        subspace_dimension=len(indices),
    )


def energy_error(energy: float, problem: ActiveSpaceProblem) -> float:
    return float(energy - problem.casci_energy)
