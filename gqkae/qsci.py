"""Quantum-selected configuration interaction post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from .chemistry import ActiveSpaceProblem, alpha_beta_from_bitstring, bitstring_from_alpha_beta


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


def _iter_set_bits(x: int):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


def _spin_symmetry_group(alpha: int, beta: int) -> list[tuple[int, int]]:
    """Mirror cited determinant total-spin symmetry-completion grouping."""
    doubly_occupied = alpha & beta
    open_shell = alpha ^ beta
    if open_shell == 0:
        return [(alpha, beta)]
    open_orbitals = list(_iter_set_bits(open_shell))
    n_alpha_open = (alpha & open_shell).bit_count()
    rows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for alpha_choice in combinations(open_orbitals, n_alpha_open):
        a_new = doubly_occupied
        b_new = doubly_occupied
        alpha_set = set(alpha_choice)
        for orbital in open_orbitals:
            if orbital in alpha_set:
                a_new |= 1 << orbital
            else:
                b_new |= 1 << orbital
        key = (a_new, b_new)
        if key not in seen:
            seen.add(key)
            rows.append(key)
    return rows


def symmetry_completion_indices(
    indices: tuple[int, ...],
    problem: ActiveSpaceProblem,
    dmax: int,
) -> tuple[int, ...]:
    """Enlarge a determinant subspace using the cited spin-symmetry completion."""
    selected: list[int] = []
    seen: set[int] = set()
    ordering = problem.basis.spin_orbital_ordering
    for idx in indices:
        bitstring = problem.basis.bitstrings[idx]
        alpha, beta = alpha_beta_from_bitstring(bitstring, problem.basis.n_orbitals, ordering)
        for a_new, b_new in _spin_symmetry_group(alpha, beta):
            group_bitstring = bitstring_from_alpha_beta(a_new, b_new, problem.basis.n_orbitals, ordering)
            group_idx = problem.basis.index_by_bitstring.get(group_bitstring)
            if group_idx is None or group_idx in seen:
                continue
            seen.add(group_idx)
            selected.append(group_idx)
            if len(selected) >= dmax:
                return tuple(selected)
    return tuple(selected)


def enlarge_determinants(
    indices: tuple[int, ...],
    problem: ActiveSpaceProblem,
    dmax: int,
    method: str = "none",
) -> tuple[int, ...]:
    normalized = str(method or "none").lower()
    if normalized in {"none", "null"}:
        return tuple(indices[:dmax])
    if normalized == "symmetry_completion":
        return symmetry_completion_indices(indices, problem, dmax=dmax)
    raise ValueError(f"unknown QSCI enlarge_method={method!r}")


def qsci_energy_from_counts(
    counts: Mapping[str, int],
    problem: ActiveSpaceProblem,
    dmax: int,
    add_hf_det: bool = True,
    enlarge_method: str = "none",
) -> QSCIResult:
    """Diagonalize the active-space Hamiltonian in the sampled determinant subspace."""
    indices = select_determinants(counts, problem, dmax=dmax, add_hf_det=add_hf_det)
    indices = enlarge_determinants(indices, problem, dmax=dmax, method=enlarge_method)
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
