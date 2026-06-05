"""UCCSD-derived discrete operator vocabulary for H4 CAS(4,4)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .chemistry import DeterminantBasis
from .config import OperatorPoolConfig


@dataclass(frozen=True)
class ExcitationOperator:
    token: int
    name: str
    annihilate: tuple[int, ...]
    create: tuple[int, ...]
    angle: float

    @property
    def rank(self) -> int:
        return len(self.annihilate)

    @property
    def is_noop(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class OperatorPool:
    operators: tuple[ExcitationOperator, ...]
    sequence_length: int

    @property
    def vocab_size(self) -> int:
        return len(self.operators)

    def __getitem__(self, token: int) -> ExcitationOperator:
        return self.operators[int(token)]


def spin_of_spin_orbital(spin_orbital: int, n_spatial: int) -> int:
    """0 for alpha, 1 for beta under the MVP alpha-first qubit ordering."""
    return 0 if spin_orbital < n_spatial else 1


def build_h4_uccsd_pool(
    basis: DeterminantBasis,
    config: OperatorPoolConfig | None = None,
) -> OperatorPool:
    """Build a compact UCCSD-like pool from the H4 Hartree-Fock determinant.

    Occupied spin orbitals are the first two alpha and first two beta active orbitals;
    virtual spin orbitals are the remaining two alpha and beta orbitals. Singles and
    doubles are included when they preserve alpha/beta electron counts. Each discrete
    token applies a fixed-angle fermionic excitation rotation.
    """
    config = config or OperatorPoolConfig()
    n = basis.n_orbitals
    alpha_occ = tuple(range(basis.n_alpha))
    beta_occ = tuple(n + i for i in range(basis.n_beta))
    alpha_virt = tuple(range(basis.n_alpha, n))
    beta_virt = tuple(n + i for i in range(basis.n_beta, n))
    occ = alpha_occ + beta_occ
    virt = alpha_virt + beta_virt

    ops: list[ExcitationOperator] = []
    if config.include_noop:
        ops.append(ExcitationOperator(0, "noop", (), (), 0.0))

    def add(name: str, annihilate: tuple[int, ...], create: tuple[int, ...]) -> None:
        ops.append(
            ExcitationOperator(
                token=len(ops),
                name=name,
                annihilate=annihilate,
                create=create,
                angle=float(config.excitation_angle),
            )
        )

    # Spin-preserving singles.
    for i, a in product(alpha_occ, alpha_virt):
        add(f"S_a_{i}->{a}", (i,), (a,))
    for i, a in product(beta_occ, beta_virt):
        add(f"S_b_{i-n}->{a-n}", (i,), (a,))

    # Spin-projection preserving doubles.
    for ij in combinations(occ, 2):
        ij_spins = sorted(spin_of_spin_orbital(x, n) for x in ij)
        for ab in combinations(virt, 2):
            if sorted(spin_of_spin_orbital(x, n) for x in ab) != ij_spins:
                continue
            add(f"D_{ij}->{ab}", tuple(sorted(ij)), tuple(sorted(ab)))

    return OperatorPool(tuple(ops), sequence_length=int(config.sequence_length))


def approximate_gate_count_for_operator(op: ExcitationOperator) -> tuple[int, int]:
    """Return rough (two_qubit, total) counts for reporting.

    The paper reports counts after decomposition; this MVP does not reproduce the exact
    CUDA-Q decomposition. These conservative estimates keep resource reporting visible.
    """
    if op.is_noop:
        return 0, 0
    if op.rank == 1:
        return 2, 8
    if op.rank == 2:
        return 8, 28
    return 12 * op.rank, 40 * op.rank


def approximate_gate_count(sequence: list[int] | np.ndarray, pool: OperatorPool) -> dict[str, int]:
    two = 0
    total = 0
    for token in sequence:
        c2, ct = approximate_gate_count_for_operator(pool[int(token)])
        two += c2
        total += ct
    return {"two_qubit": int(two), "total": int(total)}
