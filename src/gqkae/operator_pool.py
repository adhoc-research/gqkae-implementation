"""UCCSD-derived discrete operator vocabularies for H4 CAS(4,4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Literal

from .chemistry import DeterminantBasis
from .config import OperatorPoolConfig

PauliWordSpec = tuple[str, ...]
PauliTermSpec = tuple[float, PauliWordSpec]


@dataclass(frozen=True)
class ExcitationOperator:
    token: int
    name: str
    annihilate: tuple[int, ...]
    create: tuple[int, ...]
    angle: float
    pauli_terms: tuple[PauliTermSpec, ...] = field(default_factory=tuple)
    kind: str = "excitation"
    parent: str | None = None

    @property
    def rank(self) -> int:
        return len(self.annihilate)

    @property
    def is_noop(self) -> bool:
        return self.rank == 0 and not self.pauli_terms


@dataclass(frozen=True)
class OperatorPool:
    operators: tuple[ExcitationOperator, ...]
    sequence_length: int
    spec: str = "excitation"

    @property
    def vocab_size(self) -> int:
        return len(self.operators)

    def __getitem__(self, token: int) -> ExcitationOperator:
        return self.operators[int(token)]


def spin_of_spin_orbital(spin_orbital: int, n_spatial: int) -> int:
    """0 for alpha, 1 for beta under the MVP alpha-first qubit ordering."""
    return 0 if spin_orbital < n_spatial else 1


def _strip_z_ladder(word: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("I" if p == "Z" else p for p in word)


def _base_h4_uccsd_operators(basis: DeterminantBasis, config: OperatorPoolConfig) -> list[ExcitationOperator]:
    """Build the compact H4 UCCSD-like excitation list used by the MVP."""
    n = basis.n_orbitals
    alpha_occ = tuple(range(basis.n_alpha))
    beta_occ = tuple(n + i for i in range(basis.n_beta))
    alpha_virt = tuple(range(basis.n_alpha, n))
    beta_virt = tuple(n + i for i in range(basis.n_beta, n))
    occ = alpha_occ + beta_occ
    virt = alpha_virt + beta_virt

    ops: list[ExcitationOperator] = []
    if config.include_noop:
        ops.append(ExcitationOperator(0, "noop", (), (), 0.0, kind="noop"))

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
    return ops


def _processed_pauli_terms(
    op: ExcitationOperator,
    n_qubits: int,
    *,
    remove_z_ladder: bool,
    only_use_first_pauli: bool,
) -> tuple[PauliTermSpec, ...]:
    from .fermion_mapping import excitation_pauli_terms

    terms = excitation_pauli_terms(op, n_qubits)
    if only_use_first_pauli:
        terms = terms[:1]
    out: list[PauliTermSpec] = []
    for term in terms:
        word = _strip_z_ladder(term.word) if remove_z_ladder else tuple(term.word)
        if all(p == "I" for p in word):
            continue
        out.append((float(term.coefficient), tuple(word)))
    return tuple(out)


def _build_reduced_excitation_pool(
    base_ops: list[ExcitationOperator],
    basis: DeterminantBasis,
    config: OperatorPoolConfig,
) -> OperatorPool:
    ops: list[ExcitationOperator] = []
    for base in base_ops:
        if base.is_noop:
            ops.append(ExcitationOperator(len(ops), base.name, (), (), 0.0, kind="noop"))
            continue
        terms = _processed_pauli_terms(
            base,
            basis.n_qubits,
            remove_z_ladder=config.remove_z_ladder,
            only_use_first_pauli=config.only_use_first_pauli,
        )
        ops.append(
            ExcitationOperator(
                token=len(ops),
                name=base.name,
                annihilate=base.annihilate,
                create=base.create,
                angle=base.angle,
                pauli_terms=terms,
                kind="excitation_reduced",
                parent=base.name,
            )
        )
    return OperatorPool(tuple(ops), sequence_length=int(config.sequence_length), spec="excitation_reduced")


def _build_pauli_evolution_pool(
    base_ops: list[ExcitationOperator],
    basis: DeterminantBasis,
    config: OperatorPoolConfig,
) -> OperatorPool:
    """Build a cited-style Pauli-evolution token vocabulary.

    This mirrors the cited implementation structurally: expand each UCCSD-like excitation
    into Pauli strings, optionally strip Jordan-Wigner Z ladders, optionally retain only
    the first Pauli string per excitation gate, and deduplicate Pauli words.
    """
    ops: list[ExcitationOperator] = []
    seen: set[tuple[str, ...]] = set()
    if config.include_noop:
        ops.append(ExcitationOperator(0, "noop", (), (), 0.0, kind="noop"))
    for base in base_ops:
        if base.is_noop:
            continue
        terms = _processed_pauli_terms(
            base,
            basis.n_qubits,
            remove_z_ladder=config.remove_z_ladder,
            only_use_first_pauli=config.only_use_first_pauli,
        )
        for coeff, word in terms:
            if config.dedupe_pauli_words and word in seen:
                continue
            seen.add(word)
            ops.append(
                ExcitationOperator(
                    token=len(ops),
                    name=f"P_{''.join(word)}__{base.name}",
                    annihilate=(),
                    create=(),
                    angle=base.angle,
                    pauli_terms=((coeff, word),),
                    kind="pauli_evolution",
                    parent=base.name,
                )
            )
    return OperatorPool(tuple(ops), sequence_length=int(config.sequence_length), spec="pauli_evolution")


def build_h4_uccsd_pool(
    basis: DeterminantBasis,
    config: OperatorPoolConfig | None = None,
) -> OperatorPool:
    """Build the configured H4 operator pool.

    ``spec='excitation'`` with default flags preserves the original full UCCSD-like
    excitation-token vocabulary. Setting reduction flags on that spec executes/counts the
    selected Pauli terms for each excitation token. ``spec='pauli_evolution'`` exposes
    the selected Pauli evolutions themselves as discrete tokens, matching the cited
    implementation's default representation more closely.
    """
    config = config or OperatorPoolConfig()
    base_ops = _base_h4_uccsd_operators(basis, config)
    spec = config.spec.lower()
    has_reduction = config.remove_z_ladder or config.only_use_first_pauli
    if spec == "excitation" and not has_reduction:
        return OperatorPool(tuple(base_ops), sequence_length=int(config.sequence_length), spec="excitation")
    if spec in {"excitation", "excitation_reduced"}:
        return _build_reduced_excitation_pool(base_ops, basis, config)
    if spec == "pauli_evolution":
        return _build_pauli_evolution_pool(base_ops, basis, config)
    raise ValueError(f"unknown operator_pool.spec={config.spec!r}")
