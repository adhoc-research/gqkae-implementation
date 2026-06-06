"""UCCSD-derived discrete operator vocabularies for H4 CAS(4,4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Literal

import numpy as np

from .chemistry import ActiveSpaceProblem, DeterminantBasis, h4_linear_geometry
from .config import OperatorPoolConfig

PauliWordSpec = tuple[str, ...]
PauliTermSpec = tuple[float, PauliWordSpec]
SpinExcitationSpec = tuple[int, ...]


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


def _closed_shell_amplitude_dictionary(ccsd_amplitude: dict[str, object]) -> dict[tuple[int, ...], float]:
    """Return Tequila's closed-shell amplitude dictionary.

    The cited implementation delegates spin-adapted closed-shell bookkeeping to
    `tequila.quantumchemistry.chemistry_tools.ClosedShellAmplitudes`. We keep
    this helper small and explicit so paper-fidelity code can use the same
    convention while failing clearly when the optional paper dependency is not
    installed.
    """
    try:
        from tequila.quantumchemistry.chemistry_tools import ClosedShellAmplitudes
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Tequila is required for cited closed-shell CCSD amplitude expansion. "
            "Install with the optional paper extra, e.g. `uv sync --extra paper`."
        ) from exc

    amplitudes = ClosedShellAmplitudes(
        tIjAb=np.asarray(ccsd_amplitude["t2"]),
        tIA=np.asarray(ccsd_amplitude["t1"]),
    )
    return {
        tuple(int(i) for i in key): float(value)
        for key, value in amplitudes.make_parameter_dictionary(threshold=0.0, screening=False).items()
    }


def generate_cited_spin_excitations(
    ccsd_amplitude: dict[str, object],
    threshold: float = 1e-6,
) -> dict[SpinExcitationSpec, float]:
    """Mirror cited `UCCSDBasedPool.generate_excitations`.

    The returned keys are interleaved spin-orbital indices in the cited format:
    alpha spin orbital `p` is `2*p`, beta is `2*p+1`. Values are the angles/
    coefficients used by the cited `make_excitation_gate` path (`2*t` or the
    same partner-corrected variant for same-spin doubles).
    """
    amplitudes_all = _closed_shell_amplitude_dictionary(ccsd_amplitude)
    amplitudes = {
        k: v for k, v in amplitudes_all.items()
        if not np.isclose(v, 0.0, atol=float(threshold))
    }
    amplitudes = dict(sorted(amplitudes.items(), key=lambda x: np.fabs(x[1]), reverse=True))
    indices: dict[SpinExcitationSpec, float] = {}
    for key, t in amplitudes.items():
        assert len(key) % 2 == 0
        if not np.isclose(t, 0.0, atol=float(threshold)):
            if len(key) == 2:
                angle = 2.0 * t
                idx_a = (2 * key[0], 2 * key[1])
                idx_b = (2 * key[0] + 1, 2 * key[1] + 1)
                indices[idx_a] = float(angle)
                indices[idx_b] = float(angle)
            else:
                assert len(key) == 4
                angle = 2.0 * t
                idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
                indices[idx_abab] = float(angle)
                if key[0] != key[2] and key[1] != key[3]:
                    idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                    idx_bbbb = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1)
                    partner = tuple([key[2], key[1], key[0], key[3]])
                    partner_t = amplitudes_all.get(partner, 0.0)
                    anglex = 2.0 * (t - partner_t)
                    indices[idx_aaaa] = float(anglex)
                    indices[idx_bbbb] = float(anglex)
    return indices


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


def _require_tequila_for_pool():
    try:
        import tequila as tq
        from tequila.circuit import QCircuit
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "operator_pool.spec='paper_pauli_evolution' requires Tequila. "
            "Install with `uv sync --extra paper` or use a non-paper pool spec."
        ) from exc
    return tq, QCircuit


def _paulistring_to_word(paulistring, n_qubits: int, *, remove_z_ladder: bool) -> tuple[str, ...]:
    letters = ["I"] * n_qubits
    for k, v in paulistring.items():
        pauli = str(v).upper()
        if remove_z_ladder and pauli == "Z":
            continue
        if pauli in {"X", "Y", "Z"}:
            letters[int(k)] = pauli
    return tuple(letters)


def _build_tequila_uccsd_ansatz(problem: ActiveSpaceProblem, config: OperatorPoolConfig):
    """Mirror cited `make_uccsd_ansatz` using Tequila excitation gates."""
    tq, QCircuit = _require_tequila_for_pool()
    screened_indices = generate_cited_spin_excitations(
        problem.ccsd_amplitude,
        threshold=float(config.ccsd_threshold),
    )
    geometry_lines = [
        f"{atom_type} {coords[0]} {coords[1]} {coords[2]}"
        for atom_type, coords in h4_linear_geometry(problem.config.bond_length_angstrom)
    ]
    tq_molecule = tq.Molecule(
        geometry="\n".join(geometry_lines),
        basis_set=problem.config.basis,
        active_orbitals=list(problem.active_indices),
        transformation="jordan-wigner",
    )
    ansatz = QCircuit()
    for idx, angle in screened_indices.items():
        converted = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
        ansatz += tq_molecule.make_excitation_gate(indices=converted, angle=angle)
    return ansatz, screened_indices


def _build_paper_pauli_evolution_pool(
    problem: ActiveSpaceProblem,
    config: OperatorPoolConfig,
) -> OperatorPool:
    """Build the canonical cited/Tequila-backed H4 PauliEvolutionPool.

    This mirrors `moken20/gqe-for-qsci`:
    `generate_excitations -> make_uccsd_ansatz -> PauliEvolutionPool.build_operator_pool`.
    In cited-default mode (`params is None`) each token coefficient is the Tequila
    excitation gate parameter derived from the screened CCSD amplitude.
    """
    ansatz, screened_indices = _build_tequila_uccsd_ansatz(problem, config)
    del screened_indices  # ordering is represented by the ansatz gate order.

    ops: list[ExcitationOperator] = []
    seen: set[tuple[str, ...]] = set()
    if config.include_noop:
        ops.append(ExcitationOperator(0, "identity", (), (), 0.0, kind="noop"))

    params = config.params
    for gate_index, gate in enumerate(ansatz.gates):
        gate_coeff = float(gate.parameter)
        for p_index, paulistring in enumerate(gate.generator.paulistrings):
            word = _paulistring_to_word(
                paulistring,
                problem.n_qubits,
                remove_z_ladder=bool(config.remove_z_ladder),
            )
            if all(pauli == "I" for pauli in word):
                # Cited convert_pauli_to_cudaq_spin returns None for an all-identity
                # mapping; skip identity Pauli evolutions in the explicit pool.
                continue
            if config.dedupe_pauli_words and word in seen:
                continue
            seen.add(word)
            coefficients = [gate_coeff] if params is None else [float(x) for x in params]
            for coeff in coefficients:
                ops.append(
                    ExcitationOperator(
                        token=len(ops),
                        name=f"paper_P_{''.join(word)}__g{gate_index}_p{p_index}",
                        annihilate=(),
                        create=(),
                        angle=1.0,
                        pauli_terms=((float(coeff), word),),
                        kind="pauli_evolution",
                        parent=f"tequila_gate_{gate_index}",
                    )
                )
            if config.only_use_first_pauli:
                break
    return OperatorPool(tuple(ops), sequence_length=int(config.sequence_length), spec="paper_pauli_evolution")


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
    if spec == "paper_pauli_evolution":
        raise ValueError(
            "operator_pool.spec='paper_pauli_evolution' requires build_operator_pool(problem, config), "
            "not build_h4_uccsd_pool(basis, config)."
        )
    raise ValueError(f"unknown operator_pool.spec={config.spec!r}")


def build_operator_pool(
    problem: ActiveSpaceProblem,
    config: OperatorPoolConfig | None = None,
) -> OperatorPool:
    """Build an operator pool from the full active-space problem.

    This is the preferred entry point for training. It preserves the existing MVP
    pools and adds the Tequila-backed paper-fidelity pool which needs CCSD
    amplitudes and active orbital indices from `ActiveSpaceProblem`.
    """
    config = config or OperatorPoolConfig()
    if config.spec.lower() == "paper_pauli_evolution":
        return _build_paper_pauli_evolution_pool(problem, config)
    return build_h4_uccsd_pool(problem.basis, config)
