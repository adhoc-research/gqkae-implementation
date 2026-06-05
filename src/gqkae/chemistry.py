"""H4 active-space chemistry utilities.

The MVP is deliberately narrow: H4, linear-chain geometry, 6-31G, CAS(4e,4o),
with the paper's default equilibrium-style bond length of 0.88 Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import MoleculeConfig


@dataclass(frozen=True)
class DeterminantBasis:
    """PySCF-compatible determinant basis for fixed (n_alpha, n_beta)."""

    n_orbitals: int
    n_alpha: int
    n_beta: int
    alpha_strings: tuple[int, ...]
    beta_strings: tuple[int, ...]
    bitstrings: tuple[str, ...]
    index_by_bitstring: dict[str, int]
    hf_bitstring: str

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_orbitals

    @property
    def size(self) -> int:
        return len(self.bitstrings)


def h4_linear_geometry(bond_length_angstrom: float = 0.88) -> list[tuple[str, tuple[float, float, float]]]:
    """Return a linear H4 chain with adjacent H-H spacing ``bond_length_angstrom``."""
    r = float(bond_length_angstrom)
    return [("H", (i * r, 0.0, 0.0)) for i in range(4)]


def _bits_from_int(value: int, n_bits: int) -> list[int]:
    return [(value >> i) & 1 for i in range(n_bits)]


def bitstring_from_alpha_beta(alpha: int, beta: int, n_orbitals: int) -> str:
    """Encode alpha occupations then beta occupations, qubit-0 first.

    This internal convention makes bitstrings easy to map back to PySCF's alpha/beta
    strings. It is not intended as a hardware endianness convention.
    """
    occ = _bits_from_int(alpha, n_orbitals) + _bits_from_int(beta, n_orbitals)
    return "".join(str(bit) for bit in occ)


def alpha_beta_from_bitstring(bitstring: str, n_orbitals: int) -> tuple[int, int]:
    if len(bitstring) != 2 * n_orbitals:
        raise ValueError(f"expected {2 * n_orbitals} bits, got {len(bitstring)}")
    alpha = sum((bitstring[i] == "1") << i for i in range(n_orbitals))
    beta = sum((bitstring[n_orbitals + i] == "1") << i for i in range(n_orbitals))
    return alpha, beta


def make_determinant_basis(n_orbitals: int, n_alpha: int, n_beta: int) -> DeterminantBasis:
    try:
        from pyscf.fci import cistring
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PySCF is required for determinant-basis construction") from exc

    alpha_strings = tuple(int(x) for x in cistring.make_strings(range(n_orbitals), n_alpha))
    beta_strings = tuple(int(x) for x in cistring.make_strings(range(n_orbitals), n_beta))
    bitstrings = tuple(
        bitstring_from_alpha_beta(a, b, n_orbitals)
        for a, b in product(alpha_strings, beta_strings)
    )
    index_by_bitstring = {bit: i for i, bit in enumerate(bitstrings)}

    hf_alpha = sum(1 << i for i in range(n_alpha))
    hf_beta = sum(1 << i for i in range(n_beta))
    hf_bitstring = bitstring_from_alpha_beta(hf_alpha, hf_beta, n_orbitals)
    return DeterminantBasis(
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
        alpha_strings=alpha_strings,
        beta_strings=beta_strings,
        bitstrings=bitstrings,
        index_by_bitstring=index_by_bitstring,
        hf_bitstring=hf_bitstring,
    )


@dataclass
class ActiveSpaceProblem:
    config: MoleculeConfig
    hf_energy: float
    casci_energy: float
    ecore: float
    h1_active: np.ndarray
    eri_active: np.ndarray
    basis: DeterminantBasis
    hamiltonian: np.ndarray

    @property
    def n_qubits(self) -> int:
        return self.basis.n_qubits

    @property
    def n_determinants(self) -> int:
        return self.basis.size


def build_h4_problem(config: MoleculeConfig | None = None) -> ActiveSpaceProblem:
    """Build the fixed H4 CAS(4e,4o) / 6-31G active-space problem.

    The projected Hamiltonian is explicitly materialized in the fixed-Sz determinant
    basis (36 determinants for H4 CAS(4,4)), which makes QSCI subspace extraction very
    cheap for the MVP.
    """
    config = config or MoleculeConfig()
    _validate_h4_config(config)

    try:
        from pyscf import ao2mo, gto, mcscf, scf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PySCF is required for chemistry setup. Install with `uv sync`."
        ) from exc

    mol = gto.M(
        atom=h4_linear_geometry(config.bond_length_angstrom),
        basis=config.basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    ncas = config.active_orbitals
    nelecas = (config.active_electrons // 2, config.active_electrons // 2)

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.kernel(verbose=0)
    h1_active, ecore = mc.get_h1eff()
    mo_active = mc.mo_coeff[:, mc.ncore : mc.ncore + ncas]
    eri_active = ao2mo.restore(1, ao2mo.kernel(mol, mo_active), ncas)

    basis = make_determinant_basis(ncas, *nelecas)
    hamiltonian = build_fci_hamiltonian(h1_active, eri_active, ncas, nelecas, ecore)
    eig0 = float(np.linalg.eigvalsh(hamiltonian)[0])
    # Use explicit Hamiltonian eigenvalue as CASCI reference for exact consistency with QSCI.
    casci_energy = eig0

    return ActiveSpaceProblem(
        config=config,
        hf_energy=float(mf.e_tot),
        casci_energy=casci_energy,
        ecore=float(ecore),
        h1_active=np.asarray(h1_active),
        eri_active=np.asarray(eri_active),
        basis=basis,
        hamiltonian=np.asarray(hamiltonian),
    )


def build_fci_hamiltonian(
    h1: np.ndarray,
    eri: np.ndarray,
    n_orbitals: int,
    nelecas: tuple[int, int],
    ecore: float,
) -> np.ndarray:
    """Materialize the PySCF FCI Hamiltonian matrix for a tiny active space."""
    try:
        from pyscf import fci
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PySCF is required for FCI Hamiltonian construction") from exc

    from pyscf.fci import cistring

    na = len(cistring.make_strings(range(n_orbitals), nelecas[0]))
    nb = len(cistring.make_strings(range(n_orbitals), nelecas[1]))
    dim = na * nb
    ham = np.zeros((dim, dim), dtype=float)
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, n_orbitals, nelecas, 0.5)
    for col in range(dim):
        vec = np.zeros((na, nb), dtype=float)
        vec.reshape(-1)[col] = 1.0
        hvec = fci.direct_spin1.contract_2e(h2e, vec, n_orbitals, nelecas)
        hvec += ecore * vec
        ham[:, col] = hvec.reshape(-1)
    return 0.5 * (ham + ham.T)


def _validate_h4_config(config: MoleculeConfig) -> None:
    if config.name.upper() != "H4":
        raise ValueError("MVP only supports molecule.name=H4")
    if config.geometry != "linear_chain":
        raise ValueError("MVP only supports molecule.geometry=linear_chain")
    if config.active_electrons != 4 or config.active_orbitals != 4:
        raise ValueError("MVP only supports H4 CAS(4e,4o)")
    if config.basis.lower() != "6-31g":
        raise ValueError("MVP only supports 6-31G basis")
