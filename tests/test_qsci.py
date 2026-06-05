import numpy as np

from gqkae.qsci import qsci_energy_from_counts


class Basis:
    bitstrings = ("11001100", "10101010")
    index_by_bitstring = {"11001100": 0, "10101010": 1}
    hf_bitstring = "11001100"


class Problem:
    basis = Basis()
    hamiltonian = np.array([[0.0, 0.1], [0.1, 1.0]])
    casci_energy = float(np.linalg.eigvalsh(hamiltonian)[0])


def test_qsci_diagonalizes_selected_subspace():
    counts = {"11001100": 10, "10101010": 5}
    result = qsci_energy_from_counts(counts, Problem(), dmax=2, add_hf_det=True)
    assert result.subspace_dimension == 2
    assert abs(result.energy - Problem.casci_energy) < 1e-12
