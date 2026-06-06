from gqkae.chemistry import MoleculeConfig, build_h4_problem, bitstring_from_alpha_beta
from gqkae.qsci import enlarge_determinants, symmetry_completion_indices


def test_symmetry_completion_adds_spin_partners_interleaved_h4():
    problem = build_h4_problem(MoleculeConfig(spin_orbital_ordering="interleaved"))
    # Open-shell determinant: alpha orbitals {0, 1}, beta orbitals {0, 2}.
    bitstring = bitstring_from_alpha_beta(0b0011, 0b0101, 4, "interleaved")
    idx = problem.basis.index_by_bitstring[bitstring]

    enlarged = symmetry_completion_indices((idx,), problem, dmax=10)
    enlarged_bits = [problem.basis.bitstrings[i] for i in enlarged]

    expected = {
        bitstring_from_alpha_beta(0b0011, 0b0101, 4, "interleaved"),
        bitstring_from_alpha_beta(0b0101, 0b0011, 4, "interleaved"),
    }
    assert set(enlarged_bits) == expected
    for det in enlarged_bits:
        assert det.count("1") == 4


def test_enlarge_none_preserves_order_and_limit():
    problem = build_h4_problem(MoleculeConfig(spin_orbital_ordering="alpha_beta"))
    indices = (3, 2, 1, 0)
    assert enlarge_determinants(indices, problem, dmax=2, method="none") == (3, 2)
