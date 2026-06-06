from gqkae.chemistry import (
    alpha_beta_from_bitstring,
    bitstring_between_orderings,
    bitstring_from_alpha_beta,
    make_determinant_basis,
    qubit_to_spin_orbital,
    spin_orbital_to_qubit,
)


def test_alpha_beta_and_interleaved_roundtrip():
    n = 4
    alpha = 0b0101
    beta = 0b1010
    blocked = bitstring_from_alpha_beta(alpha, beta, n, "alpha_beta")
    interleaved = bitstring_from_alpha_beta(alpha, beta, n, "interleaved")

    assert blocked == "10100101"
    assert interleaved == "10011001"
    assert alpha_beta_from_bitstring(blocked, n, "alpha_beta") == (alpha, beta)
    assert alpha_beta_from_bitstring(interleaved, n, "interleaved") == (alpha, beta)
    assert bitstring_between_orderings(blocked, n, "alpha_beta", "interleaved") == interleaved
    assert bitstring_between_orderings(interleaved, n, "interleaved", "alpha_beta") == blocked


def test_interleaved_qubit_spin_orbital_map():
    n = 4
    # alpha_i -> 2*i, beta_i -> 2*i+1 under cited ordering.
    assert [spin_orbital_to_qubit(i, n, "interleaved") for i in range(2 * n)] == [0, 2, 4, 6, 1, 3, 5, 7]
    assert [qubit_to_spin_orbital(q, n, "interleaved") for q in range(2 * n)] == [0, 4, 1, 5, 2, 6, 3, 7]


def test_determinant_basis_uses_requested_ordering_and_pyscf_strings():
    blocked = make_determinant_basis(4, 2, 2, "alpha_beta")
    interleaved = make_determinant_basis(4, 2, 2, "interleaved")

    assert blocked.hf_bitstring == "11001100"
    assert interleaved.hf_bitstring == "11110000"
    assert blocked.alpha_strings == interleaved.alpha_strings
    assert blocked.beta_strings == interleaved.beta_strings
    assert bitstring_between_orderings(blocked.hf_bitstring, 4, "alpha_beta", "interleaved") == interleaved.hf_bitstring
