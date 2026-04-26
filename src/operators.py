
"""Basic finite-dimensional optical operators."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


Array = np.ndarray


def number_state_ket(n: int, n_truncation: int) -> Array:
    """Return |n> as a column vector in a Fock space truncated at n_truncation."""
    ket = np.zeros((n_truncation + 1, 1), dtype=complex)
    if 0 <= n <= n_truncation:
        ket[n, 0] = 1.0
    return ket


def annihilation_operator(n_truncation: int) -> Array:
    """Return the annihilation operator a in the truncated Fock basis."""
    dim = n_truncation + 1
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
    return a


def optical_identity(n_truncation: int) -> Array:
    return np.eye(n_truncation + 1, dtype=complex)


def beam_splitter_unitary(n_truncation: int, theta: float = np.pi / 4) -> Array:
    """Two-mode beam-splitter unitary exp[theta(a1†a2 - a1a2†)]."""
    a = annihilation_operator(n_truncation)
    I = optical_identity(n_truncation)
    a1 = np.kron(a, I)
    a2 = np.kron(I, a)
    return expm(theta * (a1.conj().T @ a2 - a1 @ a2.conj().T))


def photon_loss_kraus(n: int, zeta: float, n_truncation: int) -> Array:
    """Kraus operator for amplitude transmission parameter zeta."""
    import math
    a = annihilation_operator(n_truncation)
    num = a.conj().T @ a
    # zeta^(N/2), where N is diagonal in the Fock basis.
    zeta_num = expm(0.5 * np.log(zeta) * num) if zeta > 0 else np.diag([1.0] + [0.0] * n_truncation)
    return np.sqrt((1 - zeta) ** n / math.factorial(n)) * (zeta_num @ np.linalg.matrix_power(a, n))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    N = 6
    dim = N + 1

    print("\n=== 1) Arbitrary number state |n> ===")
    n = 3
    ket_n = number_state_ket(n, N)
    print(f"|{n}> =")
    print(ket_n.ravel())
    assert ket_n.shape == (dim, 1)
    assert np.isclose(np.linalg.norm(ket_n), 1.0)

    print("\n=== 2) Apply annihilation operator: a|n> = sqrt(n)|n-1> ===")
    a = annihilation_operator(N)
    result = a @ ket_n
    expected = np.sqrt(n) * number_state_ket(n - 1, N)
    print(f"a|{n}> =")
    print(result.ravel())
    print(f"Expected sqrt({n})|{n-1}> =")
    print(expected.ravel())
    assert np.allclose(result, expected)

    print("\n=== 3) Optical identity ===")
    I = optical_identity(N)
    print("I =")
    print(I)
    assert np.allclose(I @ ket_n, ket_n)

    print("\n=== 4) Beam splitter: Hong-Ou-Mandel effect ===")
    # Input: |1,1>
    ket_1 = number_state_ket(1, N)
    psi_in = np.kron(ket_1, ket_1)

    U_bs = beam_splitter_unitary(N, theta=np.pi / 4)
    psi_out = U_bs @ psi_in

    def two_mode_ket(n1: int, n2: int) -> Array:
        return np.kron(number_state_ket(n1, N), number_state_ket(n2, N))

    amp_20 = (two_mode_ket(2, 0).conj().T @ psi_out)[0, 0]
    amp_11 = (two_mode_ket(1, 1).conj().T @ psi_out)[0, 0]
    amp_02 = (two_mode_ket(0, 2).conj().T @ psi_out)[0, 0]

    print("Input state: |1,1>")
    print(f"Amplitude |2,0>: {amp_20:.3f}")
    print(f"Amplitude |1,1>: {amp_11:.3e}")
    print(f"Amplitude |0,2>: {amp_02:.3f}")

    print(f"Probability |2,0>: {abs(amp_20)**2:.3f}")
    print(f"Probability |1,1>: {abs(amp_11)**2:.3e}")
    print(f"Probability |0,2>: {abs(amp_02)**2:.3f}")

    # HOM bunching: coincidence |1,1> vanishes
    assert np.isclose(abs(amp_11) ** 2, 0.0, atol=1e-10)
    assert np.isclose(abs(amp_20) ** 2 + abs(amp_02) ** 2, 1.0, atol=1e-10)

    print("\n=== 5) Photon-loss Kraus completeness check ===")
    zeta = 0.7
    K_sum = np.zeros((dim, dim), dtype=complex)

    for ell in range(dim):
        K = photon_loss_kraus(ell, zeta, N)
        K_sum += K.conj().T @ K

    print("Sum_l K_l^† K_l =")
    print(K_sum)

    # In a truncated Fock space, this should be close to identity.
    assert np.allclose(K_sum, I, atol=1e-10)

    print("\noperators.py sanity checks passed")