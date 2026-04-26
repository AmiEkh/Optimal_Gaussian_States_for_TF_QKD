
"""State-preparation operators and density matrices."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .operators import annihilation_operator, number_state_ket, optical_identity


def vss_filters(q0: float, phi0: float, q1: float, phi1: float, n_truncation: int):
    """Return F_plus, F_minus for vacuum/single-photon superpositions."""
    I = optical_identity(n_truncation)
    adag = annihilation_operator(n_truncation).conj().T
    F_plus = np.sqrt(q0) * I + np.exp(1j * phi0) * np.sqrt(1 - q0) * adag
    F_minus = np.sqrt(q1) * I + np.exp(1j * phi1) * np.sqrt(1 - q1) * adag
    return F_plus, F_minus


def gaussian_filters(b0: float, phi0: float, s0: float, theta0: float,
                     b1: float, phi1: float, s1: float, theta1: float,
                     n_truncation: int):
    """Return displacement-squeezing operators for the two Gaussian bit states."""
    a = annihilation_operator(n_truncation)
    adag = a.conj().T

    beta0 = b0 * np.exp(1j * phi0)
    beta1 = b1 * np.exp(1j * phi1)
    xi0 = s0 * np.exp(2j * theta0)
    xi1 = s1 * np.exp(2j * theta1)

    S0 = expm(0.5 * (np.conj(xi0) * a @ a - xi0 * adag @ adag))
    D0 = expm(beta0 * adag - np.conj(beta0) * a)
    S1 = expm(0.5 * (np.conj(xi1) * a @ a - xi1 * adag @ adag))
    D1 = expm(beta1 * adag - np.conj(beta1) * a)

    return S0 @ D0, S1 @ D1


def thermal_state(epsilon: float, n_truncation: int):
    """Truncated thermal-like state used in the MATLAB code."""
    rho = np.zeros((n_truncation + 1, n_truncation + 1), dtype=complex)
    for n in range(n_truncation + 1):
        ket = number_state_ket(n, n_truncation)
        rho += abs(epsilon) ** (2 * n) * (ket @ ket.conj().T)
    return (1 - abs(epsilon) ** 2) * rho


def prepare_density_matrices(b_A: int, b_B: int, n_truncation: int,
                             bit0_parameters, bit1_parameters, state: str):
    """Prepare Alice/Bob density matrices and filters for the general code path."""
    state = state.upper()

    if state == "VSS":
        q0, phi0 = bit0_parameters
        q1, phi1 = bit1_parameters
        F_plus, F_minus = vss_filters(q0, phi0, q1, phi1, n_truncation)
        vac = number_state_ket(0, n_truncation)
        rho_vac = vac @ vac.conj().T
        rho1 = F_plus @ rho_vac @ F_plus.conj().T if b_A == 0 else F_minus @ rho_vac @ F_minus.conj().T
        rho2 = F_plus @ rho_vac @ F_plus.conj().T if b_B == 0 else F_minus @ rho_vac @ F_minus.conj().T
        return rho1, rho2, F_plus, F_minus

    if state == "GS":
        eps0, b0, phi0, s0, theta0 = bit0_parameters
        eps1, b1, phi1, s1, theta1 = bit1_parameters
        F_plus, F_minus = gaussian_filters(b0, phi0, s0, theta0, b1, phi1, s1, theta1, n_truncation)
        rho_th0 = thermal_state(eps0, n_truncation)
        rho_th1 = thermal_state(eps1, n_truncation)
        rho1 = F_plus @ rho_th0 @ F_plus.conj().T if b_A == 0 else F_minus @ rho_th1 @ F_minus.conj().T
        rho2 = F_plus @ rho_th0 @ F_plus.conj().T if b_B == 0 else F_minus @ rho_th1 @ F_minus.conj().T
        return rho1, rho2, F_plus, F_minus

    raise ValueError(f"Unknown state type: {state!r}")


def prepare_density_matrices_fast(b_A: int, b_B: int, n_truncation: int, parameters, state: str):
    """Prepare density matrices for the symmetric/fast MATLAB convention."""
    state = state.upper()

    if state == "VSS":
        q, phi0, phi1 = parameters
        return prepare_density_matrices(b_A, b_B, n_truncation, [q, phi0], [q, phi1], "VSS")

    if state == "GS":
        epsilon, b, phi0, phi1, s, theta0, theta1 = parameters
        return prepare_density_matrices(
            b_A, b_B, n_truncation,
            [epsilon, b, phi0, s, theta0],
            [epsilon, b, phi1, s, theta1],
            "GS",
        )

    raise ValueError(f"Unknown state type: {state!r}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    N = 6
    dim = N + 1

    print("\n=== 1) VSS filters ===")
    q = 0.8
    phi0 = 0.0
    phi1 = np.pi

    Fp, Fm = vss_filters(q, phi0, q, phi1, N)

    vac = number_state_ket(0, N)
    psi0 = Fp @ vac
    psi1 = Fm @ vac

    print("VSS bit 0 state:")
    print(psi0.ravel())

    print("VSS bit 1 state:")
    print(psi1.ravel())

    assert np.isclose(np.linalg.norm(psi0), 1.0)
    assert np.isclose(np.linalg.norm(psi1), 1.0)
    assert np.isclose(psi0[0, 0], np.sqrt(q))
    assert np.isclose(psi0[1, 0], np.sqrt(1 - q))
    assert np.isclose(psi1[1, 0], -np.sqrt(1 - q))

    print("\n=== 2) Gaussian filters ===")
    b = 0.4
    s = 0.2
    theta0 = 0.0
    theta1 = np.pi

    G0, G1 = gaussian_filters(
        b0=b,
        phi0=0.0,
        s0=s,
        theta0=theta0,
        b1=b,
        phi1=np.pi,
        s1=s,
        theta1=theta1,
        n_truncation=N,
    )

    print("Gaussian filter bit 0 shape:", G0.shape)
    print("Gaussian filter bit 1 shape:", G1.shape)

    # D and S are unitary in the infinite Hilbert space.
    # In finite truncation, they should still be close to unitary for small b, s.
    assert G0.shape == (dim, dim)
    assert G1.shape == (dim, dim)
    assert np.allclose(G0.conj().T @ G0, np.eye(dim), atol=1e-2)
    assert np.allclose(G1.conj().T @ G1, np.eye(dim), atol=1e-2)

    print("\n=== 3) Thermal state ===")
    epsilon = 0.25
    rho_th = thermal_state(epsilon, N)

    print("Thermal state diagonal:")
    print(np.real(np.diag(rho_th)))

    expected_diag = np.array(
        [(1 - epsilon**2) * epsilon ** (2 * n) for n in range(dim)]
    )

    assert rho_th.shape == (dim, dim)
    assert np.allclose(np.diag(rho_th).real, expected_diag)
    assert np.allclose(rho_th, rho_th.conj().T)
    assert np.all(np.linalg.eigvalsh(rho_th) >= -1e-12)

    # Because the state is truncated, trace is slightly less than 1.
    expected_trace = 1 - epsilon ** (2 * (N + 1))
    print("Trace:", np.trace(rho_th).real)
    print("Expected truncated trace:", expected_trace)

    assert np.isclose(np.trace(rho_th).real, expected_trace)

    print("\n=== 4) prepare_density_matrices: VSS ===")
    rho_A, rho_B, Fp, Fm = prepare_density_matrices(
        b_A=0,
        b_B=1,
        n_truncation=N,
        bit0_parameters=[q, 0.0],
        bit1_parameters=[q, np.pi],
        state="VSS",
    )

    print("rho_A diagonal for bit 0:")
    print(np.real(np.diag(rho_A)))

    print("rho_B diagonal for bit 1:")
    print(np.real(np.diag(rho_B)))

    assert rho_A.shape == (dim, dim)
    assert rho_B.shape == (dim, dim)
    assert np.allclose(rho_A, rho_A.conj().T)
    assert np.allclose(rho_B, rho_B.conj().T)
    assert np.isclose(np.trace(rho_A).real, 1.0)
    assert np.isclose(np.trace(rho_B).real, 1.0)

    print("\n=== 5) prepare_density_matrices: GS ===")
    rho_A_gs, rho_B_gs, Gp, Gm = prepare_density_matrices(
        b_A=0,
        b_B=1,
        n_truncation=N,
        bit0_parameters=[0.05, 0.3, 0.0, 0.1, 0.0],
        bit1_parameters=[0.05, 0.3, np.pi, 0.1, np.pi],
        state="GS",
    )

    print("Trace rho_A GS:", np.trace(rho_A_gs).real)
    print("Trace rho_B GS:", np.trace(rho_B_gs).real)

    assert rho_A_gs.shape == (dim, dim)
    assert rho_B_gs.shape == (dim, dim)
    assert np.allclose(rho_A_gs, rho_A_gs.conj().T, atol=1e-12)
    assert np.allclose(rho_B_gs, rho_B_gs.conj().T, atol=1e-12)
    assert np.trace(rho_A_gs).real > 0
    assert np.trace(rho_B_gs).real > 0
    assert np.all(np.linalg.eigvalsh(rho_A_gs) >= -1e-10)
    assert np.all(np.linalg.eigvalsh(rho_B_gs) >= -1e-10)

    print("\n=== 6) prepare_density_matrices_fast: VSS ===")
    rho_fast_A, rho_fast_B, _, _ = prepare_density_matrices_fast(
        b_A=0,
        b_B=1,
        n_truncation=N,
        parameters=[q, 0.0, np.pi],
        state="VSS",
    )

    assert np.allclose(rho_fast_A, rho_A)
    assert np.allclose(rho_fast_B, rho_B)

    print("Fast VSS matches full VSS.")

    print("\n=== 7) prepare_density_matrices_fast: GS ===")
    rho_fast_A_gs, rho_fast_B_gs, _, _ = prepare_density_matrices_fast(
        b_A=0,
        b_B=1,
        n_truncation=N,
        parameters=[0.05, 0.3, 0.0, np.pi, 0.1, 0.0, np.pi],
        state="GS",
    )

    assert np.allclose(rho_fast_A_gs, rho_A_gs)
    assert np.allclose(rho_fast_B_gs, rho_B_gs)

    print("Fast GS matches full GS.")

    print("\nstates.py sanity checks passed")