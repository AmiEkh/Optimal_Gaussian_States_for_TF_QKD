
"""Loss channels and Charlie measurement model."""
from __future__ import annotations

import numpy as np

from .operators import beam_splitter_unitary, optical_identity, photon_loss_kraus


def apply_loss_channel(rho: np.ndarray, zeta: float, n_truncation: int) -> np.ndarray:
    """Apply the truncated photon-loss channel."""
    out = np.zeros_like(rho, dtype=complex)
    for n in range(n_truncation + 1):
        K = photon_loss_kraus(n, zeta, n_truncation)
        out += K @ rho @ K.conj().T
    return out


def charlie_click_probabilities(rho1: np.ndarray, rho2: np.ndarray, n_truncation: int):
    """Return P00, P01, P10, P11 after Charlie's 50:50 beamsplitter."""
    I = optical_identity(n_truncation)
    Ubs = beam_splitter_unitary(n_truncation)

    rho_in = np.kron(rho1, rho2)
    rho_out = Ubs @ rho_in @ Ubs.conj().T

    vac = np.zeros((n_truncation + 1, 1), dtype=complex)
    vac[0, 0] = 1.0
    proj_vac = vac @ vac.conj().T
    proj_nonvac = I - proj_vac

    M00 = np.kron(proj_vac, proj_vac)
    M01 = np.kron(proj_vac, proj_nonvac)
    M10 = np.kron(proj_nonvac, proj_vac)
    M11 = np.kron(proj_nonvac, proj_nonvac)

    vals = [np.trace(rho_out @ M).real for M in (M00, M01, M10, M11)]
    return tuple(float(np.clip(v, -1e-14, 1 + 1e-14)) for v in vals)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    from .operators import number_state_ket
    from .states import prepare_density_matrices

    N = 6
    dim = N + 1

    print("\n=== 1) Loss channel: vacuum stays vacuum ===")
    vac = number_state_ket(0, N)
    rho_vac = vac @ vac.conj().T

    rho_vac_loss = apply_loss_channel(rho_vac, zeta=0.3, n_truncation=N)

    print("Input vacuum diagonal:")
    print(np.real(np.diag(rho_vac)))
    print("After loss diagonal:")
    print(np.real(np.diag(rho_vac_loss)))

    assert np.allclose(rho_vac_loss, rho_vac)
    assert np.isclose(np.trace(rho_vac_loss).real, 1.0)

    print("\n=== 2) Loss channel: single photon becomes mixture ===")
    ket1 = number_state_ket(1, N)
    rho1 = ket1 @ ket1.conj().T

    zeta = 0.7
    rho1_loss = apply_loss_channel(rho1, zeta=zeta, n_truncation=N)

    print("After loss on |1><1| diagonal:")
    print(np.real(np.diag(rho1_loss)))
    print("Expected: P(|0>) = 1-zeta, P(|1>) = zeta")

    assert np.isclose(rho1_loss[0, 0].real, 1 - zeta)
    assert np.isclose(rho1_loss[1, 1].real, zeta)
    assert np.isclose(np.trace(rho1_loss).real, 1.0)
    assert np.allclose(rho1_loss, rho1_loss.conj().T)

    print("\n=== 3) Loss channel: two-photon binomial decay ===")
    ket2 = number_state_ket(2, N)
    rho2 = ket2 @ ket2.conj().T

    rho2_loss = apply_loss_channel(rho2, zeta=zeta, n_truncation=N)

    print("After loss on |2><2| diagonal:")
    print(np.real(np.diag(rho2_loss)))

    expected_p0 = (1 - zeta) ** 2
    expected_p1 = 2 * zeta * (1 - zeta)
    expected_p2 = zeta ** 2

    print("Expected:")
    print([expected_p0, expected_p1, expected_p2])

    assert np.isclose(rho2_loss[0, 0].real, expected_p0)
    assert np.isclose(rho2_loss[1, 1].real, expected_p1)
    assert np.isclose(rho2_loss[2, 2].real, expected_p2)
    assert np.isclose(np.trace(rho2_loss).real, 1.0)

    print("\n=== 4) Charlie probabilities: two vacua ===")
    probs_vac = charlie_click_probabilities(rho_vac, rho_vac, N)
    print("P00, P01, P10, P11 =", probs_vac)

    assert np.allclose(probs_vac, (1.0, 0.0, 0.0, 0.0), atol=1e-12)
    assert np.isclose(sum(probs_vac), 1.0)

    print("\n=== 5) Charlie probabilities: one photon in port 1, vacuum in port 2 ===")
    probs_10 = charlie_click_probabilities(rho1, rho_vac, N)
    print("P00, P01, P10, P11 =", probs_10)

    # A single photon at a 50:50 beam splitter exits either output with probability 1/2.
    assert np.isclose(probs_10[0], 0.0, atol=1e-12)
    assert np.isclose(probs_10[1], 0.5, atol=1e-12)
    assert np.isclose(probs_10[2], 0.5, atol=1e-12)
    assert np.isclose(probs_10[3], 0.0, atol=1e-12)
    assert np.isclose(sum(probs_10), 1.0)

    print("\n=== 6) Charlie probabilities: HOM test with |1,1> ===")
    probs_11 = charlie_click_probabilities(rho1, rho1, N)
    print("P00, P01, P10, P11 =", probs_11)

    # Hong-Ou-Mandel bunching:
    # two identical photons leave together, so no coincidence click P11.
    assert np.isclose(probs_11[0], 0.0, atol=1e-12)
    assert np.isclose(probs_11[3], 0.0, atol=1e-10)
    assert np.isclose(probs_11[1] + probs_11[2], 1.0, atol=1e-10)
    assert np.isclose(sum(probs_11), 1.0, atol=1e-10)

    print("\n=== 7) Full state example: VSS + channel + Charlie ===")
    rho_A, rho_B, *_ = prepare_density_matrices(
        b_A=0,
        b_B=1,
        n_truncation=N,
        bit0_parameters=[0.85, 0.0],
        bit1_parameters=[0.85, np.pi],
        state="VSS",
    )

    eta = 0.6
    rho_A_loss = apply_loss_channel(rho_A, eta, N)
    rho_B_loss = apply_loss_channel(rho_B, eta, N)

    probs_vss = charlie_click_probabilities(rho_A_loss, rho_B_loss, N)

    print("VSS after loss, Charlie probabilities:")
    print("P00, P01, P10, P11 =", probs_vss)

    assert all(-1e-12 <= p <= 1 + 1e-12 for p in probs_vss)
    assert np.isclose(sum(probs_vss), 1.0, atol=1e-10)
    assert np.isclose(np.trace(rho_A_loss).real, 1.0, atol=1e-10)
    assert np.isclose(np.trace(rho_B_loss).real, 1.0, atol=1e-10)

    print("\nchannel.py sanity checks passed")