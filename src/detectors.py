
"""Detector probability functions ported from the MATLAB implementation."""
from __future__ import annotations

import math
import numpy as np

from .channel import apply_loss_channel, charlie_click_probabilities
from .states import prepare_density_matrices, prepare_density_matrices_fast


def detectors_probability(
    b_A: int,
    b_B: int,
    loss_db: float,
    n_truncation: int,
    bit0_parameters,
    bit1_parameters,
    state: str,
    errors=(0.0, 0.0, 0.0),
):
    """General detector probabilities P00, P01, P10, P11.

    Notes:
        errors = (Delta_Loss, Delta_BS, Delta_phase), matching the MATLAB code.
        Delta_BS and Delta_phase are parsed but not yet used in the original code path.
    """
    delta_loss, _delta_bs, _delta_phase = errors
    zeta1 = 10 ** (-loss_db / 20) + delta_loss / 2
    zeta2 = 10 ** (-loss_db / 20) - delta_loss / 2
    zeta1 = float(np.clip(zeta1, 0.0, 1.0))
    zeta2 = float(np.clip(zeta2, 0.0, 1.0))

    rho1, rho2, *_ = prepare_density_matrices(
        b_A, b_B, n_truncation, bit0_parameters, bit1_parameters, state
    )

    rho1p = apply_loss_channel(rho1, zeta1, n_truncation)
    rho2p = apply_loss_channel(rho2, zeta2, n_truncation)
    return charlie_click_probabilities(rho1p, rho2p, n_truncation)


def detectors_probability_fast(
    b_A: int,
    b_B: int,
    loss_db: float,
    n_truncation: int,
    parameters,
    state: str,
):
    """Fast symmetric detector probabilities P00, P10, F_plus, F_minus."""
    zeta = 10 ** (-loss_db / 20)
    rho1, rho2, F_plus, F_minus = prepare_density_matrices_fast(
        b_A, b_B, n_truncation, parameters, state
    )
    rho1p = apply_loss_channel(rho1, zeta, n_truncation)
    rho2p = apply_loss_channel(rho2, zeta, n_truncation)
    P00, _P01, P10, _P11 = charlie_click_probabilities(rho1p, rho2p, n_truncation)
    return P00, P10, F_plus, F_minus


def p_zz_fast(n_A: int, n_B: int, loss_db: float, pd: float) -> float:
    """Closed-form fast ZZ click probability from the MATLAB p_ZZ_fast."""
    eta = 10 ** (-loss_db / 10)
    zeta = math.sqrt(eta)
    q_zz = 0.0

    for k1 in range(n_A + 1):
        for k2 in range(n_B + 1):
            q_zz += (
                math.comb(n_A, k1)
                * math.comb(n_B, k2)
                * math.comb(k1 + k2, k1)
                * zeta ** (k1 + k2)
                * (1 - zeta) ** (n_A + n_B - k1 - k2)
                / 2 ** (k1 + k2)
            )

    q_zz -= (1 - zeta) ** (n_A + n_B)
    return float((1 - pd) * (pd * (1 - zeta) ** (n_A + n_B) + q_zz))


def p_zz_probability(
    n_A: int,
    n_B: int,
    loss_db: float,
    pd: float,
    n_truncation: int,
    errors=(0.0, 0.0, 0.0),
):
    """General ZZ probabilities p_ZZ_01, p_ZZ_10 from Fock inputs."""
    from .operators import number_state_ket

    ket1 = number_state_ket(n_A, n_truncation)
    ket2 = number_state_ket(n_B, n_truncation)
    rho1 = ket1 @ ket1.conj().T
    rho2 = ket2 @ ket2.conj().T

    delta_loss, _delta_bs, _delta_phase = errors
    zeta1 = float(np.clip(10 ** (-loss_db / 20) + delta_loss / 2, 0.0, 1.0))
    zeta2 = float(np.clip(10 ** (-loss_db / 20) - delta_loss / 2, 0.0, 1.0))

    rho1p = apply_loss_channel(rho1, zeta1, n_truncation)
    rho2p = apply_loss_channel(rho2, zeta2, n_truncation)
    P00, P01, P10, _P11 = charlie_click_probabilities(rho1p, rho2p, n_truncation)

    p_zz_10 = (1 - pd) * (pd * P00 + P10)
    p_zz_01 = (1 - pd) * (pd * P00 + P01)
    return float(p_zz_01), float(p_zz_10)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    N = 5
    pd = 1e-8

    print("\n=== 1) detectors_probability: VSS full path ===")
    probs = detectors_probability(
        b_A=0,
        b_B=1,
        loss_db=0.0,
        n_truncation=N,
        bit0_parameters=[0.85, 0.0],
        bit1_parameters=[0.85, np.pi],
        state="VSS",
    )

    print("P00, P01, P10, P11 =", probs)

    assert len(probs) == 4
    assert all(-1e-12 <= p <= 1 + 1e-12 for p in probs)
    assert np.isclose(sum(probs), 1.0, atol=1e-10)

    print("\n=== 2) detectors_probability: asymmetric loss error ===")
    probs_error = detectors_probability(
        b_A=0,
        b_B=1,
        loss_db=3.0,
        n_truncation=N,
        bit0_parameters=[0.85, 0.0],
        bit1_parameters=[0.85, np.pi],
        state="VSS",
        errors=(0.05, 0.0, 0.0),
    )

    print("With delta_loss = 0.05:")
    print("P00, P01, P10, P11 =", probs_error)

    assert len(probs_error) == 4
    assert all(-1e-12 <= p <= 1 + 1e-12 for p in probs_error)
    assert np.isclose(sum(probs_error), 1.0, atol=1e-10)

    print("\n=== 3) detectors_probability_fast: VSS symmetric path ===")
    P00_fast, P10_fast, Fp, Fm = detectors_probability_fast(
        b_A=0,
        b_B=1,
        loss_db=0.0,
        n_truncation=N,
        parameters=[0.85, 0.0, np.pi],
        state="VSS",
    )

    print("P00_fast =", P00_fast)
    print("P10_fast =", P10_fast)
    print("F_plus shape:", Fp.shape)
    print("F_minus shape:", Fm.shape)

    assert 0.0 <= P00_fast <= 1.0
    assert 0.0 <= P10_fast <= 1.0
    assert Fp.shape == (N + 1, N + 1)
    assert Fm.shape == (N + 1, N + 1)

    print("\n=== 4) Fast path should match general path when symmetric ===")
    probs_general = detectors_probability(
        b_A=0,
        b_B=1,
        loss_db=0.0,
        n_truncation=N,
        bit0_parameters=[0.85, 0.0],
        bit1_parameters=[0.85, np.pi],
        state="VSS",
    )

    assert np.isclose(P00_fast, probs_general[0], atol=1e-12)
    assert np.isclose(P10_fast, probs_general[2], atol=1e-12)

    print("Fast P00/P10 match general P00/P10.")

    print("\n=== 5) p_zz_probability: vacuum + vacuum ===")
    p01_vac, p10_vac = p_zz_probability(
        n_A=0,
        n_B=0,
        loss_db=0.0,
        pd=pd,
        n_truncation=N,
    )

    print("p_ZZ_01, p_ZZ_10 =", p01_vac, p10_vac)
    print("Expected dark-count-only value ≈ (1-pd)*pd")

    expected_dark = (1 - pd) * pd

    assert np.isclose(p01_vac, expected_dark, rtol=1e-8)
    assert np.isclose(p10_vac, expected_dark, rtol=1e-8)

    print("\n=== 6) p_zz_probability: one photon + vacuum ===")
    p01_10, p10_10 = p_zz_probability(
        n_A=1,
        n_B=0,
        loss_db=0.0,
        pd=pd,
        n_truncation=N,
    )

    print("p_ZZ_01, p_ZZ_10 =", p01_10, p10_10)

    # At zero loss, one photon entering a 50:50 beam splitter exits either detector with prob 1/2.
    assert np.isclose(p01_10, 0.5 * (1 - pd), atol=1e-8)
    assert np.isclose(p10_10, 0.5 * (1 - pd), atol=1e-8)

    print("\n=== 7) p_zz_probability: HOM case |1,1> ===")
    p01_11, p10_11 = p_zz_probability(
        n_A=1,
        n_B=1,
        loss_db=0.0,
        pd=pd,
        n_truncation=N,
    )

    print("p_ZZ_01, p_ZZ_10 =", p01_11, p10_11)
    print("Expected: no coincidence, but single-detector bunching events remain.")

    assert p01_11 >= 0.0
    assert p10_11 >= 0.0
    assert np.isclose(p01_11 + p10_11, 1.0 - pd, atol=1e-8)

    print("\n=== 8) p_zz_fast vs p_zz_probability consistency ===")
    n_A = 1
    n_B = 1
    loss_db = 0.0

    p_fast = p_zz_fast(n_A, n_B, loss_db, pd)
    p01_general, p10_general = p_zz_probability(
        n_A=n_A,
        n_B=n_B,
        loss_db=loss_db,
        pd=pd,
        n_truncation=N,
    )

    print("p_zz_fast =", p_fast)
    print("p_ZZ_01 general =", p01_general)
    print("p_ZZ_10 general =", p10_general)

    # For symmetric loss and symmetric inputs, both output ports should match.
    assert np.isclose(p01_general, p10_general, atol=1e-12)

    # p_zz_fast corresponds to one of the two single-click output probabilities.
    assert np.isclose(p_fast, p10_general, atol=1e-12)

    print("\n=== 9) p_zz_fast: loss increases vacuum/no-click contribution ===")
    p_fast_0db = p_zz_fast(1, 0, loss_db=0.0, pd=pd)
    p_fast_10db = p_zz_fast(1, 0, loss_db=10.0, pd=pd)

    print("p_zz_fast at 0 dB:", p_fast_0db)
    print("p_zz_fast at 10 dB:", p_fast_10db)

    assert p_fast_10db < p_fast_0db

    print("\ndetectors.py sanity checks passed")