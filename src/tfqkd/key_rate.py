
"""TF-QKD key-rate functions."""
from __future__ import annotations

import numpy as np

from .detectors import detectors_probability_fast, p_zz_fast, p_zz_probability
from .operators import number_state_ket
from .probabilities import p_xx_probability, x_error


def binary_entropy(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def z_error(
    loss_db: float,
    n_truncation: int,
    bit0_parameters,
    bit1_parameters,
    state: str,
    errors=(0.0, 0.0, 0.0),
    pd: float = 0.0,
):
    from .states import vss_filters, gaussian_filters

    state = state.upper()

    if state == "VSS":
        q0, phi0 = bit0_parameters
        q1, phi1 = bit1_parameters
        F_plus, F_minus = vss_filters(q0, phi0, q1, phi1, n_truncation)
        eps0 = eps1 = 0.0
    elif state == "GS":
        eps0, b0, phi0, s0, theta0 = bit0_parameters
        eps1, b1, phi1, s1, theta1 = bit1_parameters
        F_plus, F_minus = gaussian_filters(b0, phi0, s0, theta0, b1, phi1, s1, theta1, n_truncation)
    else:
        raise ValueError(f"Unknown state: {state!r}")

    term01_1 = 0.0
    term10_1 = 0.0

    for j in (0, 1):
        term01_0 = 0.0
        term10_0 = 0.0
        F_j = 0.5 * (F_plus + F_minus) if j == 0 else 0.5 * (F_plus - F_minus)

        if state == "GS" and (eps0 != 0 or eps1 != 0):
            epsilon_j = eps0 if j == 0 else eps1
            for n_A in range(n_truncation + 1):
                ket_n_A = number_state_ket(n_A, n_truncation)
                for n_B in range(n_truncation + 1):
                    ket_n_B = number_state_ket(n_B, n_truncation)
                    p01, p10 = p_zz_probability(n_A, n_B, loss_db, pd, n_truncation, errors)
                    for m_A in range(n_truncation + 1):
                        ket_m_A = number_state_ket(m_A, n_truncation)
                        a_amp = (ket_n_A.conj().T @ F_j @ ket_m_A)[0, 0]
                        for m_B in range(n_truncation + 1):
                            ket_m_B = number_state_ket(m_B, n_truncation)
                            b_amp = (ket_n_B.conj().T @ F_j @ ket_m_B)[0, 0]
                            amp = abs((epsilon_j ** (m_A + m_B)) * a_amp * b_amp)
                            term01_0 += amp * np.sqrt(max(p01, 0.0))
                            term10_0 += amp * np.sqrt(max(p10, 0.0))
        else:
            ket_vac = number_state_ket(0, n_truncation)
            for n_A in range(n_truncation + 1):
                ket_n_A = number_state_ket(n_A, n_truncation)
                a_amp = (ket_n_A.conj().T @ F_j @ ket_vac)[0, 0]
                for n_B in range(n_truncation + 1):
                    ket_n_B = number_state_ket(n_B, n_truncation)
                    b_amp = (ket_n_B.conj().T @ F_j @ ket_vac)[0, 0]
                    p01, p10 = p_zz_probability(n_A, n_B, loss_db, pd, n_truncation, errors)
                    amp = abs(a_amp * b_amp)
                    term01_0 += amp * np.sqrt(max(p01, 0.0))
                    term10_0 += amp * np.sqrt(max(p10, 0.0))

        term01_1 += abs(term01_0) ** 2
        term10_1 += abs(term10_0) ** 2

    p_xx01 = p_xx_probability(0, 1, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)
    p_xx10 = p_xx_probability(1, 0, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)

    return float(term01_1 / p_xx01), float(term10_1 / p_xx10)


def tf_qkd(
    loss_db: float,
    n_truncation: int,
    bit0_parameters,
    bit1_parameters,
    state: str,
    errors=(0.0, 0.0, 0.0),
    pd: float = 0.0,
    f: float = 1.16,
):
    e_x01, e_x10 = x_error(loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)
    e_z01, e_z10 = z_error(loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)
    p_xx01 = p_xx_probability(0, 1, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)
    p_xx10 = p_xx_probability(1, 0, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)

    r01 = p_xx01 * (1 - f * binary_entropy(e_x01) - binary_entropy(min(0.5, e_z01)))
    r10 = p_xx10 * (1 - f * binary_entropy(e_x10) - binary_entropy(min(0.5, e_z10)))
    return float(r01 + r10), float(e_x01), float(e_z01), float(p_xx01), float(e_x10), float(e_z10), float(p_xx10)


def tf_qkd_fast(
    loss_db: float,
    n_truncation: int,
    parameters,
    state: str,
    pd: float = 0.0,
    f: float = 1.16,
):
    p_xx = 0.0
    saved = None
    for b_A in (0, 1):
        for b_B in (0, 1):
            P00, P10, Fp, Fm = detectors_probability_fast(b_A, b_B, loss_db, n_truncation, parameters, state)
            p_xx += (1 - pd) * (pd * P00 + P10) / 4
            saved = (Fp, Fm)

    e_x = 0.0
    for b_A, b_B in ((1, 0), (0, 1)):
        P00, P10, Fp, Fm = detectors_probability_fast(b_A, b_B, loss_db, n_truncation, parameters, state)
        e_x += (1 - pd) * (pd * P00 + P10) / p_xx / 4
        saved = (Fp, Fm)

    F_plus, F_minus = saved
    state = state.upper()
    epsilon = parameters[0] if state == "GS" else 0.0

    term1 = 0.0
    for j in (0, 1):
        term0 = 0.0
        F_j = 0.5 * (F_plus + F_minus) if j == 0 else 0.5 * (F_plus - F_minus)

        if state == "GS" and epsilon != 0:
            for n_A in range(n_truncation + 1):
                ket_n_A = number_state_ket(n_A, n_truncation)
                for n_B in range(n_truncation + 1):
                    ket_n_B = number_state_ket(n_B, n_truncation)
                    sqrt_p = np.sqrt(max(p_zz_fast(n_A, n_B, loss_db, pd), 0.0))
                    for m_A in range(n_truncation + 1):
                        ket_m_A = number_state_ket(m_A, n_truncation)
                        a_amp = (ket_n_A.conj().T @ F_j @ ket_m_A)[0, 0]
                        for m_B in range(n_truncation + 1):
                            ket_m_B = number_state_ket(m_B, n_truncation)
                            b_amp = (ket_n_B.conj().T @ F_j @ ket_m_B)[0, 0]
                            term0 += abs((epsilon ** (m_A + m_B)) * a_amp * b_amp) * sqrt_p
        else:
            ket_vac = number_state_ket(0, n_truncation)
            for n_A in range(n_truncation + 1):
                ket_n_A = number_state_ket(n_A, n_truncation)
                a_amp = (ket_n_A.conj().T @ F_j @ ket_vac)[0, 0]
                for n_B in range(n_truncation + 1):
                    ket_n_B = number_state_ket(n_B, n_truncation)
                    b_amp = (ket_n_B.conj().T @ F_j @ ket_vac)[0, 0]
                    term0 += abs(a_amp * b_amp) * np.sqrt(max(p_zz_fast(n_A, n_B, loss_db, pd), 0.0))

        term1 += abs(term0) ** 2

    e_z = term1 / p_xx
    r_xx = 2 * p_xx * (1 - f * binary_entropy(e_x) - binary_entropy(min(0.5, e_z)))
    r_xx = np.real_if_close(r_xx)
    if np.real(r_xx) <= 0 or abs(np.imag(r_xx)) >= 1e-15:
        r_xx = 1e-20
    return float(np.real(r_xx)), float(e_x), float(e_z), float(p_xx)


if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(precision=6, suppress=True)

    N = 4
    pd = 1e-8
    f = 1.16

    bit0_vss = [0.85, 0.0]
    bit1_vss = [0.85, np.pi]
    params_vss = [0.85, 0.0, np.pi]

    print("\n=== 1) binary_entropy checks ===")

    h0 = binary_entropy(0.0)
    h_half = binary_entropy(0.5)
    h1 = binary_entropy(1.0)

    print("H(0)   =", h0)
    print("H(0.5) =", h_half)
    print("H(1)   =", h1)

    assert np.isclose(h_half, 1.0, atol=1e-12)
    assert h0 < 1e-10
    assert h1 < 1e-10

    print("\n=== 2) z_error: VSS symmetric case ===")

    ez01, ez10 = z_error(
        loss_db=0.0,
        n_truncation=N,
        bit0_parameters=bit0_vss,
        bit1_parameters=bit1_vss,
        state="VSS",
        pd=pd,
    )

    print("e_Z01 =", ez01)
    print("e_Z10 =", ez10)

    assert 0.0 <= ez01 <= 1.5
    assert 0.0 <= ez10 <= 1.5
    assert np.isclose(ez01, ez10, atol=1e-10)

    print("\n=== 3) tf_qkd full version: VSS ===")

    result_full = tf_qkd(
        loss_db=0.0,
        n_truncation=N,
        bit0_parameters=bit0_vss,
        bit1_parameters=bit1_vss,
        state="VSS",
        pd=pd,
        f=f,
    )

    r, ex01, ez01, pxx01, ex10, ez10, pxx10 = result_full

    print("Full result:")
    print("R      =", r)
    print("e_X01  =", ex01)
    print("e_Z01  =", ez01)
    print("P_XX01 =", pxx01)
    print("e_X10  =", ex10)
    print("e_Z10  =", ez10)
    print("P_XX10 =", pxx10)

    assert len(result_full) == 7
    assert r >= 0.0 or np.isclose(r, 0.0)
    assert 0.0 <= ex01 <= 1.0
    assert 0.0 <= ex10 <= 1.0
    assert 0.0 <= pxx01 <= 1.0
    assert 0.0 <= pxx10 <= 1.0
    assert np.isclose(ex01, ex10, atol=1e-10)
    assert np.isclose(ez01, ez10, atol=1e-10)
    assert np.isclose(pxx01, pxx10, atol=1e-10)

    print("\n=== 4) tf_qkd_fast: VSS ===")

    result_fast = tf_qkd_fast(
        loss_db=0.0,
        n_truncation=N,
        parameters=params_vss,
        state="VSS",
        pd=pd,
        f=f,
    )

    r_fast, ex_fast, ez_fast, pxx_fast = result_fast

    print("Fast result:")
    print("R     =", r_fast)
    print("e_X   =", ex_fast)
    print("e_Z   =", ez_fast)
    print("P_XX  =", pxx_fast)

    assert len(result_fast) == 4
    assert r_fast >= 0.0
    assert 0.0 <= ex_fast <= 1.0
    assert 0.0 <= pxx_fast <= 1.0

    print("\n=== 5) full vs fast consistency: symmetric VSS ===")

    print("Full R:", r)
    print("Fast R:", r_fast)

    print("Full e_X avg:", 0.5 * (ex01 + ex10))
    print("Fast e_X:", ex_fast)

    print("Full e_Z avg:", 0.5 * (ez01 + ez10))
    print("Fast e_Z:", ez_fast)

    print("Full P_XX avg:", 0.5 * (pxx01 + pxx10))
    print("Fast P_XX:", pxx_fast)

    assert np.isclose(ex_fast, 0.5 * (ex01 + ex10), atol=1e-8)
    assert np.isclose(ez_fast, 0.5 * (ez01 + ez10), atol=1e-8)
    assert np.isclose(pxx_fast, 0.5 * (pxx01 + pxx10), atol=1e-8)

    print("\n=== 6) loss trend check ===")

    r0, *_ = tf_qkd_fast(0.0, N, params_vss, "VSS", pd=pd, f=f)
    r10, *_ = tf_qkd_fast(10.0, N, params_vss, "VSS", pd=pd, f=f)
    r20, *_ = tf_qkd_fast(20.0, N, params_vss, "VSS", pd=pd, f=f)

    print("R at 0 dB  =", r0)
    print("R at 10 dB =", r10)
    print("R at 20 dB =", r20)

    assert r0 >= r10 or np.isclose(r0, r10)
    assert r10 >= r20 or np.isclose(r10, r20)

    print("\n=== 7) GS smoke test ===")

    params_gs = [0.02, 0.3, 0.0, np.pi, 0.1, 0.0, np.pi]

    r_gs, ex_gs, ez_gs, pxx_gs = tf_qkd_fast(
        loss_db=0.0,
        n_truncation=N,
        parameters=params_gs,
        state="GS",
        pd=pd,
        f=f,
    )

    print("GS fast result:")
    print("R    =", r_gs)
    print("e_X  =", ex_gs)
    print("e_Z  =", ez_gs)
    print("P_XX =", pxx_gs)

    assert r_gs >= 0.0
    assert 0.0 <= ex_gs <= 1.0
    assert pxx_gs >= 0.0

    print("\nkey_rate.py sanity checks passed")