
"""X-basis and known-click probability utilities."""
from __future__ import annotations

from .detectors import detectors_probability


def known_bits_probability(
    kc: int,
    kd: int,
    b_A: int,
    b_B: int,
    loss_db: float,
    n_truncation: int,
    bit0_parameters,
    bit1_parameters,
    state: str,
    errors=(0.0, 0.0, 0.0),
    pd: float = 0.0,
) -> float:
    P00, P01, P10, P11 = detectors_probability(
        b_A, b_B, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors
    )
    q_xx = (
        (1 - kc) * (1 - kd) * P00
        + kc * (1 - kd) * P10
        + (1 - kc) * kd * P01
        + kc * kd * P11
    )

    return float(
        (1 - pd) ** 2 * q_xx
        + kc * (1 - kd) * pd * (1 - pd) * (P00 + P10)
        + (1 - kc) * kd * pd * (1 - pd) * (P00 + P01)
        + kc * kd * pd**2
    )


def p_xx_probability(
    kc: int,
    kd: int,
    loss_db: float,
    n_truncation: int,
    bit0_parameters,
    bit1_parameters,
    state: str,
    errors=(0.0, 0.0, 0.0),
    pd: float = 0.0,
) -> float:
    p_xx = 0.0
    for b_A in (0, 1):
        for b_B in (0, 1):
            p_xx += known_bits_probability(
                kc, kd, b_A, b_B, loss_db, n_truncation,
                bit0_parameters, bit1_parameters, state, errors, pd
            ) / 4
    return float(p_xx)


def known_clicks_probability(
    b_A: int,
    b_B: int,
    kc: int,
    kd: int,
    loss_db: float,
    n_truncation: int,
    bit0_parameters,
    bit1_parameters,
    state: str,
    errors=(0.0, 0.0, 0.0),
    pd: float = 0.0,
) -> float:
    numerator = known_bits_probability(
        kc, kd, b_A, b_B, loss_db, n_truncation,
        bit0_parameters, bit1_parameters, state, errors, pd
    ) / 4
    denominator = p_xx_probability(
        kc, kd, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def x_error(
    loss_db: float,
    n_truncation: int,
    bit0_parameters,
    bit1_parameters,
    state: str,
    errors=(0.0, 0.0, 0.0),
    pd: float = 0.0,
):
    e_x01 = known_clicks_probability(0, 0, 0, 1, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)
    e_x01 += known_clicks_probability(1, 1, 0, 1, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)

    e_x10 = known_clicks_probability(0, 1, 1, 0, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)
    e_x10 += known_clicks_probability(1, 0, 1, 0, loss_db, n_truncation, bit0_parameters, bit1_parameters, state, errors, pd)
    return float(e_x01), float(e_x10)


if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(precision=5, suppress=True)

    N = 6
    bit0 = [0.85, 0.0]
    bit1 = [0.85, np.pi]
    loss_db = 0.0
    pd = 1e-8

    print("\n=== 1) known_bits_probability: fixed bits and clicks ===")
    p_kb = known_bits_probability(
        kc=0,
        kd=1,
        b_A=0,
        b_B=1,
        loss_db=loss_db,
        n_truncation=N,
        bit0_parameters=bit0,
        bit1_parameters=bit1,
        state="VSS",
        pd=pd,
    )

    print("P(kc=0, kd=1 | bA=0, bB=1) =", p_kb)

    assert 0.0 <= p_kb <= 1.0

    print("\n=== 2) known_bits_probability: click probabilities sum to 1 for fixed bits ===")
    total_fixed_bits = 0.0

    for kc in (0, 1):
        for kd in (0, 1):
            p = known_bits_probability(
                kc=kc,
                kd=kd,
                b_A=0,
                b_B=1,
                loss_db=loss_db,
                n_truncation=N,
                bit0_parameters=bit0,
                bit1_parameters=bit1,
                state="VSS",
                pd=pd,
            )
            print(f"P(kc={kc}, kd={kd} | bA=0, bB=1) = {p:.8f}")
            total_fixed_bits += p

    print("Total over clicks =", total_fixed_bits)

    assert np.isclose(total_fixed_bits, 1.0, atol=1e-10)

    print("\n=== 3) p_xx_probability: averaged over all bit choices ===")
    p_xx_01 = p_xx_probability(
        kc=0,
        kd=1,
        loss_db=loss_db,
        n_truncation=N,
        bit0_parameters=bit0,
        bit1_parameters=bit1,
        state="VSS",
        pd=pd,
    )

    print("P_X X(kc=0, kd=1) =", p_xx_01)

    assert 0.0 <= p_xx_01 <= 1.0

    print("\n=== 4) p_xx_probability: all click outcomes sum to 1 ===")
    total_xx = 0.0

    for kc in (0, 1):
        for kd in (0, 1):
            p = p_xx_probability(
                kc=kc,
                kd=kd,
                loss_db=loss_db,
                n_truncation=N,
                bit0_parameters=bit0,
                bit1_parameters=bit1,
                state="VSS",
                pd=pd,
            )
            print(f"P_X X(kc={kc}, kd={kd}) = {p:.8f}")
            total_xx += p

    print("Total averaged click probability =", total_xx)

    assert np.isclose(total_xx, 1.0, atol=1e-10)

    print("\n=== 5) known_clicks_probability: posterior probability ===")
    posterior = known_clicks_probability(
        b_A=0,
        b_B=1,
        kc=0,
        kd=1,
        loss_db=loss_db,
        n_truncation=N,
        bit0_parameters=bit0,
        bit1_parameters=bit1,
        state="VSS",
        pd=pd,
    )

    print("P(bA=0,bB=1 | kc=0,kd=1) =", posterior)

    assert 0.0 <= posterior <= 1.0

    print("\n=== 6) known_clicks_probability: posterior over bits sums to 1 for fixed clicks ===")
    total_posterior = 0.0

    for b_A in (0, 1):
        for b_B in (0, 1):
            p = known_clicks_probability(
                b_A=b_A,
                b_B=b_B,
                kc=0,
                kd=1,
                loss_db=loss_db,
                n_truncation=N,
                bit0_parameters=bit0,
                bit1_parameters=bit1,
                state="VSS",
                pd=pd,
            )
            print(f"P(bA={b_A}, bB={b_B} | kc=0,kd=1) = {p:.8f}")
            total_posterior += p

    print("Total posterior probability =", total_posterior)

    assert np.isclose(total_posterior, 1.0, atol=1e-10)

    print("\n=== 7) x_error: X-basis error probabilities ===")
    ex01, ex10 = x_error(
        loss_db=loss_db,
        n_truncation=N,
        bit0_parameters=bit0,
        bit1_parameters=bit1,
        state="VSS",
        pd=pd,
    )

    print("e_X01 =", ex01)
    print("e_X10 =", ex10)

    assert 0.0 <= ex01 <= 1.0
    assert 0.0 <= ex10 <= 1.0

    print("\n=== 8) Symmetric system check: e_X01 ≈ e_X10 ===")
    print("Difference:", abs(ex01 - ex10))

    assert np.isclose(ex01, ex10, atol=1e-10)

    print("\n=== 9) Dark-count effect check ===")
    ex_no_dark = x_error(
        loss_db=loss_db,
        n_truncation=N,
        bit0_parameters=bit0,
        bit1_parameters=bit1,
        state="VSS",
        pd=0.0,
    )

    ex_with_dark = x_error(
        loss_db=loss_db,
        n_truncation=N,
        bit0_parameters=bit0,
        bit1_parameters=bit1,
        state="VSS",
        pd=1e-4,
    )

    print("x_error with pd=0:", ex_no_dark)
    print("x_error with pd=1e-4:", ex_with_dark)

    assert all(0.0 <= v <= 1.0 for v in ex_no_dark)
    assert all(0.0 <= v <= 1.0 for v in ex_with_dark)

    print("\nprobabilities.py sanity checks passed")