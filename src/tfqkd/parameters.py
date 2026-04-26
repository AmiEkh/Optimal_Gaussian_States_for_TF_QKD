
"""Parameter containers for the TF-QKD simulations."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VSSParams:
    """Vacuum/single-photon superposition parameters."""
    q0: float
    phi0: float
    q1: float
    phi1: float

    @classmethod
    def from_vector(cls, x):
        return cls(q0=float(x[0]), phi0=float(x[1]), q1=float(x[2]), phi1=float(x[3]))

    def bit0(self) -> np.ndarray:
        return np.array([self.q0, self.phi0], dtype=float)

    def bit1(self) -> np.ndarray:
        return np.array([self.q1, self.phi1], dtype=float)

    def fast_vector(self) -> np.ndarray:
        # MATLAB fast convention: [q, phi0, phi1] with q0=q1=q.
        if not np.isclose(self.q0, self.q1):
            raise ValueError("fast VSS requires q0 == q1.")
        return np.array([self.q0, self.phi0, self.phi1], dtype=float)


@dataclass(frozen=True)
class GSParams:
    """Gaussian-state parameters for two bit states."""
    eps0: float
    b0: float
    phi0: float
    s0: float
    theta0: float
    eps1: float
    b1: float
    phi1: float
    s1: float
    theta1: float

    @classmethod
    def symmetric_from_vector(cls, x):
        # Convenient optimizer convention: [epsilon, b, phi0, phi1, s, theta0, theta1]
        return cls(float(x[0]), float(x[1]), float(x[2]), float(x[4]), float(x[5]),
                   float(x[0]), float(x[1]), float(x[3]), float(x[4]), float(x[6]))

    def bit0(self) -> np.ndarray:
        return np.array([self.eps0, self.b0, self.phi0, self.s0, self.theta0], dtype=float)

    def bit1(self) -> np.ndarray:
        return np.array([self.eps1, self.b1, self.phi1, self.s1, self.theta1], dtype=float)

    def fast_vector(self) -> np.ndarray:
        if not (np.isclose(self.eps0, self.eps1) and np.isclose(self.b0, self.b1) and np.isclose(self.s0, self.s1)):
            raise ValueError("fast GS expects shared epsilon, b, and s.")
        return np.array([self.eps0, self.b0, self.phi0, self.phi1, self.s0, self.theta0, self.theta1], dtype=float)


if __name__ == "__main__":
    print("\n=== VSS Example ===")

    # Symmetric VSS: same amplitude, phase difference = π (optimal case)
    vss = VSSParams(
        q0=0.8,
        phi0=0.0,
        q1=0.8,
        phi1=np.pi
    )

    print("Bit 0 params:", vss.bit0())
    print("Bit 1 params:", vss.bit1())
    print("Fast vector:", vss.fast_vector())

    # Check physical intuition
    assert np.isclose(vss.q0, vss.q1), "Amplitudes should match in symmetric VSS"
    assert np.isclose(abs(vss.phi0 - vss.phi1), np.pi), "Phase difference should be π"

    print("\n=== Gaussian State (GS) Example ===")

    # Typical TF-QKD optimal structure:
    # - same energy (epsilon), displacement (b), squeezing (s)
    # - phases differ by π
    # - squeezing aligned with displacement
    gs = GSParams(
        eps0=0.05,
        b0=0.4,
        phi0=0.0,
        s0=0.2,
        theta0=0.0,

        eps1=0.05,
        b1=0.4,
        phi1=np.pi,
        s1=0.2,
        theta1=np.pi
    )

    print("Bit 0 params:", gs.bit0())
    print("Bit 1 params:", gs.bit1())
    print("Fast vector:", gs.fast_vector())

    # Physics consistency checks
    assert np.isclose(gs.eps0, gs.eps1), "Energy mismatch"
    assert np.isclose(gs.b0, gs.b1), "Displacement mismatch"
    assert np.isclose(gs.s0, gs.s1), "Squeezing mismatch"
    assert np.isclose(abs(gs.phi0 - gs.phi1), np.pi), "Phase difference should be π"
    assert np.isclose(gs.theta0, gs.phi0), "Optimal squeezing alignment"
    assert np.isclose(gs.theta1, gs.phi1), "Optimal squeezing alignment"

    print("\nparameters.py sanity checks passed")