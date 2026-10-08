import numpy as np


class HaarWavelet:
    def __init__(self, u: float, a: float, dt: float = 1.0):
        self.u = float(u)  # Translation (time anchor)
        self.a = float(a)  # Scale (width)
        self.dt = float(dt)  # Sampling step

    def transform_kernel(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluates psi_{u, a}(t) on time grid t.
        Haar wavelet: +1/sqrt(a) on [u, u + a/2), -1/sqrt(a) on [u + a/2, u + a)
        """
        norm = 1.0 / np.sqrt(self.a)
        midpoint = self.u + (self.a / 2.0)

        # Vectorized piece-wise step
        psi = np.zeros_like(t, dtype=np.float64)
        psi[(t >= self.u) & (t < midpoint)] = norm
        psi[(t >= midpoint) & (t < self.u + self.a)] = -norm
        return psi

    def forward(self, signal: np.ndarray, t: np.ndarray = None) -> float:
        """
        Computes the CWT coefficient W(u, a) = \int signal(t) * psi_{u, a}(t) dt
        """
        if t is None:
            # Default time grid based on signal length and dt
            t = np.arange(len(signal)) * self.dt

        kernel = self.transform_kernel(t)
        # Numerical integration: sum(signal * psi) * dt
        return float(np.sum(signal * kernel) * self.dt)