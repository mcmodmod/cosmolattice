from m_over_H import lam_from_mu
from pathlib import Path
import numpy as np

M_PL = 2.435e18


class Simulation:
    mus = {
        1e1: M_PL * 10 ** (-6),
        1e2: M_PL * 10 ** (-7),
        1e3: M_PL * 10 ** (-7),
        1e4: M_PL * 10 ** (-8),
        1e5: M_PL * 10 ** (-9),
        1e6: M_PL * 10 ** (-9),
        1e7: M_PL * 10 ** (-10),
        1e8: M_PL * 10 ** (-11),
        1e9: M_PL * 10 ** (-11),
    }

    def __init__(self, input_dir: Path, output_dir: Path, m_over_H: float):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.m_over_H = m_over_H
        self.mu = self.mus[m_over_H]
        self.lam = lam_from_mu(self.mu, self.m_over_H)
        self.omega_star = self.mu
        self.f_star = self.mu / np.sqrt(self.lam)
        self.H = self.mu**2 / (np.sqrt(12) * M_PL * np.sqrt(self.lam))
        self.vev = self.mu / np.sqrt(self.lam)


if __name__ == "__main__":
    sim = Simulation(Path("../output/test1"), Path("figures/test1"), 1e3)
