from scipy.integrate import quad
import numpy as np
from pathlib import Path
from scipy.integrate import trapezoid, simpson
from load_data import load_last_gw_spectrum_f, spectrum_peak
from fit_GW_spectra import fit_template, gw_template


class GW_analysis:
    def __init__(self, m_over_H, H_star, T_rh, g_star=106.75):
        self.m_over_H = m_over_H
        self.m = self.m_over_H * H_star
        self.H_star = H_star
        self.T0 = 2.352531980526e-13  # GeV
        self.T_rh = T_rh
        self.g_star = g_star
        self.g_0 = 3.91
        self.redshift_freq_factor = 1.65e-7 * self.T_rh * (self.g_star / 100) ** (1 / 6)
        self.redshift_amplitude_factor = 1.67e-5 * (100 / self.g_star) ** (1 / 3)

    def GW_peak_amplitude(self):
        return 5.58e-5 * (100 / self.g_star) ** (1 / 3) * self.m_over_H ** (-2)

    def GW_peak_frequency(self):
        kappa_peak = 1.21
        return (
            2.62e-8
            * kappa_peak
            * self.m_over_H
            * self.T_rh
            * (self.g_star / 100) ** (1 / 6)
        )

    def GW_template(self, f):
        kappa_from_f_0 = (
            f
            / (2.62e-8)
            * self.m_over_H ** (-1)
            / self.T_rh
            * (100 / self.g_star) ** (1 / 6)
        )
        kappa_peak = 1.21
        gamma = 1.43
        p = 0.52
        S = gw_template(kappa_from_f_0, kappa_peak, gamma, p)
        return 5.58e-5 * (100 / self.g_star) ** (1 / 3) * self.m_over_H ** (-2) * S

    def SNR(self, t_obs, sens_curve, fmin, fmax):
        """
        computes SNR (sensitivity curve of experiment is an input)
        """
        # res = quad(lambda f: (self.GW_template(f)/sens_curve(f))**2, fmin, fmax, epsrel = 1e-20,epsabs=1e-20)[0]
        x = np.logspace(np.log10(fmin), np.log10(fmax), 10000)
        sens = np.array([sens_curve(i) for i in x])
        y = (self.GW_template(x) / sens) ** 2
        res = simpson(y, x)
        # simp_result = simpson(y, x)
        return np.sqrt(2 * t_obs * res)

    def SNR_auto(self, t_obs, sens_curve, fmin, fmax):
        """
        computes SNR (auto-correlated)
        """
        # res = quad(lambda f: (self.GW_template(f)/sens_curve(f))**2, fmin, fmax, epsrel = 1e-30,epsabs=1e-30)[0]
        x = np.logspace(np.log10(fmin), np.log10(fmax), 10000)
        sens = np.array([sens_curve(i) for i in x])

        y = (self.GW_template(x) / sens) ** 2
        res = simpson(y, x)
        return np.sqrt(t_obs * res)
