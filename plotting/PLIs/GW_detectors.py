from scipy.interpolate import interp1d
import numpy as np


class GW_detectors:

    def __init__(self, name):
        self.name = name
        # self.f = f

        self.et_data = np.genfromtxt("PLIs/ET-0000A-18_ETDSensitivityCurveTxtFile.txt")
        self.et_freq = np.array([a[0] for a in self.et_data])
        self.test3 = np.array([a[3] for a in self.et_data])
        self.et_interp = interp1d(self.et_freq, self.test3, fill_value="extrapolate")

    def spectral_density(self, f):
        import numpy as np

        self.f = f
        if self.name == "lisa":
            L = 2.5e9  # m
            f_star = 299792458 / (2 * np.pi * L)  # Hz
            P_oms = (1.5e-11) ** 2 * (1 + (2e-3 / self.f) ** 4)  # m**2 Hz**-1
            P_acc = (
                (3e-15) ** 2 * (1 + (0.4e-3 / self.f) ** 2) * (1 + (self.f / 8e-3) ** 4)
            )  # m**2 s**-3
            S_c = (
                9e-45
                * self.f ** -(7 / 3)
                * np.exp(-((self.f) ** 0.138) - 221 * self.f * np.sin(521 * self.f))
                * (1 + np.tanh(1680 * (0.0013 - self.f)))
            )
            S_lisa = (
                10
                / (3 * L**2)
                * (
                    P_oms
                    + 2
                    * (1 + np.cos(self.f / f_star) ** 2)
                    * (P_acc / (2 * np.pi * self.f) ** 4)
                )
                * (1 + (6 / 10) * (self.f / f_star) ** 2)
                + S_c
            )
            return (
                0.674**2
                * 2
                * np.pi**2
                / (3 * (100e3 / 3.086e22) ** 2)
                * self.f**3
                * S_lisa
            )

        elif self.name == "bdecigo":
            S_b_decigo = 2.020e-45 * (
                1 + 1.584e-2 * self.f ** (-4) + 1.584e-3 * self.f**2
            )
            return 4 * np.pi**2 / (3 * (100e3 / 3.086e22) ** 2) * self.f**3 * S_b_decigo

        elif self.name == "decigo" or self.name == "bbo":
            temp = []
            T = 4 * 365 * 24 * 3600
            F = np.exp(-2 * (self.f / 0.05) ** 2)

            f_p = 7.36
            dN_df = 2e-3 * self.f ** (-11 / 3)
            k = 4.5

            S_NS = 1.3e-48 * self.f ** (-7 / 3) * 1e-1
            S_ex_gal = 4.2e-47 * self.f ** (-7 / 3)
            S_gal = 2.1e-45 * self.f ** (-7 / 3)

            if self.name == "decigo":
                S_decigo = 5.3e-48 * (
                    (1 + (self.f / f_p) ** 2)
                    + 2.3e-7 * (self.f / f_p) ** (-4) * (1 / (1 + (self.f / f_p) ** 2))
                    + 2.6e-8 * (self.f / f_p) ** (-4)
                )
                A = min(
                    [S_decigo / np.exp(-k * T ** (-1) * dN_df), S_decigo + S_gal * F]
                )

            else:
                S_bbo = 1.8e-49 * self.f**2 + 2.9e-49 + 9.2e-52 * self.f**-4
                A = min([S_bbo / np.exp(-k * T ** (-1) * dN_df), S_bbo + S_gal * F])

            temp = (
                0.674**2
                * (2 * np.pi**2 * self.f**3 / (3 * (100e3 / 3.086e22) ** 2))
                * (5 * A + S_ex_gal * F)
            )
            return np.array(temp)

        elif self.name == "ska":
            sigma = 100e-9
            delta_t = 14 * 24 * 3600
            N_p = 1000
            H_0 = 0.674 * 100 * 3.24e-20
            return (
                0.674**2
                * np.sqrt(2 / (N_p * (N_p - 1)))
                * (64 * np.sqrt(3) * np.pi**4 * sigma**2 * delta_t * self.f**5)
                / H_0**2
            )

        elif self.name == "et":
            # et_data = np.genfromtxt('ET-0000A-18_ETDSensitivityCurveTxtFile.txt')
            # freq = np.array([a[0] for a in et_data])
            # test3 = np.array([a[3] for a in et_data])

            # return 0.674**2 * 2*np.pi**2/(3*(100e3/3.086e22)**2) * self.freq**3 * self.test3**2
            return (
                0.674**2
                * 2
                * np.pi**2
                / (3 * (100e3 / 3.086e22) ** 2)
                * self.f**3
                * self.et_interp(self.f) ** 2
            )
