import numpy as np
from scipy.integrate import solve_ivp

# from scipy import interpolate

MPL = 2.435e18


class InputParameters:
    def __init__(self, gBL, mZPrime):
        """
        :param gBL: B-L gauge coupling at mu = m_Z'
        :param mZPrime: Z' mass at mu = m_Z'
        """
        self.gBL_at_mZPrime = gBL
        self.mZPrime = mZPrime
        self.lamPhi_at_mZPrime = (
            self.gBL_at_mZPrime**4 / np.pi**2
        )  # Coleman Weinberg relation
        self.vPhi_at_mZPrime = self.mZPrime / (2 * self.gBL_at_mZPrime)
        """
        constants
        """
        self.G_mu = 1.1663787e-5
        self.M_W = 80.379
        self.M_t = 172.76
        self.M_Z = 91.1876
        self.M_h = 125.1
        """
        SM input at the electroweak scale M_Z
        """
        self.g3_at_mZ = np.sqrt(1.48409)
        self.g2_at_mZ = np.sqrt(8 * self.M_W**2 * self.G_mu / np.sqrt(2))
        self.g1_at_mZ = np.sqrt(
            self.g2_at_mZ**2 / self.M_W**2 * (self.M_Z**2 - self.M_W**2)
        )
        self.yt_at_mZ = np.sqrt(self.g2_at_mZ**2 / 2 * self.M_t**2 / self.M_W**2)
        self.lamh_at_mZ = self.g2_at_mZ**2 / (16 * self.M_W**2) * 2 * self.M_h**2
        self.g1_at_mZPrime = solve_ivp(
            self.betaFunc_g1,
            y0=[self.g1_at_mZ],
            t_span=[self.M_Z, self.mZPrime],
            t_eval=np.linspace(self.M_Z, self.mZPrime, 10000),
            atol=1e-10,
            rtol=1e-10,
        ).y[:, -1][0]
        self.vh_at_mZ = np.sqrt(4 * self.M_W**2 / self.g2_at_mZ**2)
        """
        Determine the kinetic mixing g_tilde at mu=m_Z' such that it vanishes, g_tilde = 0 at mu = M_Z
        """
        self.gTilde_at_mZPrime = self.solve_at_mZ(1e-30)

        """
        These functions determine the B-L input parameters at the electroweak scale mu = M_Z
        """
        X0 = [
            self.g1_at_mZPrime,
            self.gBL_at_mZPrime,
            self.gTilde_at_mZPrime,
            self.lamPhi_at_mZPrime,
        ]
        self.params_at_MZ = solve_ivp(
            self.betaFunc_Init,
            y0=X0,
            t_span=[self.mZPrime, self.M_Z],
            t_eval=np.linspace(self.mZPrime, self.M_Z, 10000),
            atol=1e-10,
            rtol=1e-10,
        ).y[:, -1]
        self.g1_at_mZ, self.g_BL_at_mZ, self.gTilde_at_mZ, self.lamPhi_at_mZ = (
            self.params_at_MZ
        )

        """
        This function determines the portal coupling at mu = M_Z such that the electroweak vacuum is generated.
        For this, we first compute the value of the VEV at mu = M_Z (see 2210.07075 for further details)
        """
        # self.vPhi_at_mZ = np.sqrt(self.M_Z**2/(4*self.g_BL_at_mZ**2) * np.exp(self.g_BL_at_mZ**4 / 3 - self.lamPhi_at_mZ * np.pi**2/(3*self.g_BL_at_mZ**4))) #BUG
        self.vPhi_at_mZ = np.sqrt(
            self.M_Z**2
            / (4 * self.g_BL_at_mZ**2)
            * np.exp(1 / 3 - self.lamPhi_at_mZ * np.pi**2 / (3 * self.g_BL_at_mZ**4))
        )

        self.lamMix_at_mZ = 2 * self.lamh_at_mZ * self.vh_at_mZ**2 / self.vPhi_at_mZ**2

    def evolve_to_mZ(self, gTilde):
        g1 = self.g1_at_mZPrime
        gBL = self.gBL_at_mZPrime
        lamphi = self.lamPhi_at_mZPrime

        params_at_MZ = solve_ivp(
            self.betaFunc_Init,
            y0=[g1, gBL, gTilde, lamphi],
            t_span=[self.mZPrime, self.M_Z],
            t_eval=np.linspace(self.mZPrime, self.M_Z, 10000),
            atol=1e-10,
            rtol=1e-10,
        ).y[:, -1]
        return params_at_MZ[2]

    def solve_at_mZ(self, value):
        g_upper_at_mZPrime = 2
        g_lower_at_mZPrime = -2
        g_mid_at_mZPrime = 1
        count = 0
        while count < 100:
            g_mid_at_mZ = self.evolve_to_mZ(g_mid_at_mZPrime)
            if g_mid_at_mZ > value:
                g_upper_at_mZPrime = g_mid_at_mZPrime

            elif g_mid_at_mZ < value:
                g_lower_at_mZPrime = g_mid_at_mZPrime
            g_mid_at_mZPrime = (g_upper_at_mZPrime + g_lower_at_mZPrime) / 2
            count += 1
        return g_mid_at_mZPrime

    def betaFunc_g1(self, mu, g1):
        Y = g1**3 / (16 * np.pi**2 * mu) * 41 / 10  # 6
        return Y

    def betaFunc_Init(self, mu, X):
        """
        these beta functions are used to initially run the B-L parameters down to mu=m_t.
        Here we ignore the term ~l_mix in the beta function for l_phi, since it is numerically small and otherwise not possible
        """
        Y = np.zeros(len(X))

        g1 = X[0]
        gbl = X[1]
        gtilde = X[2]
        l_phi = X[3]

        Y[0] = g1**3 / (16 * np.pi**2 * mu) * 41 / 10
        # Y[0] = g1 ** 3 / (16 * np.pi ** 2 * mu) * 41 / 6

        Y[1] = (
            (12 * gbl**3 + 32 / 3 * gbl**2 * gtilde + 41 / 6 * gbl * gtilde**2)
            * 1
            / (16 * np.pi**2 * mu)
        )

        Y[2] = (
            (
                gtilde * 41 / 6 * (gtilde**2 + 6 / 5 * g1**2)
                + 32 / 3 * gbl * (gtilde**2 + 3 / 5 * g1**2)
                + 12 * gbl**2 * gtilde
            )
            * 1
            / (16 * np.pi**2 * mu)
        )
        # Y[2] = (gtilde * 41 / 6 * (gtilde ** 2 + 2 * g1 ** 2) + 32 / 3 * gbl * (gtilde ** 2 + g1 ** 2) + 12 * gbl ** 2 * gtilde) * 1 / (16 * np.pi ** 2 * mu)

        Y[3] = (
            (20 * l_phi**2 - 48 * l_phi * gbl**2 + 96 * gbl**4)
            * 1
            / (16 * np.pi**2 * mu)
        )
        # Y[3] = (20 * l_phi ** 2 - 48 * l_phi * gbl ** 2 + 96 * gbl ** 4) * 1 / (16 * np.pi ** 2 * mu)

        return Y

    def BetaFunctionsBL(self, mu, X):
        """
        full beta functions from arXiV:1403.4953
        for mu < Lambda_QCD, running has to be switched off since strong gauge coupling and top Yukawa diverge
        """
        Y = np.zeros(len(X))

        g3 = X[0]
        g2 = X[1]
        g1 = X[2]
        gbl = X[3]
        gtilde = X[4]
        yt = X[5]
        l_phi = X[6]
        l_mix = X[7]
        l_h = X[8]

        Y[0] = -7 * g3**3 * 1 / (16 * np.pi**2 * mu)
        Y[1] = (-19 / 6) * g2**3 * 1 / (16 * np.pi**2 * mu)

        Y[2] = g1**3 / (16 * np.pi**2 * mu) * 41 / 10
        # Y[2] = g1 ** 3 / (16 * np.pi ** 2 * mu) * 41 / 6

        Y[3] = (
            (12 * gbl**3 + 32 / 3 * gbl**2 * gtilde + 41 / 6 * gbl * gtilde**2)
            * 1
            / (16 * np.pi**2 * mu)
        )

        Y[4] = (
            (
                gtilde * 41 / 6 * (gtilde**2 + 6 / 5 * g1**2)
                + 32 / 3 * gbl * (gtilde**2 + 3 / 5 * g1**2)
                + 12 * gbl**2 * gtilde
            )
            * 1
            / (16 * np.pi**2 * mu)
        )
        # Y[4] = (gtilde * 41 / 6 * (gtilde ** 2 + 2 * g1 ** 2) + 32 / 3 * gbl * (gtilde ** 2 + g1 ** 2) + 12 * gbl ** 2 * gtilde) * 1 / (16 * np.pi ** 2 * mu)

        Y[5] = (
            (
                yt
                * (
                    -5 / 3 * gtilde * gbl
                    - 17 / 12 * gtilde**2
                    - 17 / 20 * g1**2
                    - 9 / 4 * g2**2
                    - 8 * g3**2
                    - 2 / 3 * gbl**2
                )
                + 9 / 2 * yt**3
            )
            * 1
            / (16 * np.pi**2 * mu)
        )
        # Y[5] = (yt * (-5 / 3 * gtilde * gbl - 17 / 12 * gtilde ** 2 - 17 / 12 * g1 ** 2 - 9 / 4 * g2 ** 2 - 8 * g3 ** 2 - 2 / 3 * gbl ** 2) + 9 / 2 * yt ** 3) * 1 / (16 * np.pi ** 2 * mu)

        Y[6] = (
            (20 * l_phi**2 + 2 * l_mix**2 - 48 * l_phi * gbl**2 + 96 * gbl**4)
            * 1
            / (16 * np.pi**2 * mu)
        )

        Y[7] = (
            (
                l_mix
                * (
                    12 * l_h
                    + 8 * l_phi
                    - 4 * l_mix
                    + 6 * yt**2
                    - 9 / 2 * g2**2
                    - 9 / 10 * g1**2
                    - 3 / 2 * gtilde**2
                    - 24 * gbl**2
                )
                - 12 * gtilde**2 * gbl**2
            )
            * 1
            / (16 * np.pi**2 * mu)
        )
        # Y[7] = (l_mix * (12 * l_h + 8 * l_phi - 4 * l_mix + 6 * yt ** 2 - 9 / 2 * g2 ** 2 - 9 / 10 * g1 ** 2 - 3 / 2 * gtilde ** 2 - 24 * gbl ** 2)) * 1 / (16 * np.pi ** 2 * mu)
        # Y[7] = (l_mix * (12 * l_h + 8 * l_phi + 4 * l_mix + 6 * yt ** 2 - 9 / 2 * g2 ** 2 - 3 / 2 * g1 ** 2 - 3 / 2 * gtilde ** 2 - 24 * gbl ** 2) + 12 * gtilde ** 2 * gbl ** 2) * 1 / (16 * np.pi ** 2 * mu)

        Y[8] = (
            (
                -6 * yt**4
                + 24 * l_h**2
                + l_mix**2
                + l_h * (12 * yt**2 - 9 / 5 * g1**2 - 9 * g2**2 - 3 * gtilde**2)
                + 27 / 200 * g1**4
                + 9 / 20 * g2**2 * g1**2
                + 9 / 8 * g2**4
                + 3 / 4 * g2**2 * gtilde**2
                + 9 / 20 * g1**2 * gtilde**2
                + 3 / 8 * gtilde**4
            )
            * 1
            / (16 * np.pi**2 * mu)
        )
        # Y[8] = (-6 * yt ** 4 + 24 * l_h ** 2 + l_mix ** 2 + l_h * (12 * yt ** 2 - 3 * g1 ** 2 - 9 * g2 ** 2 - 3 * gtilde ** 2) + 3 / 8 * g1 ** 4 + 3 / 4 * g2 ** 2 * g1 ** 2 + 9 / 8 * g2 ** 4 + 3 / 4 * g2 ** 2 * gtilde ** 2 + 3 / 4 * g1 ** 2 * gtilde ** 2 + 3 / 8 * gtilde ** 4) * 1 / (16 * np.pi ** 2 * mu)

        return Y

    def runParams(self, mu):
        """
        runs parameters to RG scale in high-T, mu = pi T
        """
        mu = mu if mu >= 0.1 else 0.1
        X0 = (
            self.g3_at_mZ,
            self.g2_at_mZ,
            self.g1_at_mZ,
            self.g_BL_at_mZ,
            self.gTilde_at_mZ,
            self.yt_at_mZ,
            self.lamPhi_at_mZ,
            self.lamMix_at_mZ,
            self.lamh_at_mZ,
        )

        t = np.logspace(np.log10(self.M_Z), np.log10(mu), 1000)
        sol = solve_ivp(
            self.BetaFunctionsBL,
            y0=X0,
            t_span=[t[0], t[-1]],
            t_eval=t,
            rtol=1e-10,
            atol=1e-10,
        )
        return sol.y[:, -1]

    # def interpolations(self):
    #     """
    #     This method creates interpolation functions, such that the beta functions do not have to be evaluated every time
    #     """
    #     T_range = np.logspace(-1, 8, 1000)
    #     sols = []
    #     for temp in T_range:
    #         sols.append(self.runParams(temp))
    #     sols = np.asarray(sols).T
    #
    #     self.g3 = interpolate.interp1d(T_range, sols[0], fill_value="extrapolate")
    #     self.g2 = interpolate.interp1d(T_range, sols[1], fill_value="extrapolate")
    #     self.g1 = interpolate.interp1d(T_range, sols[2], fill_value="extrapolate")
    #     self.gbl = interpolate.interp1d(T_range, sols[3], fill_value="extrapolate")
    #     self.gtilde = interpolate.interp1d(T_range, sols[4], fill_value="extrapolate")
    #     self.yt = interpolate.interp1d(T_range, sols[5], fill_value="extrapolate")
    #     self.l_phi = interpolate.interp1d(T_range, sols[6], fill_value="extrapolate")
    #     self.l_mix = interpolate.interp1d(T_range, sols[7], fill_value="extrapolate")
    #     self.l_h = interpolate.interp1d(T_range, sols[8], fill_value="extrapolate")


def params_from_mH(target, tol=0.01):
    mZps = np.logspace(5, 7, 20)
    gBLs = np.logspace(-3, -1, 20)
    params = []
    for mZp in mZps:
        for gBL in gBLs:
            mH = m_over_H(mZp, gBL)
            if np.abs(mH - target) / target < tol:
                params.append((mZp, gBL))
    return params


def m_over_H(mZp, gBL):
    v_hQCD = 0.1  # GeV
    input_obj = InputParameters(gBL, mZp)
    T = 0.1
    out = input_obj.runParams(T)
    lam_p = out[7]
    return 8 * np.pi * MPL * np.sqrt(lam_p) * v_hQCD / mZp**2


if __name__ == "__main__":
    params = params_from_mH(10**4)
    print(params)
