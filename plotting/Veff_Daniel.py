import numpy as np
import pickle
from inputParameters import *
from scipy import interpolate
from decimal import *
from scipy.integrate import quad
import cmath
import scipy
from scipy.optimize import fmin, minimize

# from cosmoTransitions.finiteT import Jb, Jb_spline, Jb_exact

getcontext().prec = 50


class EffectivePotential:
    def __init__(self, gBL, mZPrime, vh_qcd):
        self.gBL_at_mZPrime = gBL
        self.mZPrime = mZPrime
        self.v_phi = mZPrime / (2 * gBL)

        self.inputClass = InputParameters(self.gBL_at_mZPrime, self.mZPrime)
        # self.testInput = self.inputClass.runParams(.085)

        self.vh_qcd = vh_qcd
        self.M_Z = 91.1876

        # self.T_qcd = 0.085
        self.T_qcd = 0.085
        self.MPl = 2.44e18

        self.phi_range, self.JB_interp = pickle.load(open("Jb_interpolation.p", "rb"))
        self.phi_range, self.dJB_interp = pickle.load(open("dJb_interpolation.p", "rb"))

    def vacuumEnergy(self, T):
        rho_vac = self.Veff(1e-10, self.vh_qcd, T) - self.Veff(
            self.v_phi, self.vh_qcd, T
        )
        return rho_vac

    def rho_vac_T(self, T):
        phimin = self.v_phi  # self.minimize_Veff(T)
        # print(phimin)
        return self.Veff(1e-10, 0, T) - self.Veff(phimin, 0, T)

    def rho_rad(self, T):
        g_star = 106.75 + 3 + 1
        return np.pi**2 / 30 * g_star * T**4

    # def minimize_Veff(self, T):
    #     return fmin(lambda phi: self.Veff(phi, 1e-50, T), x0=self.v_phi, disp=0)[0]
    def minimize_Veff(self, T):

        res = minimize(
            lambda x: self.Veff(x[0], 1e-50, T),
            x0=[self.v_phi],
            method="Nelder-Mead",
        )

        return res.x[0]

    def H_init(self, T):
        rho_vac = self.vacuumEnergy(T)
        return np.sqrt(rho_vac / (3 * self.MPl**2))

    def Hubble(self, T):
        rho_vac = self.vacuumEnergy(T)
        rho_rad = self.rho_rad(T)
        return np.sqrt((rho_vac + rho_rad) / (3 * self.MPl**2))

    def interpolations(self):
        """This method will create interpolation functions such that the beta functions don't have to be evaluated every time."""
        T_range = np.logspace(-1, 8, 1000)
        sols = []
        for temp in T_range:
            sols.append(self.inputClass.runParams(temp))
        sols = np.asarray(sols).T

        self.fun0 = interpolate.interp1d(T_range, sols[0], fill_value="extrapolate")
        self.fun1 = interpolate.interp1d(T_range, sols[1], fill_value="extrapolate")
        self.fun2 = interpolate.interp1d(T_range, sols[2], fill_value="extrapolate")
        self.fun3 = interpolate.interp1d(T_range, sols[3], fill_value="extrapolate")
        self.fun4 = interpolate.interp1d(T_range, sols[4], fill_value="extrapolate")
        self.fun5 = interpolate.interp1d(T_range, sols[5], fill_value="extrapolate")
        self.fun6 = interpolate.interp1d(T_range, sols[6], fill_value="extrapolate")
        self.fun7 = interpolate.interp1d(T_range, sols[7], fill_value="extrapolate")
        self.fun8 = interpolate.interp1d(T_range, sols[8], fill_value="extrapolate")

        # phi_range1 = np.logspace(-5,3,1000)
        # phi_range2 = -np.logspace(2,-5,1000)
        # phi_range=np.concatenate((phi_range1,phi_range2))
        # Jbs = [self.solve_J_boson(i) for i in phi_range]
        # dJbs = [self.solve_dJ_boson_interpolate(i) for i in phi_range]

        self.interpolate_Jb = interpolate.interp1d(
            self.phi_range, self.JB_interp, fill_value="extrapolate"
        )
        self.interpolate_dJb = interpolate.interp1d(
            self.phi_range, self.dJB_interp, fill_value="extrapolate"
        )

        # self.interpolate_Jb = interpolate.CubicSpline(phi_range, Jbs, extrapolate=True)
        # self.interpolate_dJb = interpolate.CubicSpline(phi_range, dJbs, extrapolate=True)

    def J_boson(self, y, x):
        """
        Temperature-dependent integrals in effective potential
        """
        return np.real(y**2 * np.log(1 - np.exp(-cmath.sqrt(x + y**2))))

    def solve_J_boson(self, x):
        """
        Integrates the temperature integrals
        """
        # X0 = quad(self.J_boson, 0, np.inf, args=(0))[0]

        if isinstance(x, np.ndarray):
            return np.array(
                [quad(self.J_boson, 0, np.inf, args=(x[i]))[0] for i in range(len(x))]
            )
        else:
            return quad(self.J_boson, 0, np.inf, args=(x), epsrel=1e-10)[0]

    def dJ_boson(self, y, x, dxdphi):
        # derivative of the integrand of J_boson, used for analytical derivative, input is m**2/T**2
        # return np.real(y ** 2 * (np.exp(-cmath.sqrt(x + y ** 2)) / (1 - (np.exp(-cmath.sqrt(x + y ** 2))))) * 0.5 * cmath.sqrt(x + y ** 2) ** (-1) * dxdphi)
        return np.real(
            y**2
            * (np.exp(-cmath.sqrt(x + y**2)) / (1 - (np.exp(-cmath.sqrt(x + y**2)))))
            * 0.5
            * cmath.sqrt(x + y**2) ** (-1)
            * dxdphi
        )

    def solve_dJ_boson(self, x, dxdphi):
        # integrates dJ_boson
        return quad(self.dJ_boson, 1e-100, np.inf, args=(x, dxdphi), epsabs=1e-10)[0]
        # return Jb(x, approx='spline', deriv=1, n=8)

    def dJ_boson_interpolate(self, y, x):
        return np.real(
            y**2
            * (np.exp(-cmath.sqrt(x + y**2)) / (1 - (np.exp(-cmath.sqrt(x + y**2)))))
            * 0.5
            * cmath.sqrt(x + y**2) ** (-1)
        )  # * dxdphi)

    def solve_dJ_boson_interpolate(self, x):
        # integrates dJ_boson
        return quad(self.dJ_boson_interpolate, 1e-100, np.inf, args=(x), epsabs=1e-10)[
            0
        ]

    def dJ2_boson(self, y, x, dxdphi, dx2dphi2):
        sqrt_func = np.sqrt(x + y**2)
        exp_func = np.exp(-sqrt_func)
        first_term = exp_func * dx2dphi2 / (2 * (1 - exp_func) * sqrt_func)
        second_term = exp_func * dxdphi**2 / (4 * (1 - exp_func) * sqrt_func**2)
        third_term = (
            np.exp(-2 * sqrt_func)
            * dxdphi**2
            / (4 * (1 - exp_func) ** 2 * sqrt_func**2)
        )
        fourth_term = exp_func * dxdphi**2 / (4 * (1 - exp_func) * sqrt_func**3)
        return np.real(y**2 * (first_term - second_term - third_term - fourth_term))

    def solve_dJ2_boson(self, x, dxdphi, dx2dphi2):
        # integrates dJ2_boson
        return quad(
            self.dJ2_boson, 1e-100, np.inf, args=(x, dxdphi, dx2dphi2), epsabs=1e-10
        )[0]

    def V_T(self, mSq, T):
        n_i = 3
        # return T**4/(2*np.pi**2) * n_i * self.solve_J_boson(mSq/T**2)
        try:
            res = T**4 / (2 * np.pi**2) * n_i * self.interpolate_Jb(mSq / T**2)
        except:
            res = T**4 / (2 * np.pi**2) * n_i * self.solve_J_boson(mSq / T**2)
        # return T**4/(2*np.pi**2) * n_i * self.interpolate_Jb(mSq/T**2)
        return res

    def Veff_decimal(self, phi, h, T):
        mZPr = 2 * self.gBL_at_mZPrime * phi
        mu = max([mZPr, np.pi * T, 0.1])

        try:
            lamPhi = self.fun6(mu)
            gBL = self.fun3(mu)
            lamMix = self.fun7(mu)
        except:
            sol = self.inputClass.runParams(mu)
            lamPhi = sol[6]
            gBL = sol[3]
            lamMix = sol[7]

        # V_tree = Decimal(1/4 * lamPhi * phi**4)
        # V_cw = Decimal(3/(64*np.pi**2) * (2*gBL*phi)**4 * (np.log((2*gBL*phi)**2/mu**2) - 5/6))
        # V_portal = Decimal(-lamMix/4 * h**2 * phi**2)

        V_tree = 1 / 4 * lamPhi * phi**4
        V_cw = (
            3
            / (64 * np.pi**2)
            * (2 * gBL * phi) ** 4
            * (np.log((2 * gBL * phi) ** 2 / mu**2) - 5 / 6)
        )
        V_portal = -lamMix / 4 * h**2 * phi**2

        mZSq = 4 * gBL**2 * phi**2
        Pi_Z = 4 * gBL**2 * T**2
        if T != 0:
            # V_T = Decimal(self.V_T(mZSq, T))
            # V_daisy = Decimal(-T/(12*np.pi) * 3 * ((mZSq + Pi_Z)**(3/2) - mZSq**(3/2)))

            V_T = self.V_T(mZSq, T)
            # V_daisy = -T / (12 * np.pi) * 3 * ((mZSq + Pi_Z) ** (3 / 2) - mZSq ** (3 / 2))
            V_daisy = -T / (12 * np.pi) * ((mZSq + Pi_Z) ** (3 / 2) - mZSq ** (3 / 2))

            return V_tree + V_cw + V_portal + V_T + V_daisy
        else:
            return V_tree + V_cw + V_portal

    def Veff(self, phi, h, T):
        return float(self.Veff_decimal(phi, h, T))

    def dVeff(self, phi0, h, T):
        return float(
            scipy.misc.derivative(
                lambda phi, h, T: self.Veff(phi, h, T),
                x0=phi0,
                dx=1e-8,
                n=1,
                args=(h, T),
                order=5,
            )
        )

    def dVeff_analytic(self, phi, h, T):
        mZPr = 2 * self.gBL_at_mZPrime * phi
        mu = max([mZPr, np.pi * T, 0.1])
        lamPhi = self.fun6(mu)
        gBL = self.fun3(mu)
        lamMix = self.fun7(mu)

        # dV_tree = Decimal(lamPhi * phi ** 3)
        # dV_cw = Decimal(phi**3 /np.pi**2 * (-gBL**4 + 3 * gBL**4 * np.log(4*gBL**2*phi**2/mu**2)))
        # dV_portal = Decimal(-lamMix/2 * h**2 * phi)

        dV_tree = lamPhi * phi**3
        dV_cw = (
            phi**3
            / np.pi**2
            * (-(gBL**4) + 3 * gBL**4 * np.log(4 * gBL**2 * phi**2 / mu**2))
        )
        dV_portal = -lamMix / 2 * h**2 * phi

        if T != 0:
            mZSq = 4 * gBL**2 * phi**2
            PiZ = 4 * gBL**2 * T**2
            dmZSq_dphi = 8 * gBL * phi * gBL
            x_mZ = mZSq / T**2
            dx_mZ = dmZSq_dphi / T**2
            # dJ_z = 3*self.solve_dJ_boson(x_mZ, dx_mZ)
            dJ_z = 3 * self.interpolate_dJb(x_mZ) * dx_mZ
            # dV_T = Decimal(T**4/(2*np.pi**2) * dJ_z)
            dV_T = T**4 / (2 * np.pi**2) * dJ_z

            """
            Daisy
            """
            # dV_daisy = Decimal(-3*T/(12*np.pi) * 3/2 * ((mZSq + PiZ)**0.5 * dmZSq_dphi - mZSq**0.5 * dmZSq_dphi))
            # dV_daisy = -3*T/(12*np.pi) * 3/2 * ((mZSq + PiZ)**0.5 * dmZSq_dphi - mZSq**0.5 * dmZSq_dphi)
            dV_daisy = (
                -T
                / (12 * np.pi)
                * 3
                / 2
                * ((mZSq + PiZ) ** 0.5 * dmZSq_dphi - mZSq**0.5 * dmZSq_dphi)
            )

        else:
            dV_T = 0
            dV_daisy = 0

        return float(dV_tree + dV_portal + dV_cw + dV_T + dV_daisy)

    def d2Veff(self, phi, h, T):
        mZPr = 2 * self.gBL_at_mZPrime * phi
        mu = max([mZPr, np.pi * T, 0.1])
        try:
            lamPhi = self.fun6(mu)
            gBL = self.fun3(mu)
            lamMix = self.fun7(mu)
            # lamMix = self.inputClass.lamMix_at_mZ

        except:
            sol = self.inputClass.runParams(mu)
            lamPhi = sol[6]
            gBL = sol[3]
            lamMix = sol[7]
            # lamMix = self.inputClass.lamMix_at_mZ

        d2V_tree = 3 * lamPhi * phi**2
        if abs(phi) < 1e-12:
            d2V_cw = 0.0
        else:
            d2V_cw = (
                3
                * gBL**4
                * phi**2
                / np.pi**2
                * (3 * np.log(4 * gBL**2 * phi**2 / mu**2) + 1)
            )
        d2V_portal = -lamMix / 2 * h**2

        if T != 0:
            mZSq = 4 * gBL**2 * phi**2
            PiZ = +4 * gBL**2 * T**2
            dmZSq_dphi = 8 * gBL * phi * gBL
            dmZSq2_dphi2 = 8 * gBL**2
            x_mZ = mZSq / T**2
            dx_mZ = dmZSq_dphi / T**2
            dx2_mZ = dmZSq2_dphi2 / T**2
            dJ2_z = 3 * self.solve_dJ2_boson(x_mZ, dx_mZ, dx2_mZ)
            d2V_T = T**4 / (2 * np.pi**2) * dJ2_z

            """
            Daisy
            """
            # first_term = dmZSq2_dphi2 * ((dmZSq_dphi + PiZ)**0.5 - dmZSq2_dphi2**0.5)
            # second_term = dmZSq_dphi * 0.5 * dmZSq2_dphi2 * (np.sqrt(dmZSq_dphi + PiZ)**(-1) - np.sqrt(dmZSq_dphi)**(-1))

            first_term = dmZSq2_dphi2 * ((mZSq + PiZ) ** 0.5 - mZSq**0.5)
            second_term = (
                dmZSq_dphi**2
                * 0.5
                * (np.sqrt(mZSq + PiZ) ** (-1) - np.sqrt(mZSq) ** (-1))
            )
            # d2V_daisy = (-9*T/(24*np.pi)) * (first_term + second_term)
            d2V_daisy = (-3 * T / (24 * np.pi)) * (first_term + second_term)

        else:
            d2V_T = 0
            d2V_daisy = 0

        return float(d2V_tree + d2V_portal + d2V_cw + d2V_T + d2V_daisy)

    def Veff_CosmoTrans(self, Fields, T):
        h = self.vh_qcd
        phi = Fields  # [..., 0]

        mZPr = 2 * self.gBL_at_mZPrime * phi

        if type(phi) != list and type(phi) != np.ndarray:
            mu = max([mZPr, np.pi * T, 0.1])
        else:
            maxmu = np.max([np.pi * T, 0.1])
            mu = np.fmax(mZPr, maxmu)

        lamPhi = self.fun6(mu)
        gBL = self.fun3(mu)
        lamMix = self.fun7(mu)

        V_tree = 1 / 4 * lamPhi * phi**4
        V_cw = (
            3
            / (64 * np.pi**2)
            * (2 * gBL * phi) ** 4
            * (np.log((2 * gBL * phi) ** 2 / mu**2) - 5 / 6)
        )
        V_portal = -lamMix / 4 * h**2 * phi**2 if T <= self.T_qcd else 0

        mZSq = 4 * gBL**2 * phi**2
        Pi_Z = 4 * gBL**2 * T**2
        if T != 0:
            V_T = self.V_T(mZSq, T)
            # V_daisy = -T / (12 * np.pi) * 3 * ((mZSq + Pi_Z) ** (3 / 2) - mZSq ** (3 / 2))
            V_daisy = -T / (12 * np.pi) * ((mZSq + Pi_Z) ** (3 / 2) - mZSq ** (3 / 2))

            if T <= self.T_qcd:
                return V_tree + V_cw + V_portal + V_T + V_daisy
            else:
                return V_tree + V_cw + V_T + V_daisy
        else:
            return V_tree + V_cw + V_portal

    def dVeff_CosmoTrans(self, Fields, T):
        h = self.vh_qcd
        phi = Fields  # [..., 0]

        mZPr = 2 * self.gBL_at_mZPrime * phi
        if type(phi) != list and type(phi) != np.ndarray:
            mu = max([mZPr, np.pi * T, 0.1])
        else:
            maxmu = np.max([np.pi * T, 0.1])
            mu = np.fmax(mZPr, maxmu)

        lamPhi = self.fun6(mu)
        gBL = self.fun3(mu)
        lamMix = self.fun7(mu)

        dV_tree = Decimal(lamPhi * phi**3)
        dV_cw = Decimal(
            phi**3
            / np.pi**2
            * (-(gBL**4) + 3 * gBL**4 * np.log(4 * gBL**2 * phi**2 / mu**2 + 1e-50))
        )
        dV_portal = Decimal(-lamMix / 2 * h**2 * phi)

        if T != 0:
            mZSq = 4 * gBL**2 * phi**2
            PiZ = 4 * gBL**2 * T**2
            dmZSq_dphi = 8 * gBL * phi * gBL
            x_mZ = mZSq / T**2
            dx_mZ = dmZSq_dphi / T**2
            # dJ_z = 3 * self.solve_dJ_boson(x_mZ, dx_mZ)
            try:
                dJ_z = 3 * self.interpolate_dJb(x_mZ) * dx_mZ
            except:
                dJ_z = 3 * self.solve_dJ_boson(x_mZ, dx_mZ)

            dV_T = Decimal(T**4 / (2 * np.pi**2) * dJ_z)
            # dV_T = T**4/(2*np.pi**2) * dJ_z

            """
            Daisy
            """
            # dV_daisy = Decimal(-3*T/(12*np.pi) * 3/2 * ((mZSq + PiZ)**0.5 * dmZSq_dphi - mZSq**0.5 * dmZSq_dphi))
            # dV_daisy = -3*T/(12*np.pi) * 3/2 * ((mZSq + PiZ)**0.5 * dmZSq_dphi - mZSq**0.5 * dmZSq_dphi)
            dV_daisy = Decimal(
                -T
                / (12 * np.pi)
                * 3
                / 2
                * ((mZSq + PiZ) ** 0.5 * dmZSq_dphi - mZSq**0.5 * dmZSq_dphi)
            )

        else:
            dV_T = 0
            dV_daisy = 0

        if T <= self.T_qcd:
            return float(dV_tree + dV_portal + dV_cw + dV_T + dV_daisy)

        else:
            return float(dV_tree + dV_cw + dV_T + dV_daisy)

    def d2Veff_CosmoTrans(self, Fields, T):
        h = self.vh_qcd
        phi = Fields  # Fields[..., 0]

        mZPr = 2 * self.gBL_at_mZPrime * phi
        mu = max([mZPr, np.pi * T, 0.1])
        lamPhi = self.fun6(mu)
        gBL = self.fun3(mu)
        lamMix = self.fun7(mu)

        d2V_tree = 3 * lamPhi * phi**2
        d2V_cw = (
            3
            * gBL**4
            * phi**2
            / np.pi**2
            * (3 * np.log(4 * gBL**2 * phi**2 / mu**2) + 1)
        )

        d2V_portal = -lamMix / 2 * h**2  # if T <= 0.1 else 0

        if T != 0:
            mZSq = 4 * gBL**2 * phi**2
            PiZ = +4 * gBL**2 * T**2
            dmZSq_dphi = 8 * gBL * phi * gBL
            dmZSq2_dphi2 = 8 * gBL**2
            x_mZ = mZSq / T**2
            dx_mZ = dmZSq_dphi / T**2
            dx2_mZ = dmZSq2_dphi2 / T**2
            dJ2_z = 3 * self.solve_dJ2_boson(x_mZ, dx_mZ, dx2_mZ)
            d2V_T = T**4 / (2 * np.pi**2) * dJ2_z

            """
            Daisy
            """
            first_term = dmZSq2_dphi2 * ((mZSq + PiZ) ** 0.5 - mZSq**0.5)
            second_term = (
                dmZSq_dphi**2
                * 0.5
                * (np.sqrt(mZSq + PiZ) ** (-1) - np.sqrt(mZSq) ** (-1))
            )
            # d2V_daisy = (-9*T/(24*np.pi)) * (first_term + second_term)
            d2V_daisy = (-3 * T / (24 * np.pi)) * (first_term + second_term)

        else:
            d2V_T = 0
            d2V_daisy = 0
        if T <= self.T_qcd:
            return float(d2V_tree + d2V_portal + d2V_cw + d2V_T + d2V_daisy)
        else:
            return float(d2V_tree + d2V_cw + d2V_T + d2V_daisy)

    def find_T_vac(self):
        T_upper = 10 * self.mZPrime
        T_lower = 1e-2 * self.mZPrime
        T_mid = (T_upper + T_lower) / 2
        count = 0
        while count < 30:
            rho_vac_minus_rho_rad = self.rho_vac_T(T_mid) - self.rho_rad(T_mid)
            if rho_vac_minus_rho_rad > 0:
                T_lower = T_mid
            elif rho_vac_minus_rho_rad < 0:
                T_upper = T_mid
            T_mid = (T_upper + T_lower) / 2
            count += 1
        return T_mid

    def find_Tc(self):
        T_upper = 10 * self.mZPrime
        T_lower = 1e-2 * self.mZPrime
        T_mid = (T_upper + T_lower) / 2
        count = 0
        while count < 30:
            print("T_mid", T_mid)
            V_false = self.Veff_CosmoTrans(1e-10, T_mid)
            true_min = minimize(
                lambda phi: self.Veff_CosmoTrans(phi, T_mid), x0=self.v_phi
            ).x
            V_true = self.Veff_CosmoTrans(true_min, T_mid)
            if V_true - V_false < 0:
                T_lower = T_mid
            elif V_true - V_false > 0:
                T_upper = T_mid
            T_mid = (T_upper + T_lower) / 2
            count += 1
        return T_mid

    def find_T_barrier(self):
        T_upper = 1
        T_lower = 1e-10
        T_mid = (T_upper + T_lower) / 2
        phi_0 = 1e-10
        count = 0
        while count < 30:
            curvature_mid = self.d2Veff(phi_0, self.vh_qcd, T_mid)
            if curvature_mid > 0:
                T_upper = T_mid
            elif curvature_mid < 0:
                T_lower = T_mid
            T_mid = (T_upper + T_lower) / 2
            count += 1
        return T_mid

    def mu(self):
        return np.sqrt(abs(self.d2Veff(phi=0, h=self.vh_qcd, T=0)))

    def m_over_H(self):

        # true vacuum
        phi_true = self.minimize_Veff(T=0)

        # vacuum energy
        DeltaV = self.Veff(1e-10, self.vh_qcd, 0) - self.Veff(phi_true, self.vh_qcd, 0)

        H = np.sqrt(DeltaV / (3 * self.MPl**2))

        # curvature at origin
        m = np.sqrt(abs(self.d2Veff(phi=0, h=self.vh_qcd, T=0)))

        return m / H


if __name__ == "__main__":
    vh_qcd = 0.1
    gBL_vals = np.logspace(-4, 0, 10)
    mZ_vals = np.logspace(4, 8, 20)
    ratio_grid = np.zeros((len(gBL_vals), len(mZ_vals)))

    for i, gBL in enumerate(gBL_vals):
        for j, mZ in enumerate(mZ_vals):
            print(f"Computing. Indeces: {i=} {j=}")
            model = EffectivePotential(gBL, mZ, vh_qcd)
            model.interpolations()

            ratio_grid[i, j] = model.m_over_H()
    print(ratio_grid)
