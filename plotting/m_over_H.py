import numpy as np
from Veff_Daniel import EffectivePotential

M_PL = 2.435 * 10 ** (18)


def mH_from_gBL_mZ(gBL, mZp, vh_qcd=0.1):
    model = EffectivePotential(gBL, mZp, vh_qcd)
    model.interpolations()
    return model.m_over_H()


def mu_over_sqrtlam_from_mH(mH):
    """Takes a ratio m/H at the origin and returns the ratio mu/sqrt(lambda)"""
    return np.sqrt(12) * M_PL / mH


def lam_from_mu(mu, mH):
    M_PL = 2.435 * 10 ** (18)
    mu_sqrtlam = np.sqrt(12) * M_PL / mH
    lam = (mu / mu_sqrtlam) ** 2
    return lam


def main():
    # mHs = np.array([10 ** (i) for i in range(1, 10)])
    mHs = np.array([1e5])
    vevs = mu_over_sqrtlam_from_mH(mHs)
    mus = np.array([M_PL * 10 ** (-i) for i in range(5, 15, 1)])

    for mH, vev in zip(mHs, vevs):
        lams = np.array([lam_from_mu(mu, mH) for mu in mus])
        print(f"{mH=:.1E}, {vev=:.3E}")
        for mu, lam in zip(mus, lams):
            print(f"{mu/M_PL=:.1E},  {mu=:.3E},  {lam=:.3E}")
        print("----------------")

    # vh_qcd = 0.1
    # gBL = 2e-3
    # mZp = 2e5
    # mH = mH_from_gBL_mZ(gBL, mZp, vh_qcd)
    # print(f"{mH=:.1E}")
    #
    # gBLs = np.logspace(-4, -1, 2)
    # mZps = np.logspace(4, 7, 2)
    # ratio_grid = np.zeros((len(gBLs), len(mZps)))
    #
    # for i, gBL in enumerate(gBLs):
    #     for j, mZp in enumerate(mZps):
    #         print(f"Computing indices: {i=} {j=}")
    #         ratio_grid[i, j] = mH_from_gBL_mZ(gBL, mZp, vh_qcd)
    # print(ratio_grid)


if __name__ == "__main__":
    main()
