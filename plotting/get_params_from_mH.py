import numpy as np

M_PL = 2.435 * 10 ** (18)


def mu_over_sqrtlam_from_mH(mH):
    """Takes a ratio m/H at the origin and returns the ratio mu/sqrt(lambda)"""
    return np.sqrt(12) * M_PL / mH


def lam_from_mu(mu, mH):
    M_PL = 2.435 * 10 ** (18)
    mu_sqrtlam = np.sqrt(12) * M_PL / mH
    lam = (mu / mu_sqrtlam) ** 2
    return lam


def main():

    mHs = np.array([5] + [10**i for i in range(1, 5)])

    mus = np.array([M_PL * 10 ** (-i) for i in range(1, 10, 1)])

    for mH in mHs:
        lams = np.array([lam_from_mu(mu, mH) for mu in mus])
        print(f"{mH=:.1E}")
        for mu, lam in zip(mus, lams):
            print(f"{mu=:.2E}   {lam=:.2E}")
        print("-----------")


if __name__ == "__main__":
    main()
