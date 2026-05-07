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

    mHs = np.array([10**i for i in range(3, 5)])
    rs = mu_over_sqrtlam_from_mH(mHs)

    mus = np.array([M_PL * 10 ** (-i) for i in range(1, 9, 1)])

    for i, r in enumerate(rs):
        lams = (mus / r) ** 2
        print(f"{mHs[i]=:.1E}")
        for mu, lam in zip(mus, lams):
            print(f"{mu=:.2E}   {lam=:.2E}")
        print("-----------")

    print(lam_from_mu(mus[0], mHs[0]))


if __name__ == "__main__":
    main()
