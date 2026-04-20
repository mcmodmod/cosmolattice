def main():
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.labelsize": 20,
            "legend.fontsize": 13,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )

    directory = "../output/Veff_N32_eta5e6_init500_dt02/"
    # directory = "../output/Veff_N32_eta5e6_init1e3_dt05/"
    print("Using data from ", directory)
    # savefile = "./figures/Veff_N64_eta3e6_init500_LF.pdf"
    savefile = "./figures/test.pdf"
    phiDat = np.loadtxt(directory + "average_scalar_0.txt")

    eta = phiDat[:, 0]
    phi = phiDat[:, 1]

    # print(len(eta))
    # lim1 = 32000
    # lim2 = 34000
    # eta = eta[lim1:lim2]
    # phi = phi[lim1:lim2]

    # ---------------------------
    # Plotting average field values
    # ---------------------------

    plt.figure(figsize=(8, 6))
    plt.plot(eta, phi)
    plt.xlabel(r"$\tilde\eta$")
    plt.ylabel(r"$\langle\tilde\phi\rangle$")
    # plt.title("N=64, tMax=3e6, dt=0.05, phi_init=500GeV, mZp=1e5, g=0.01, LF")
    plt.grid(True)
    plt.tight_layout()
    print("Saving figure to ", savefile)
    plt.savefig(savefile, format="pdf")


if __name__ == "__main__":
    main()
    print("All done!")
