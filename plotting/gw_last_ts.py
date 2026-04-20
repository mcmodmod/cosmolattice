import numpy as np
import matplotlib.pyplot as plt


def main():
    plt.style.use("classic")
    plt.tight_layout()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.labelsize": 20,
            "legend.fontsize": 12,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )
    # filename = "../output/lphi4_single/spectra_gws.txt"
    # filename = "../output/mexhat/spectra_gws.txt"
    # filename = "./test.txt"
    plt.figure(figsize=(8, 6))
    for i in range(2, 9, 2):
        print(i)
        filename = f"../output/mexhat/mu_MPle-{i}/spectra_gws.txt"
        # Load the file, split into blocks separated by blank lines
        with open(filename) as f:
            content = f.read().strip().split("\n\n")
        content = content[-1:]

        data = np.loadtxt(content[0].splitlines())
        kappa = data[:, 0]
        omega_gw = data[:, 1]
        plt.plot(kappa, omega_gw, label=f"mu=MPl e-{i}")

    # plt.ylim(10e-41, 10e-33)
    # plt.xlim(0.2, 13)
    plt.xlabel(r"$\kappa=k/\omega_\star$")
    plt.ylabel(r"$\Omega_\mathrm{GW}(k,t)$")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.legend(loc="lower left")
    # plt.title("mexhat, N=128, dt=0.05, kIR=0.1")
    # plt.title("mexhat, N=256, dt=0.05, kIR=0.1")
    # plt.savefig(f"./figures/mexhat_spec_gws_N128_lam1e-1-5.pdf")
    savefile = "./figures/mexhat_GWs.pdf"
    print(f"Saving figure to {savefile}")
    plt.savefig(savefile, format="pdf")
    print("All done!")


if __name__ == "__main__":
    main()
