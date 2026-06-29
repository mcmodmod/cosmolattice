import numpy as np
import matplotlib.pyplot as plt
from get_params_from_mH import lam_from_mu
from pathlib import Path

M_PL = 2.435 * 10 ** (18)


def T_rh(gBL, mZp, g_star_eps):
    return (
        1.4 * 10**4 * g_star_eps ** (-0.25) * gBL ** (1 / 2) * (1e6 / mZp) ** (3 / 2)
    )  # GeV


def main():

    plt.rcParams.update(
        {
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "axes.labelsize": 20,
            "legend.fontsize": 13,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.constrained_layout.use": True,
        }
    )
    base_dir = Path("../output/mH1e4")
    input_dirs = [
        base_dir / dir
        for dir in ["mH1e4_VV10_64", "mH1e4_LF_128", "mH1e4_VV10_256", "mH1e4_LF_512"]
    ]
    filenames = [in_dir / "spectra_gws.txt" for in_dir in input_dirs]

    plt.figure(figsize=(8, 6))
    for file in filenames:
        # Load the file, split into blocks separated by blank lines
        with open(file) as f:
            content = f.read().strip().split("\n\n")
        content = content[-1:]
        data = np.loadtxt(content[0].splitlines())

        mH = 10**4
        mu = 2.435e10
        lam = lam_from_mu(mu, mH)
        omega_star = mu
        f_star = omega_star / np.sqrt(lam)
        H = mu**2 / (np.sqrt(12) * M_PL * np.sqrt(lam))

        g_star = 106.75

        kH = data[:, 0] / H * omega_star

        redshift_amplitude_factor = 1.67e-5 * (100 / g_star) ** (1 / 3)
        omega_gw = data[:, 1]  # * redshift_amplitude_factor

        plt.plot(kH, omega_gw)
    plt.xlabel(r"$k/H$")
    plt.ylabel(r"$\Omega_\mathrm{GW}(k,t)$")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(alpha=0.5)
    savefile = "./figures/test.pdf"
    print(f"Saving figure to {savefile}")
    plt.savefig(savefile, format="pdf", backend="pgf")
    print("All done!")


if __name__ == "__main__":
    main()
