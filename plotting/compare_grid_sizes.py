import numpy as np
import matplotlib.pyplot as plt
from m_over_H import lam_from_mu
from pathlib import Path
from Veff_Daniel import EffectivePotential

M_PL = 2.435 * 10 ** (18)


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
        for dir in [
            # "mH1e4_VV10_64",
            # "mH1e4_LF_128",
            # "mH1e4_VV10_256",
            # "mH1e4_256_kIR1e-2",
            "mH1e4_LF_512",
            "mH1e4_512_kIR1e-2",
            # "mH1e4_LF_1024",
        ]
    ]
    labels = [
        # r"$N=64$",
        # r"$N=128$",
        # r"$N=256$, $k_{\mathrm{IR}}=0.1$",
        # r"$N=256$, $k_{\mathrm{IR}}=0.01$",
        r"$N=512$, $k_{\mathrm{IR}}=0.1$",
        r"$N=512$, $k_{\mathrm{IR}}=0.01$",
        # r"$N=1024$",
    ]
    filenames = [in_dir / "spectra_gws.txt" for in_dir in input_dirs]
    plt.figure(figsize=(8, 6))
    for i, file in enumerate(filenames):
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
        kH = data[:, 0] / H * omega_star
        omega_gw = data[:, 1]
        plt.plot(kH, omega_gw, label=labels[i])
    plt.xlabel(r"$k/H$")
    plt.ylabel(r"$\Omega_\mathrm{GW}(k,t)$")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(alpha=0.5)
    plt.legend()
    savefile = "./figures/grid_sizes_gw.pdf"
    print(f"Saving figure to {savefile}")
    plt.savefig(savefile, format="pdf", backend="pgf")


if __name__ == "__main__":
    main()
    print("All done!")
