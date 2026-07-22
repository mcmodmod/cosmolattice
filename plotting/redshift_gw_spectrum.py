import numpy as np
import matplotlib.pyplot as plt
from m_over_H import lam_from_mu
from pathlib import Path
from fit_GW_spectra import load_data_kstar
from Veff_Daniel import EffectivePotential

M_PL = 2.435e18


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
    base_dir = Path("../output/mH1e2_512")
    file = base_dir / "spectra_gws.txt"
    mH = 10**2
    mu = M_PL * 10 ** (-7)
    lam = lam_from_mu(mu, mH)
    omega_star = mu
    H = mu**2 / (np.sqrt(12) * M_PL * np.sqrt(lam))
    g_star = 106.75

    data = load_data_kstar(file, mu)

    plt.figure(figsize=(8, 6))

    redshift_amplitude_factor = 1.67e-5 * (100 / g_star) ** (1 / 3)

    omega_gw_0 = data.omega_gw_star * redshift_amplitude_factor

    plt.plot(kH, omega_gw_0)
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
