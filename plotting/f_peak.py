import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from load_data import load_gw_spectra, frequency_from_kappa
from plot_data import save_figure
from pathlib import Path


def main():
    base_dirs = [
        "mH1e2_512",
        "mH1e3_512",
        "mH1e4_512",
        "mH1e5_512",
        "mH1e6_512",
        "mH1e7_512",
        "mH1e8_512",
        "mH1e9_512",
    ]
    input_dirs = [Path("../output") / d for d in base_dirs]
    output_dirs = [Path("./figures") / d for d in base_dirs]

    m_over_Hs = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])

    sims = [
        Simulation(inp, out, m_over_H)
        for inp, out, m_over_H in zip(input_dirs, output_dirs, m_over_Hs)
    ]

    peaks = np.empty(len(sims))
    f_peaks = np.empty(len(sims))

    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")

        max_omega = -np.inf
        max_kappa = np.nan

        for spec in spectra:
            idx = np.argmax(spec["omega_gw"][2:])
            omega = spec["omega_gw"][idx]

            if omega > max_omega:
                max_omega = omega
                max_kappa = spec["kappa"][idx]

        peaks[i] = max_omega
        f_peaks[i] = frequency_from_kappa(max_kappa, sim.mu)

    fig, ax = plt.subplots()
    ax.plot(m_over_Hs, f_peaks, linestyle="", marker=".", markersize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m/H$")
    ax.set_ylabel(r"$f_{\rm peak}$")
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    save_figure(fig, Path("./figures/f_peaks.pdf"))


if __name__ == "__main__":
    main()
    print("\nAll done!")
