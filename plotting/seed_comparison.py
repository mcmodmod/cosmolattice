import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from simulation import Simulation
from load_data import (
    load_last_gw_spectrum_f,
    spectrum_peak,
    load_gw_spectra,
    load_scale_factor,
)
from plot_data import save_figure


def main():

    base_dirs = [
        "mH1e3_0",
        "mH1e3_1",
        "mH1e3_2",
        "mH1e3_3",
        "mH1e3_4",
        # "mH1e3_5",
        "mH1e3_6",
        # "mH1e3_7",
    ]
    input_dirs = [Path("../output/seed_comparison") / d for d in base_dirs]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H) for input_dir in input_dirs
    ]
    # labels = []
    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    avg_peak = np.mean(peaks)
    std_peak = np.std(peaks)
    print(f"{avg_peak=:.2E}, {std_peak=:.2E}, {std_peak/avg_peak=:.3f}")
    x = [0, 1, 2, 3, 4, 6]
    fig, ax = plt.subplots()
    ax.plot(x, peaks, linestyle="--", color="grey")
    for el, p in zip(x, peaks):
        ax.plot(el, p, marker=".", markersize=20)
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$", fontsize=24)
    # ax.set_xlabel(r"$\delta \tilde x$", fontsize=24)
    save_figure(fig, Path("./figures/seed_comparison.pdf"))

    fig, ax = plt.subplots()
    for i, sim in enumerate(sims):
        # Last time step:
        _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")
        spectrum = load_gw_spectra(sim.input_dir / "spectra_gws.txt")[-1]
        kappa = spectrum["kappa"] / a[-1]  # * sim.omega_star / sim.H
        ax.plot(
            kappa,
            spectrum["omega_gw"],
            linewidth=2,
        )
    ax.set_xlabel(r"$k_\mathrm{phys}/\mu$", fontsize=24)
    ax.set_ylabel(r"$h^2\Omega_\mathrm{GW}$", fontsize=24)
    ax.set(xscale="log", yscale="log")
    # ax.legend(frameon=False, fontsize=19)
    savefile = Path("./figures/seed_comparison_spectra.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
