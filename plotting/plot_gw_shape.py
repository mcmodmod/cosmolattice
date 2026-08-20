import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from simulation import Simulation
from load_data import (
    load_scale_factor,
    load_single_gw_spectrum,
    spectrum_peak,
)
from plot_data import save_figure


def main():

    base_dirs = [
        "mH1e3_512_new",
        "mH1e4_512_new",
        "mH1e5_512_new",
        "mH1e6_512_new",
        "mH1e7_512_new",
        "mH1e8_512_new",
        "mH1e9_512_new",
    ]
    input_dirs = [Path("../output") / d for d in base_dirs]
    m_over_Hs = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H)
        for input_dir, m_over_H in zip(input_dirs, m_over_Hs)
    ]
    labels = [
        r"$\mu/H = 10^3$",
        r"$\mu/H = 10^4$",
        r"$\mu/H = 10^5$",
        r"$\mu/H = 10^6$",
        r"$\mu/H = 10^7$",
        r"$\mu/H = 10^8$",
        r"$\mu/H = 10^9$",
    ]

    labels = {sim: l for sim, l in zip(sims, labels)}
    fig, ax = plt.subplots()
    for sim in sims:
        kappa, omega_gw = load_single_gw_spectrum(
            sim.input_dir / "spectra_gws.txt", 119
        )
        kappa = kappa  # * sim.omega_star / sim.H
        k_peak, omega_peak = spectrum_peak(kappa, omega_gw)
        ax.plot(
            kappa / k_peak,
            omega_gw / omega_peak,
            label=labels[sim],
        )
        ax.vlines(
            [4],
            ymin=1e-10,
            ymax=1e3,
            linestyle="--",
            color="black",
        )
    ax.set_ylim(1e-8, 5e1)
    ax.set(
        xlabel=r"$k/k_\mathrm{peak}$",
        ylabel=r"$\Omega_\mathrm{GW} / \Omega_\mathrm{GW}^\mathrm{peak}$",
    )
    ax.set(xscale="log", yscale="log")
    ax.legend(frameon=False)
    savefile = Path("./figures/common_shape_gw.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
