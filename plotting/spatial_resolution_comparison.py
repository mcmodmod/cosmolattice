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
        "mH1e3_N64",
        "mH1e3_N128",
        "mH1e3_N256",
        "mH1e3_N512",
    ]
    input_dirs = [
        Path("../output/spatial_resolution_comparison") / d for d in base_dirs
    ]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H) for input_dir in input_dirs
    ]
    labels = [
        r"$\delta \tilde x\simeq 0.98$",
        r"$\delta \tilde x\simeq 0.49$",
        r"$\delta \tilde x\simeq 0.25$",
        r"$\delta \tilde x\simeq 0.12$",
        # r"$\delta \tilde x=0.06$",
    ]
    Ns = [64, 128, 256, 512]
    dx = np.array([2 * np.pi / (N * 0.1) for N in Ns])
    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    # for i, sim in enumerate(sims):
    #     f_star, omega_gw_star = load_last_gw_spectrum_f(
    #         sim.input_dir / "spectra_gws.txt", mu
    #     )
    #     _, peaks[i] = spectrum_peak(f_star, omega_gw_star)
    dOmega = np.abs(peaks[-1] - peaks[-2]) / peaks[-1] * 100  # %
    print(f"{dOmega=:.2f} %")
    fig, ax = plt.subplots()
    ax.plot(dx, peaks, linestyle="--", color="grey")
    for x, p in zip(dx, peaks):
        ax.plot(x, p, marker=".", markersize=20)
    ax.invert_xaxis()
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$", fontsize=24)
    ax.set_xlabel(r"$\delta \tilde x$", fontsize=24)
    save_figure(fig, Path("./figures/spatial_resolution_comparison.pdf"))

    fig, ax = plt.subplots()
    for i, sim in enumerate(sims):
        # Last time step:
        _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")
        spectrum = load_gw_spectra(sim.input_dir / "spectra_gws.txt")[-1]
        kappa = spectrum["kappa"] / a[-1]  # * sim.omega_star / sim.H
        ax.plot(
            kappa,
            spectrum["omega_gw"],
            label=labels[i],
            linewidth=2,
        )
    ax.set_xlabel(r"$k_\mathrm{phys}/\mu$", fontsize=24)
    ax.set_ylabel(r"$h^2\Omega_\mathrm{GW}$", fontsize=24)
    ax.set(xscale="log", yscale="log")
    ax.legend(frameon=False, fontsize=19)
    savefile = Path("./figures/spatial_resolution_comparison_spectra.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
