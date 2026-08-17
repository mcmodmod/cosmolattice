import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from simulation import Simulation
from load_data import load_gw_spectra, load_scale_factor
from plot_data import save_figure


def main():

    base_dirs = [
        # "mH1e3_N32_kIR16e-1",
        "mH1e3_N64_kIR8e-1",
        # "mH1e3_N128_kIR4e-1",
        "mH1e3_N256_kIR2e-1",
        "mH1e3_N512_kIR1e-1",
    ]
    input_dirs = [Path("../output/box_size_comparison") / d for d in base_dirs]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H) for input_dir in input_dirs
    ]
    # dx = [2 * np.pi / (N * kIR) for kIR, N in zip(kIRs, Ns)]
    # Ns = [64, 128, 256, 512]
    kIRs = [8e-1, 2e-1, 1e-1]
    Ls = [2 * np.pi / (kIR) for kIR in kIRs]
    for L in Ls:
        print(L / np.pi)
    labels = [
        r"$\tilde L=2.5 \pi$",
        # r"$\tilde L=5 \pi$",
        r"$\tilde L=10 \pi$",
        r"$\tilde L=20 \pi$",
    ]

    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    dOmega = np.abs(peaks[-1] - peaks[-2]) / peaks[-1] * 100  # %
    print(f"{dOmega=:.2f} %")
    fig, ax = plt.subplots()
    ax.plot(Ls, peaks, linestyle="--", color="grey")
    for L, p in zip(Ls, peaks):
        ax.plot(L, p, linestyle="", marker=".", markersize=20)
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$", fontsize=24)
    ax.set_xlabel(r"$\tilde L$", fontsize=24)
    # ax.tick_params(axis="x", which="minor", bottom=True, top=True)
    save_figure(fig, Path("./figures/box_size_comparison.pdf"))

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
    savefile = Path("./figures/box_size_comparison_spectra.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
