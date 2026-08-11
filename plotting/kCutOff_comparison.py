import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from simulation import Simulation
from load_data import load_gw_spectra
from plot_data import save_figure


def main():

    base_dirs = [
        "mH1e3_kCutOff_0",
        "mH1e3_kCutOff_4",
    ]
    input_dirs = [Path("../output/kCutOff_comparison") / d for d in base_dirs]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H) for input_dir in input_dirs
    ]
    dts = [0, 4]
    labels = [
        r"$\mathtt{kCutOff=inf}$",
        r"$\mathtt{kCutOff=4}$",
    ]

    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    dOmega = np.abs(peaks[1] - peaks[0]) / peaks[0] * 100  # %
    print(f"{dOmega=:.3f} %")
    fig, ax = plt.subplots()
    ax.plot(dts, peaks, linestyle="", marker=".", markersize=12)
    ax.plot(dts[1], peaks[1], linestyle="", marker="*", color="red", markersize=12)
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$k_\mathrm{cut}$")
    # ax.tick_params(axis="x", which="minor", bottom=True, top=True)
    save_figure(fig, Path("./figures/kCutOff_comparison.pdf"))

    fig, ax = plt.subplots()
    for i, sim in enumerate(sims):
        # Last time step:
        spectrum = load_gw_spectra(sim.input_dir / "spectra_gws.txt")[-1]
        kappa = spectrum["kappa"]  # * sim.omega_star / sim.H
        ax.plot(
            kappa,
            spectrum["omega_gw"],
            label=labels[i],
        )
    ax.set(xlabel=r"$\tilde k$", ylabel=r"$h^2\Omega_\mathrm{GW}$")
    ax.set(xscale="log", yscale="log")
    ax.legend()
    savefile = Path("./figures/kCutOff_comparison_spectra.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
