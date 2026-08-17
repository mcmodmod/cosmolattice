import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from simulation import Simulation
from load_data import load_gw_spectra, load_gw_energies
from plot_data import save_figure


def main():

    base_dirs = [
        "mH1e3_kCutOff_2",
        "mH1e3_kCutOff_4",
        "mH1e3_kCutOff_8",
        "mH1e3_kCutOff_10",
        "mH1e3_kCutOff_15",
        "mH1e3_kCutOff_20",
        "mH1e3_kCutOff_30",
        "mH1e3_kCutOff_50",
    ]
    input_dirs = [Path("../output/kCutOff_comparison") / d for d in base_dirs]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H) for input_dir in input_dirs
    ]
    kCutOffs = [2, 4, 8, 10, 15, 20, 30, 50]
    labels = [
        r"$\mathtt{kCutOff=2}$",
        r"$\mathtt{kCutOff=4}$",
        r"$\mathtt{kCutOff=8}$",
        r"$\mathtt{kCutOff=10}$",
        r"$\mathtt{kCutOff=15}$",
        r"$\mathtt{kCutOff=20}$",
        r"$\mathtt{kCutOff=30}$",
        r"$\mathtt{kCutOff=50}$",
    ]

    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    dOmega = np.abs(peaks[-1] - peaks[2]) / peaks[-1] * 100  # %
    print(f"{dOmega=:.4f} %")
    fig, ax = plt.subplots()
    ax.plot(kCutOffs, peaks, linestyle="", marker=".", markersize=12)
    # ax.plot(kCutOffs[2], peaks[2], linestyle="", marker="*", color="red", markersize=12)
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$k_\mathrm{cut}/\mu$")
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

    fig, ax = plt.subplots()
    for i, sim in enumerate(sims):
        eta, rho_frac, _ = load_gw_energies(sim.input_dir / "energy_gws.txt")
        ax.plot(eta, rho_frac, label=labels[i])
    ax.set(yscale="log")
    ax.set_xlabel(r"$\tilde\eta$")
    ax.set_ylabel(r"$\rho_\mathrm{GW}/\rho_\mathrm{tot}$")
    ax.legend()
    savefile = Path("./figures/kCutOff_comparison_gw_energies.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
