import matplotlib.pyplot as plt
from pathlib import Path
from plot_data import save_figure
from load_data import load_gw_spectra
from simulation import Simulation
import numpy as np

if __name__ == "__main__":
    base_dirs = [
        "mH1e3_mu10_lam-12/",
        # "mH1e3_mu11_lam-10/",
        "mH1e3_mu12_lam-8/",
        "mH1e3_mu13_lam-6/",
        "mH1e3_mu14_lam-4/",
    ]
    input_dirs = [Path("../output/mulam_dependence") / d for d in base_dirs]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("figures"), m_over_H) for input_dir in input_dirs
    ]
    labels = [
        r"$\mu=2.435\times 10^{10}$",
        # r"$\mu=2.435\times 10^{11}$",
        r"$\mu=2.435\times 10^{12}$",
        r"$\mu=2.435\times 10^{13}$",
        r"$\mu=2.435\times 10^{14}$",
    ]

    mus = [2.435e10, 2.435e12, 2.435e13, 2.435e14]
    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    dOmega = np.abs(peaks[1] - peaks[0]) / peaks[0] * 100  # %
    print(f"{dOmega=:.3f} %")
    fig, ax = plt.subplots()
    ax.plot(mus, peaks, linestyle="", marker=".", markersize=12)
    ax.plot(mus[0], peaks[0], linestyle="", marker="*", color="red", markersize=12)
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$\mu$")
    ax.set_xscale("log")
    save_figure(fig, Path("./figures/mulam_comparison.pdf"))
    fig, ax = plt.subplots()
    for i, sim in enumerate(sims):
        spectrum = load_gw_spectra(sim.input_dir / "spectra_gws.txt")[-1]
        k_over_H = spectrum["kappa"] * sim.omega_star / sim.H
        ax.plot(k_over_H, spectrum["omega_gw"], label=labels[i], alpha=0.75)
    ax.set_ylim(1e-17, 1e-5)
    ax.set(xscale="log", yscale="log")
    ax.set(xlabel=r"$k/H$", ylabel=r"$h^2\Omega_\mathrm{GW}$")
    fig.legend(loc="center")

    savename = Path("./figures/mulam_comparison_spectra.pdf")
    save_figure(fig, savename)
