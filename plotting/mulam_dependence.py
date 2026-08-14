import matplotlib.pyplot as plt
from pathlib import Path
from plot_data import save_figure
from load_data import load_gw_spectra
from simulation import Simulation
import numpy as np

if __name__ == "__main__":
    base_dirs = [
        # "mH1e3_mu7_lam-18/",
        "mH1e3_mu8_lam-16/",
        "mH1e3_mu9_lam-14/",
        "mH1e3_mu10_lam-12/",
        "mH1e3_mu11_lam-10/",
        "mH1e3_mu12_lam-8/",
        # "mH1e3_mu13_lam-6/",
        # "mH1e3_mu14_lam-4/",
        # "mH1e3_mu15_lam-2/",
    ]
    input_dirs = [Path("../output/mulam_comparison") / d for d in base_dirs]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("figures"), m_over_H) for input_dir in input_dirs
    ]
    labels = [
        # r"$\mu=2.435\times 10^{7}$",
        r"$\mu=2.435\times 10^{8}$",
        r"$\mu=2.435\times 10^{9}$",
        r"$\mu=2.435\times 10^{10}$",
        r"$\mu=2.435\times 10^{11}$",
        r"$\mu=2.435\times 10^{12}$",
        # r"$\mu=2.435\times 10^{13}$",
        # r"$\mu=2.435\times 10^{14}$",
        # r"$\mu=2.435\times 10^{15}$",
    ]

    mus = [
        # 2.435e7,
        2.435e8,
        2.435e9,
        2.435e10,
        2.435e11,
        2.435e12,
    ]  # , 2.435e13, 2.435e14, 2.435e15]
    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    dOmega = np.abs(peaks[-2] - peaks[-3]) / peaks[0] * 100  # %
    print(f"{dOmega=:.3f} %")
    fig, ax = plt.subplots()
    ax.plot(mus, peaks, linestyle="", marker=".", markersize=12)
    ax.plot(mus[-2], peaks[-2], linestyle="", marker="*", color="red", markersize=12)
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
