import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pathlib import Path
from plot_data import save_figure
from load_data import load_average_field, load_gw_spectra, load_scale_factor
from simulation import Simulation

if __name__ == "__main__":
    input_dir = Path("../output/tMax/mH1e3_tMax_500")
    m_over_H = 1e3
    sim = Simulation(input_dir, Path(".figures/"), m_over_H)
    tMax = np.sort(np.append(np.arange(1, 275, 20), [150, 275]))
    # tMax = np.arange(0, 500, 1)
    peaks = np.empty(len(tMax))
    for i, tm in enumerate(tMax):
        print(i, tm)
        spec = load_gw_spectra(sim.input_dir / "spectra_gws.txt")[tm - 1]
        peaks[i] = spec["omega_gw"][1:].max()

    dOmega = np.abs(peaks[-8] - peaks[-1]) / peaks[-1] * 100  # %
    print(f"{dOmega=:.3f} %")
    fig, ax = plt.subplots()
    ax.plot(tMax, peaks, linestyle="--", color="grey")
    ax.plot(tMax, peaks, linestyle="", marker=".", markersize=14)
    # ax.plot(tMax[1], peaks[1], linestyle="", marker="*", color="red", markersize=12)
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$\tilde \eta_\mathrm{max}$")
    # ax.tick_params(axis="x", which="minor", bottom=True, top=True)
    save_figure(fig, Path("./figures/tMax_comparison.pdf"))

    # final_phis = np.empty(len(sims))
    # eta_max = np.empty(len(sims))
    # for i, sim in enumerate(sims):
    #     eta, phi = load_average_field(sim.input_dir / "average_scalar_0.txt")
    #     final_phis[i] = phi[-1]
    #     eta_max[i] = eta[-1]
    #
    # fig, ax = plt.subplots()
    # ax.plot(eta_max, final_phis, ls="", marker="*", markersize=8)
    # ax.set_xlabel(r"$\tilde \eta_\mathrm{max}$")
    # ax.set_ylabel(r"$\langle \tilde\phi \rangle(\tilde \eta_\mathrm{max})$")
    # ax.set_xscale("log")
    # ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    # ax.set_xticks(np.logspace(2, 5, num=4))
    # ax.xaxis.set_minor_formatter(NullFormatter())
    # ax.set_xlim(min(eta_max) * 0.8, max(eta_max) * 1.2)
    # ax.set_ylim(min(final_phis) * 0.99, max(final_phis) * 1.01)
    # save_figure(fig, Path("./figures/tMax_comparison_final_phis.pdf"))
