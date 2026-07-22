import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pathlib import Path
from plot_data import save_figure
from load_data import load_average_field
from simulation import Simulation

if __name__ == "__main__":
    base_dirs = [
        "mH1e4_tMax200",
        "mH1e4_tMax600",
        "mH1e4_tMax2000",
        "mH1e4_tMax6000",
        "mH1e4_tMax20000",
    ]
    input_dirs = [Path("../output/tMax") / d for d in base_dirs]
    m_over_H = 1e4
    sims = [
        Simulation(input_dir, Path(".figures/tMax"), m_over_H)
        for input_dir in input_dirs
    ]
    final_phis = np.empty(len(sims))
    eta_max = np.empty(len(sims))
    for i, sim in enumerate(sims):
        eta, phi = load_average_field(sim.input_dir / "average_scalar_0.txt")
        final_phis[i] = phi[-1]
        eta_max[i] = eta[-1]

    fig, ax = plt.subplots()
    ax.plot(eta_max, final_phis, ls="", marker="*", markersize=8)
    ax.set_xlabel(r"$\tilde \eta_\mathrm{max}$")
    ax.set_ylabel(r"$\langle \tilde\phi \rangle(\tilde \eta_\mathrm{max})$")
    ax.set_xscale("log")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(np.logspace(2, 5, num=4))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(eta_max) * 0.8, max(eta_max) * 1.2)
    ax.set_ylim(min(final_phis) * 0.99, max(final_phis) * 1.01)
    save_figure(fig, Path("./figures/tMax/tMax.pdf"))
