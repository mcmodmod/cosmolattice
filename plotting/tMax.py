import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pathlib import Path

plt.rcParams.update(
    {
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "axes.labelsize": 20,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.constrained_layout.use": True,
    }
)
if __name__ == "__main__":
    base_dirs = [
        "mH1e4_tMax200",
        "mH1e4_tMax600",
        "mH1e4_tMax2000",
        "mH1e4_tMax6000",
        "mH1e4_tMax20000",
    ]
    input_dirs = [Path("../output/tMax") / d for d in base_dirs]
    filenames = [input_dir / Path("average_scalar_0.txt") for input_dir in input_dirs]

    final_phis = np.empty(len(filenames))
    etaMax = np.empty(len(filenames))
    for i, filename in enumerate(filenames):
        phiDat = np.loadtxt(filename)
        etaMax[i] = phiDat[-1, 0] + 1
        final_phis[i] = phiDat[-1, 1]
    fig, ax = plt.subplots()
    ax.plot(etaMax, final_phis, ls="", marker="*", markersize=8)
    ax.set_xlabel(r"$\tilde \eta_\mathrm{max}$")
    ax.set_ylabel(r"$\langle \tilde\phi \rangle(\tilde \eta_\mathrm{max})$")
    ax.set_xscale("log")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(np.logspace(2, 5, num=4))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(etaMax) * 0.8, max(etaMax) * 1.2)
    ax.set_ylim(min(final_phis) * 0.99, max(final_phis) * 1.01)
    # ax.grid(True)
    fig.savefig("./figures/tMax/tMax.pdf", format="pdf", backend="pgf")
