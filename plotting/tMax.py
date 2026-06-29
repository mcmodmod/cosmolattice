import numpy as np
import matplotlib.pyplot as plt
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
    base_dirs = ["mH1e4_tMax200", "mH1e4_tMax2000", "mH1e4_tMax20000"]
    input_dirs = [Path("../output/tMax") / d for d in base_dirs]
    filenames = [input_dir / Path("average_scalar_0.txt") for input_dir in input_dirs]

    final_phis = np.empty(len(filenames))
    etaMax = np.empty(len(filenames))
    for i, filename in enumerate(filenames):
        phiDat = np.loadtxt(filename)
        etaMax[i] = phiDat[-1, 0] + 1
        final_phis[i] = phiDat[-1, 1]
    fig, ax = plt.subplots()
    ax.plot(etaMax, final_phis, ls="", marker="*")
    ax.set_xscale("log")
    fig.savefig("./figures/tMax/tMax.pdf", format="pdf", backend="pgf")
