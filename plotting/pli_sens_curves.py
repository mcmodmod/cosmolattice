import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

plt.rcParams.update(
    {
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "axes.labelsize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.constrained_layout.use": True,
    }
)


def plot_labels(ax):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.text(7e-10, 5e-12, r"$\mathrm{SKA}$", c=colors[0], fontsize=17, rotation=90)
    ax.text(5e-5, 1e-10, r"$\mathrm{LISA}$", c=colors[1], fontsize=17, rotation=295)
    ax.text(
        1.3e-7, 5e-12, r"$\mu\mathrm{ARES}$", c=colors[2], fontsize=17, rotation=310
    )
    ax.text(3e1, 1e-10, r"$\mathrm{DECIGO}$", c=colors[3], fontsize=17, rotation=70)
    ax.text(2e2, 1e-10, r"$\mathrm{ET}$", c=colors[4], fontsize=17, rotation=60)
    ax.text(2e0, 3e-11, r"$\mathrm{B-DECIGO}$", c=colors[5], fontsize=17, rotation=60)
    ax.text(8e0, 4e-11, r"$\mathrm{BBO}$", c=colors[6], fontsize=17, rotation=65)


if __name__ == "__main__":
    filenames = [
        "PLISensCurves/" + filename for filename in os.listdir("PLISensCurves")
    ]
    freq = dict()
    sens = dict()
    for file in filenames:
        with open(file, "rb") as f:
            data = np.array(pickle.load(f))
            freq[file] = data[0, :]
            sens[file] = data[1, :]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for file in filenames:
        print(file)
        ax.plot(freq[file], sens[file])
        step = 100
        if len(sens[file]) > 1000:
            step = 10
        else:
            step = 1
        ax.fill_between(freq[file][::step], sens[file][::step], 5e-8, alpha=0.1)
    plot_labels(ax)
    ax.set_xlabel(r"$f_0\,\mathrm{[Hz]}$")
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}$")
    ax.set_ylim(1e-18, 5e-8)
    ax.set_xlim(1e-10, 7e2)
    ax.set_xticks([10**i for i in range(-10, 3, 2)])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(alpha=0.4)
    fig.savefig("figures/sens_curves.pdf", format="pdf", backend="pgf")
