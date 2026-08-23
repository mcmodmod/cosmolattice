import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_data import save_figure

# -------------------------------------------------------------------------
# Load exclusion line
# -------------------------------------------------------------------------

with open("FOPT_exclusion_line.p", "rb") as file:
    g_bl, m_zprime = pickle.load(file)


# -------------------------------------------------------------------------
# Plot limits
# -------------------------------------------------------------------------

x_min = 5e4
x_max = 4e7
y_min = 1e-4
y_max = 2e-1

g_bl = np.append(g_bl, [y_max])
m_zprime = np.append(m_zprime, [1e7])

m_fill = np.append(m_zprime, [1e7, x_max])
g_fill = np.append(g_bl, [y_max, y_max])


# -------------------------------------------------------------------------
# Load SNR grids
# -------------------------------------------------------------------------

data = np.load("SNR_scan.npz")

m_grid = data["m_zprime"]
g_grid = data["g_bl"]

M, G = np.meshgrid(m_grid, g_grid)


# -------------------------------------------------------------------------
# Allowed parameter region
# -------------------------------------------------------------------------

# Classical-rolling boundary
sort_idx = np.argsort(m_fill)

g_classical_max = np.interp(
    m_grid,
    m_fill[sort_idx],
    g_fill[sort_idx],
    left=np.nan,
    right=np.nan,
)

classical_mask = G >= g_classical_max[np.newaxis, :]


# m_phi < 2 m_h boundary
m_h = 125.1
g_higgs_grid = 4 * np.pi * m_h / (np.sqrt(6) * m_grid)

higgs_mask = G <= g_higgs_grid[np.newaxis, :]


# Everything outside the physical region is masked
forbidden_mask = classical_mask | higgs_mask


# -------------------------------------------------------------------------
# Experiments
# -------------------------------------------------------------------------

experiments = {
    "LISA": {
        "snr": data["snr_lisa"],
        "threshold": 10,
    },
    "BBO": {
        "snr": data["snr_bbo"],
        "threshold": 10,
    },
    "DECIGO": {
        "snr": data["snr_decigo"],
        "threshold": 10,
    },
    "ET": {
        "snr": data["snr_et"],
        "threshold": 5,
    },
}


# -------------------------------------------------------------------------
# Create one figure per experiment
# -------------------------------------------------------------------------

for name, experiment in experiments.items():

    snr = experiment["snr"]
    threshold = experiment["threshold"]

    snr_masked = np.ma.masked_invalid(snr)
    snr_masked = np.ma.masked_where(forbidden_mask, snr_masked)

    # 0 = undetectable, 1 = detectable
    detectable = np.ma.masked_where(
        forbidden_mask,
        (snr >= threshold).astype(float),
    )

    fig, ax = plt.subplots()

    # ---------------------------------------------------------------------
    # Detectable / undetectable regions
    # ---------------------------------------------------------------------

    ax.contourf(
        M,
        G,
        detectable,
        levels=[-0.5, 0.5, 1.5],
        colors=["lightsteelblue", "cornflowerblue"],
        alpha=0.8,
    )

    # ---------------------------------------------------------------------
    # Detection threshold
    # ---------------------------------------------------------------------

    snr_min = snr_masked.min()
    snr_max = snr_masked.max()

    if snr_min <= threshold <= snr_max:
        ax.contour(
            M,
            G,
            snr_masked,
            levels=[threshold],
            colors="black",
            linewidths=2.5,
        )

    # ---------------------------------------------------------------------
    # Forbidden regions
    # ---------------------------------------------------------------------

    # Bubble-percolation region
    ax.fill_between(
        m_fill,
        g_fill,
        y_max,
        color="0.7",
        zorder=10,
    )

    # m_phi < 2 m_h region
    m = np.logspace(np.log10(x_min), np.log10(x_max), 500)
    g_higgs = 4 * np.pi * m_h / (np.sqrt(6) * m)

    mask = g_higgs > y_min

    ax.fill_between(
        m[mask],
        y_min,
        g_higgs[mask],
        color="0.7",
        zorder=10,
    )

    ax.plot(
        m[mask],
        g_higgs[mask],
        color="black",
        zorder=18,
    )

    ax.plot(
        m_zprime,
        g_bl,
        color="black",
        zorder=18,
    )

    # ---------------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------------

    ax.text(
        1.5e5,
        7e-2,
        "Bubble\npercolation",
        ha="center",
        va="center",
        fontsize=20,
        zorder=12,
    )

    ax.text(
        2e5,
        5e-4,
        r"$m_\varphi < 2m_h$",
        ha="center",
        va="center",
        fontsize=22,
        zorder=12,
    )

    ax.text(
        0.97,
        0.95,
        name,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=18,
    )

    # ---------------------------------------------------------------------
    # Plot formatting
    # ---------------------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(r"$m_{Z'}\,[\mathrm{GeV}]$")
    ax.set_ylabel(r"$g_{B-L}$")

    ax.grid(False)

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------

    savefile = Path(f"./figures/SNR/{name.lower()}_snr.pdf")
    save_figure(fig, savefile)

    plt.close(fig)
