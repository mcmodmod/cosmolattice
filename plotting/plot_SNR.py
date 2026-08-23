import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from plot_data import save_figure

# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------

x_min, x_max = 5e4, 4e7
y_min, y_max = 1e-4, 2e-1

m_h = 125.1

region_cmap = ListedColormap(
    [
        "navy",  # undetectable
        "gold",  # detectable
    ]
)


# -------------------------------------------------------------------------
# Load exclusion line
# -------------------------------------------------------------------------

with open("FOPT_exclusion_line.p", "rb") as file:
    g_bl, m_zprime = pickle.load(file)

# Extend exclusion boundary to the top/right plot edges
m_fill = np.append(m_zprime, [1e7, 1e7, x_max])
g_fill = np.append(g_bl, [y_max, y_max, y_max])

# -------------------------------------------------------------------------
# Load SNR grids
# -------------------------------------------------------------------------

data = np.load("SNR_scan.npz")

m_grid = data["m_zprime"]
g_grid = data["g_bl"]

data2 = np.load("./data/m_over_H_grid.npz")

MH = data2["m_over_H"]
M, G = np.meshgrid(m_grid, g_grid)


# -------------------------------------------------------------------------
# Physical parameter region
# -------------------------------------------------------------------------

# Bubble-percolation boundary
order = np.argsort(m_fill)

g_classical_max = np.interp(
    m_grid,
    m_fill[order],
    g_fill[order],
    left=np.nan,
    right=np.nan,
)

classical_mask = G >= g_classical_max[None, :]


# m_phi < 2 m_h boundary
g_higgs_grid = 4 * np.pi * m_h / (np.sqrt(6) * m_grid)

higgs_mask = G <= g_higgs_grid[None, :]


# Combined forbidden region
forbidden_mask = classical_mask | higgs_mask


# -------------------------------------------------------------------------
# Experiments
# -------------------------------------------------------------------------

experiments = {
    "LISA": (data["snr_lisa"], 10),
    "BBO": (data["snr_bbo"], 10),
    "DECIGO": (data["snr_decigo"], 10),
    "ET": (data["snr_et"], 5),
}


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

for name, (snr, threshold) in experiments.items():

    # Masking is only used for the threshold contour.
    snr_masked = np.ma.masked_invalid(snr)
    snr_masked = np.ma.masked_where(forbidden_mask, snr_masked)

    # Binary field covering the entire SNR grid:
    # 0 = undetectable
    # 1 = detectable
    detectable = (snr >= threshold).astype(float)

    fig, ax = plt.subplots()

    # ---------------------------------------------------------------------
    # Detectable / undetectable regions
    # ---------------------------------------------------------------------

    ax.contourf(
        M,
        G,
        detectable,
        levels=[-0.5, 0.5, 1.5],
        colors=["navy", "gold"],
        alpha=0.8,
    )

    # ---------------------------------------------------------------------
    # Detection threshold
    # ---------------------------------------------------------------------

    if snr_masked.min() <= threshold <= snr_masked.max():
        ax.contour(
            M,
            G,
            snr_masked,
            levels=[threshold],
            colors="black",
            linewidths=2.5,
        )

    g_higgs_grid = 4 * np.pi * 125.1 / (np.sqrt(6) * M)

    higgs_mask = G < g_higgs_grid

    combined_mask = classical_mask | higgs_mask

    MH_masked = np.ma.masked_where(combined_mask, MH)

    ax.contour(
        M,
        G,
        MH_masked,
        levels=[1e3],
        colors="red",
        linestyles="solid",
    )
    # ---------------------------------------------------------------------
    # Forbidden regions
    # ---------------------------------------------------------------------

    # These are drawn on top of the blue background.

    ax.fill_between(
        m_fill,
        g_fill,
        y_max,
        color="0.7",
        zorder=10,
    )

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
        m_fill,
        g_fill,
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

    ax.set_title(name)
    # ax.text(
    #     0.97,
    #     0.95,
    #     name,
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=18,
    # )

    # ---------------------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------------------

    ax.set(
        xscale="log",
        yscale="log",
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        xlabel=r"$m_{Z'}\,[\mathrm{GeV}]$",
        ylabel=r"$g_{B-L}$",
    )

    ax.grid(False)

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------

    savefile = Path(f"./figures/SNR/{name.lower()}_snr.pdf")
    save_figure(fig, savefile)

    plt.close(fig)
