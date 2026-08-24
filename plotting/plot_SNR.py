import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_data import save_figure
from scipy.interpolate import RegularGridInterpolator

# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------

x_min, x_max = 5e4, 4e7
y_min, y_max = 1e-4, 2e-1

m_h = 125.1


# -------------------------------------------------------------------------
# Load exclusion line
# -------------------------------------------------------------------------

tamara_m, tamara_g = np.loadtxt("tamara_exclusion_line.txt", unpack=True)

with open("FOPT_exclusion_line.p", "rb") as file:
    g_bl, m_zprime = pickle.load(file)

# Extend exclusion boundary to the top/right plot edges
m_fill = np.append(m_zprime, [1e7, 1e7, x_max])
g_fill = np.append(g_bl, [y_max, y_max, y_max])


# -------------------------------------------------------------------------
# Load SNR grids
# -------------------------------------------------------------------------

data = np.load("SNR_scan_2.npz")

m_grid = data["m_zprime"]
g_grid = data["g_bl"]

M, G = np.meshgrid(m_grid, g_grid)


# -------------------------------------------------------------------------
# Load m/H grid
# -------------------------------------------------------------------------

mh_data = np.load("./data/m_over_H_grid.npz")

mh_m_grid = mh_data["m_grid"]
mh_g_grid = mh_data["g_grid"]
MH = mh_data["m_over_H"]

MH_M, MH_G = np.meshgrid(mh_m_grid, mh_g_grid)

mh_level = 1000.0

# Use a temporary contour plot to extract the contour coordinates
fig_tmp, ax_tmp = plt.subplots()

cs = ax_tmp.contour(
    MH_M,
    MH_G,
    MH,
    levels=[mh_level],
)

# Each entry is an (N, 2) array:
# column 0 = m
# column 1 = g
mh1000_segments = [segment for segment in cs.allsegs[0] if len(segment) > 1]

plt.close(fig_tmp)


# -------------------------------------------------------------------------
# Physical parameter region -- SNR grid
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


# -------------------------------------------------------------------------
# Interpolate m/H from its 75x75 grid onto the 100x100 SNR grid
# -------------------------------------------------------------------------

mh_interpolator = RegularGridInterpolator(
    (mh_g_grid, mh_m_grid),
    MH,
    bounds_error=False,
    fill_value=np.nan,
)

points_snr = np.column_stack(
    (
        G.ravel(),
        M.ravel(),
    )
)

MH_on_snr_grid = mh_interpolator(points_snr).reshape(M.shape)


# m/H < 1 forbidden region, evaluated on the SNR grid
mh_forbidden_mask = MH_on_snr_grid < 1.0


# Combined forbidden region on the SNR grid
forbidden_mask = classical_mask | higgs_mask | mh_forbidden_mask

# -------------------------------------------------------------------------
# Physical parameter region -- native m/H grid
# -------------------------------------------------------------------------

# Bubble-percolation boundary evaluated on the m/H grid
g_classical_max_mh = np.interp(
    mh_m_grid,
    m_fill[order],
    g_fill[order],
    left=np.nan,
    right=np.nan,
)

classical_mask_mh = MH_G >= g_classical_max_mh[None, :]


# m_phi < 2 m_h boundary evaluated on the m/H grid
g_higgs_grid_mh = 4 * np.pi * m_h / (np.sqrt(6) * mh_m_grid)

higgs_mask_mh = MH_G <= g_higgs_grid_mh[None, :]


# m/H < 1
mh_forbidden_mask_mh = MH < 1.0


# Combined forbidden mask on the native m/H grid
forbidden_mask_mh = classical_mask_mh | higgs_mask_mh | mh_forbidden_mask_mh


# Masked m/H array used for the m/H = 1000 contour
MH_allowed = np.ma.masked_invalid(MH)
MH_allowed = np.ma.masked_where(
    forbidden_mask_mh,
    MH_allowed,
)
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
# Determine common SNR contour levels
# -------------------------------------------------------------------------

# Use one common colour scale for all four panels so that the same shade
# corresponds to the same SNR everywhere.
all_snr = []

for snr, _ in experiments.values():

    snr_valid = np.ma.masked_invalid(snr)
    snr_valid = np.ma.masked_where(forbidden_mask, snr_valid)

    # Logarithmic colour scales require strictly positive values.
    snr_valid = np.ma.masked_less_equal(snr_valid, 0)

    if snr_valid.count() > 0:
        all_snr.append(snr_valid.compressed())

all_snr = np.concatenate(all_snr)

snr_min = all_snr.min()
snr_min = 1e-6
snr_max = all_snr.max()
print(f"{snr_min=:.2E}")
print(f"{snr_max=:.2E}")
snr_max = 1e14

# Logarithmically spaced filled-contour levels.
contour_levels = np.geomspace(
    snr_min, snr_max, (int(np.log10(snr_max / snr_min))) // 2 + 1
)

snr_norm = LogNorm(
    vmin=snr_min,
    vmax=snr_max,
)


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
    # sharex=True,
    # sharey=True,
)

axes = axes.flatten()

# Curve used for the m_phi < 2 m_h excluded region
m = np.logspace(
    np.log10(x_min),
    np.log10(x_max),
    500,
)

g_higgs = 4 * np.pi * m_h / (np.sqrt(6) * m)
higgs_plot_mask = g_higgs > y_min


for ax, (name, (snr, threshold)) in zip(axes, experiments.items()):

    # ---------------------------------------------------------------------
    # m/H = 1000 contour
    # ---------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Forbidden mask on the m_over_H grid
    # -------------------------------------------------------------------------

    # Bubble-percolation boundary evaluated on mh_m_grid
    order = np.argsort(m_fill)

    g_classical_max_mh = np.interp(
        mh_m_grid,
        m_fill[order],
        g_fill[order],
        left=np.nan,
        right=np.nan,
    )

    classical_mask_mh = MH_G >= g_classical_max_mh[None, :]

    # m_phi < 2 m_h boundary evaluated on mh_m_grid
    g_higgs_mh = 4 * np.pi * m_h / (np.sqrt(6) * mh_m_grid)

    higgs_mask_mh = MH_G <= g_higgs_mh[None, :]

    # Combined forbidden region
    forbidden_mask_mh = classical_mask_mh | higgs_mask_mh
    MH_masked = np.ma.masked_where(forbidden_mask_mh, MH)

    ax.contour(
        MH_M,
        MH_G,
        MH_masked,
        levels=[1000],
        colors="tab:red",
        linestyles="-",
        linewidths=2.0,
        zorder=17,
    )
    # ---------------------------------------------------------------------
    # Mask SNR
    # ---------------------------------------------------------------------

    snr_masked = np.ma.masked_invalid(snr)
    snr_masked = np.ma.masked_where(forbidden_mask, snr_masked)
    snr_masked = np.ma.masked_less_equal(snr_masked, 0)

    # ---------------------------------------------------------------------
    # SNR contours
    # ---------------------------------------------------------------------

    snr_masked2 = np.ma.masked_invalid(snr)
    snr_masked2 = np.ma.masked_less_equal(snr_masked2, 0)
    contourf = ax.contourf(
        M,
        G,
        snr_masked2,
        levels=contour_levels,
        cmap="Greens",
        norm=snr_norm,
        extend="both",
    )

    # ---------------------------------------------------------------------
    # Detection threshold
    # ---------------------------------------------------------------------

    if snr_masked.count() > 0 and snr_masked.min() <= threshold <= snr_masked.max():
        threshold_contour = ax.contour(
            M,
            G,
            snr_masked,
            levels=[threshold],
            colors="blue",
            linewidths=2.5,
            linestyles="-",
            zorder=15,
        )

        # Optional label directly on the threshold contour
        # ax.clabel(
        #     threshold_contour,
        #     fmt={threshold: rf"$\mathrm{{SNR}}={threshold}$"},
        #     inline=True,
        #     fontsize=19,
        # )

    # ---------------------------------------------------------------------
    # Forbidden regions
    # ---------------------------------------------------------------------

    ax.fill_between(
        m_fill,
        g_fill,
        y_max,
        color="0.7",
        zorder=10,
    )

    ax.fill_between(
        m[higgs_plot_mask],
        y_min,
        g_higgs[higgs_plot_mask],
        color="0.7",
        zorder=10,
    )

    ax.plot(
        m[higgs_plot_mask],
        g_higgs[higgs_plot_mask],
        color="black",
        linewidth=1.5,
        zorder=18,
    )
    # ax.plot(
    #     tamara_m,
    #     tamara_g,
    #     color="magenta",
    #     linewidth=2,
    #     zorder=18,
    # )

    ax.plot(
        m_fill,
        g_fill,
        color="black",
        linewidth=1.5,
        zorder=18,
    )

    # ---------------------------------------------------------------------
    # m/H < 1 forbidden region
    # ---------------------------------------------------------------------

    MH_plot = np.ma.masked_invalid(MH)

    if np.nanmin(MH) < 1.0:

        ax.contourf(
            MH_M,
            MH_G,
            MH_plot,
            levels=[np.nanmin(MH), 1.0],
            colors=["0.7"],
            zorder=10,
        )
    MH_plot = np.ma.masked_invalid(MH_masked)
    if np.nanmin(MH) <= 1.0 <= np.nanmax(MH):
        mh1_contour = ax.contour(
            MH_M,
            MH_G,
            MH_plot,
            levels=[1.0],
            colors="black",
            linewidths=1.5,
            linestyles="-",
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
        8e6,
        1e-3,
        r"$\frac{m_{\varphi,\mathrm{QCD}}}{H} < 1$",
        ha="center",
        va="center",
        fontsize=22,
        zorder=12,
    )

    ax.set_title(
        name,
        fontsize=22,
    )

    # ---------------------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.grid(False)


# -------------------------------------------------------------------------
# Axis labels
# -------------------------------------------------------------------------

# Bottom row
for ax in axes[2:]:
    ax.set_xlabel(
        r"$m_{Z'}\,[\mathrm{GeV}]$",
        fontsize=20,
    )

# Left column
for ax in axes[::2]:
    ax.set_ylabel(
        r"$g_{B-L}$",
        fontsize=20,
    )


# -------------------------------------------------------------------------
# Shared colour bar
# -------------------------------------------------------------------------

cbar = fig.colorbar(
    contourf,
    ax=axes,
    pad=0.02,
    fraction=0.04,
)

cbar.set_label(
    r"$\rho_{\mathrm{SNR}}$",
    fontsize=20,
)


# -------------------------------------------------------------------------
# Layout and save
# -------------------------------------------------------------------------

# fig.subplots_adjust(
#     left=0.09,
#     right=0.86,
#     bottom=0.09,
#     top=0.94,
#     wspace=0.08,
#     hspace=0.12,
# )

savefile = Path("./figures/SNR/all_experiments_snr.pdf")
save_figure(fig, savefile)

plt.close(fig)
