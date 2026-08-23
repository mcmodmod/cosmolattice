import pickle
from plot_data import save_figure
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from Veff_Daniel import EffectivePotential


def m_over_H(g_bl, m_zprime):
    veff = EffectivePotential(g_bl, m_zprime, vh_qcd=0.1)
    veff.interpolations()
    return veff.m_over_H()


with open("FOPT_exclusion_line.p", "rb") as file:
    g_bl, m_zprime = pickle.load(file)

fig, ax = plt.subplots()

x_min = 5e4
x_max = 4e7
y_min = 1e-4
y_max = 2e-1

g_bl = np.append(g_bl, [y_max])
m_zprime = np.append(m_zprime, [1e7])

m_fill = np.append(m_zprime, [1e7, x_max])
g_fill = np.append(g_bl, [y_max, y_max])


# -------------------------------------------------------------------------
# m/H contours in the Classical Rolling region
# -------------------------------------------------------------------------

data = np.load("./data/m_over_H_grid.npz")

m_grid = data["m_grid"]
g_grid = data["g_grid"]
MH = data["m_over_H"]

M, G = np.meshgrid(m_grid, g_grid)


# Find the upper boundary of the Classical Rolling region at each m_Z'.
#
# np.interp requires monotonically increasing x values, so sort the
# exclusion-line coordinates first.
sort_idx = np.argsort(m_fill)
m_boundary = m_fill[sort_idx]
g_boundary = g_fill[sort_idx]

g_classical_max = np.interp(
    m_grid,
    m_boundary,
    g_boundary,
    left=np.nan,
    right=np.nan,
)

# Mask everything outside the Classical Rolling region.
classical_mask = G > g_classical_max[np.newaxis, :]
# MH_masked = np.ma.masked_where(classical_mask, MH)
MH_masked = MH


# Requested contour thresholds.
levels = np.array(
    [
        1e-6,
        1e-4,
        1e-2,
        1e0,
        1e2,
        1e4,
        1e6,
    ]
)

# Use discrete shades from the Blues colormap.
cmap = plt.get_cmap("Blues", len(levels) - 1)

norm = colors.BoundaryNorm(
    boundaries=levels,
    ncolors=cmap.N,
)

contours = ax.contourf(
    M,
    G,
    MH_masked,
    levels=levels,
    cmap=cmap,
    norm=norm,
    # extend="both",
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

# Colorbar showing the m/H scale.
cbar = fig.colorbar(
    contours,
    ax=ax,
    pad=0.02,
    ticks=levels,
)

cbar.set_label(r"$m_{\varphi, \mathrm{QCD}}/H$")
cbar.ax.set_yticklabels(
    [
        r"$10^{-6}$",
        r"$10^{-4}$",
        r"$10^{-2}$",
        r"$10^{0}$",
        r"$10^{2}$",
        r"$10^{4}$",
        r"$10^{6}$",
    ]
)


# -------------------------------------------------------------------------
# Other regions / boundaries
# -------------------------------------------------------------------------

# Grey region above the Classical Rolling boundary.
ax.fill_between(
    m_fill,
    g_fill,
    y_max,
    color="0.7",
)

m = np.logspace(np.log10(x_min), np.log10(x_max), 500)

m_h = 125.1
g_higgs = 4 * np.pi * m_h / (np.sqrt(6) * m)

mask = g_higgs > y_min

ax.fill_between(
    m[mask],
    y_min,
    g_higgs[mask],
    color="0.7",
)

ax.plot(
    m[mask],
    g_higgs[mask],
    color="black",
    # linewidth=5,
)

ax.plot(
    m_zprime,
    g_bl,
    color="black",
    # linewidth=5,
)


# -------------------------------------------------------------------------
# Plot formatting
# -------------------------------------------------------------------------

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel(r"$m_{Z'}\,[\mathrm{GeV}]$")
ax.set_ylabel(r"$g_{B-L}$")

ax.text(
    1.5e5,
    7e-2,
    "Bubble\npercolation",
    ha="center",
    va="center",
    fontsize=20,
)

ax.text(
    2e5,
    5e-4,
    r"$m_\varphi < 2 m_h$",
    ha="center",
    va="center",
    fontsize=22,
)

ax.grid(False)

savefile = Path("./figures/m_over_H_contours.pdf")
save_figure(fig, savefile)
