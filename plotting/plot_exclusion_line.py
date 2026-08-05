import pickle
from plot_data import save_figure
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

ax.fill_between(m_fill, y_min, g_fill, alpha=0.45)
ax.fill_between(m_fill, g_fill, y_max, color="0.7")
m = np.logspace(np.log10(x_min), np.log10(x_max), 500)
m_h = 125.1
g_higgs = 4 * np.pi * m_h / (np.sqrt(6) * m)

mask = g_higgs > y_min
ax.fill_between(m[mask], y_min, g_higgs[mask], color="0.7")
ax.plot(m[mask], g_higgs[mask], color="black")

ax.plot(m_zprime, g_bl, linewidth=2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel(r"$m_{Z'}\,[\mathrm{GeV}]$")
ax.set_ylabel(r"$g_{B-L}$")

ax.text(1.5e5, 7e-2, "Bubble\npercolation", ha="center", va="center", fontsize=22)
ax.text(3e6, 6e-3, "Classical rolling", ha="center", va="center", fontsize=24)
ax.text(2e5, 5e-4, r"$m_\varphi < 2 m_h$", ha="center", va="center", fontsize=24)

ax.grid(False)

savefile = Path("./figures/fopt_exclusion_line.pdf")
save_figure(fig, savefile)
