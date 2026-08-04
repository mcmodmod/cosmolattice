import numpy as np
import matplotlib.pyplot as plt
from plot_data import save_figure
from pathlib import Path

# Dimensionless illustrative parameters
phi = np.linspace(-2.2, 2.2, 600)
lam = 0.25

mass_squared_values = {
    r"$m_{\mathrm{eff}}^2>0$": 0.8,
    r"$m_{\mathrm{eff}}^2=0$": 0.0,
    r"$m_{\mathrm{eff}}^2<0$": -0.8,
}

line_styles = ["-", ":", "-"]


# ------------------------------------------------------------------
# Left panel: effective potential
# ------------------------------------------------------------------
fig, ax = plt.subplots()
for (label, mass_squared), line_style in zip(
    mass_squared_values.items(),
    line_styles,
):
    potential = 0.5 * mass_squared * phi**2 + 0.25 * lam * phi**4
    ax.plot(
        phi,
        potential,
        linestyle=line_style,
        linewidth=3,
        label=label,
    )

# ax.axvline(0.0, linewidth=0.8, alpha=0.4)
ax.axhline(0.0, linewidth=0.8, alpha=0.4)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-0.8, 1.25)
ax.set_xlabel(r"$\phi$", fontsize=25)
ax.set_ylabel(r"$V_{\mathrm{eff}}(\phi)$", fontsize=25)
ax.legend(frameon=False, fontsize=22)
save_figure(fig, Path("figures/curvature_at_origin.pdf"))

# ------------------------------------------------------------------
# Right panel: instability band
# ------------------------------------------------------------------
fig, ax = plt.subplots()

m_abs = 1
k_phys = np.linspace(0.0, 2.0 * m_abs, 500)
omega_squared = k_phys**2 - m_abs**2

x = k_phys / m_abs
y = omega_squared / m_abs**2

ax.plot(
    x,
    y,
    linewidth=3,
)

# Highlight momentum-space regions
ax.axvspan(
    0.0,
    1.0,
    alpha=0.18,
    label=r"unstable: $\omega_k^2<0$",
)

ax.axvspan(
    1.0,
    2.0,
    alpha=0.07,
    hatch="//",
    label=r"stable: $\omega_k^2>0$",
)

ax.axhline(0.0, linewidth=0.8, alpha=0.5)
ax.axvline(1.0, linestyle="--", linewidth=1.0)

ax.text(0.5, 2.30, "tachyonic", ha="center", va="center", fontsize=24)

ax.text(1.5, 2.30, "oscillatory", ha="center", va="center", fontsize=24)

# ax.annotate(
#     r"$k_{\mathrm{phys}}=\sqrt{-m_{\mathrm{eff}}^2}$",
#     xy=(1.0, 0.0),
#     xytext=(1.12, -0.65),
#     arrowprops={"arrowstyle": "-|>"},
#     fontsize=15,
# )

ax.set_xlim(0.0, 2.0)
ax.set_ylim(-1.08, 3.08)
ax.set_xlabel(r"$k_{\mathrm{phys}}/\sqrt{-m_{\mathrm{eff}}^2}$", fontsize=21)
ax.set_ylabel(r"$\omega_k^2/\left(-m_{\mathrm{eff}}^2\right)$", fontsize=21)
save_figure(fig, Path("figures/instability_band.pdf"))
