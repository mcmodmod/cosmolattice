from Veff_Daniel import EffectivePotential
import numpy as np
import matplotlib.pyplot as plt
from plot_data import save_figure
from pathlib import Path


def main():
    vhqcd = 0.1
    gBL = 0.1
    mZp = 1.0e3

    veff = EffectivePotential(gBL, mZp, vhqcd)
    veff.interpolations()

    v_phi = mZp / (2.0 * gBL)
    T_c = veff.find_Tc()
    print(T_c)

    phis = np.linspace(1.0e-5, 1.25 * v_phi, 1600)
    x = phis / v_phi

    def potential(T):
        V0 = veff.Veff(1.0e-5, veff.vh_qcd, T)
        return np.array([veff.Veff(phi, veff.vh_qcd, T) - V0 for phi in phis])

    fig, ax = plt.subplots()

    ax.axhline(0.0, color="0.6", linewidth=0.8)
    # Plot all non-critical temperatures first.
    for ratio in [3.0, 1.25, 0.75]:
        ax.plot(
            x,
            potential(ratio * T_c),
            linestyle="--",
            label=rf"$T={ratio:g}\,T_c$",
        )

    # Plot the critical-temperature curve last so that it appears on top.
    ax.plot(
        x,
        potential(0),
        linestyle="--",
        label=rf"$T=0$",
    )
    ax.plot(
        x,
        potential(T_c),
        color="black",
        linewidth=3,
        label=r"$T=T_c$",
    )

    ax.set_xlim(0.0, 1.25)
    ax.set_ylim(-2.9e9, 4.0e9)
    ax.set_xlabel(r"$\varphi/v_\varphi$")
    ax.set_ylabel(r"$\Delta V_{\mathrm{eff}}(\varphi,T)[\mathrm{GeV}^4]$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    save_figure(fig, Path("figures/veff_T.pdf"))


if __name__ == "__main__":
    main()
