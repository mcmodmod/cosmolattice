import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
from Veff_Daniel import EffectivePotential
from gw_peaks import omega_peaks_best_fit
from matplotlib.ticker import NullLocator
from fit_GW_spectra import spectrum_best_fit
from load_data import frequency_from_kappa

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


def plot_compare_sens_curves():
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
    fig.set_size_inches(8, 5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for file in filenames:
        ax.plot(freq[file], sens[file], linewidth=1)
        if len(sens[file]) > 1000:
            step = 8
        else:
            step = 1
        ax.fill_between(freq[file][::step], sens[file][::step], 1e-6, alpha=0.075)
    plot_labels(ax)

    ax.set_xlabel(r"$f_0\,\mathrm{[Hz]}$")
    ax.set_ylabel(r"$h^2 \Omega_{\mathrm{GW}, 0}$")
    ax.set_ylim(1e-32, 1e-6)
    ax.set_xlim(1e-10, 7e2)
    ax.set_xticks([10**i for i in range(-10, 3, 2)])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(linestyle="--", alpha=0.4)
    return fig, ax


def get_peaks_from_params(gBL, mZp):
    g_star = 106.75
    a_star = 1

    veff = EffectivePotential(gBL, mZp, vh_qcd=0.1)
    veff.interpolations()
    m_over_H = veff.m_over_H()
    T_rh = veff.find_T_vac()
    H_rh = veff.Hubble(T_rh)

    omega_peak = omega_peaks_best_fit(m_over_H)
    omega_peak_0 = 1.67e-5 * (100 / g_star) ** (1 / 3) * omega_peak
    k_peak = np.sqrt(abs(veff.d2Veff(phi=0, h=veff.vh_qcd, T=0)))
    f_peak_0 = 1.65e-7 * 1 / (a_star * H_rh) * T_rh * (g_star / 100) ** (1 / 6) * k_peak

    print(f"{m_over_H=:.1E}")
    return f_peak_0, omega_peak_0


def plot_labels(ax):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.text(7e-10, 5e-12, r"$\mathrm{SKA}$", c=colors[0], fontsize=17, rotation=90)
    ax.text(5e-5, 1e-10, r"$\mathrm{LISA}$", c=colors[1], fontsize=17, rotation=295)
    ax.text(
        1.3e-7, 5e-12, r"$\mu\mathrm{ARES}$", c=colors[2], fontsize=17, rotation=310
    )
    # ax.text(3e1, 1e-10, r"$\mathrm{DECIGO}$", c=colors[3], fontsize=17, rotation=70)
    ax.text(2e2, 1e-10, r"$\mathrm{ET}$", c=colors[4], fontsize=17, rotation=60)
    # ax.text(2.5e0, 8e-11, r"$\mathrm{B-DECIGO}$", c=colors[5], fontsize=17, rotation=62)
    # ax.text(8e0, 4e-11, r"$\mathrm{BBO}$", c=colors[6], fontsize=17, rotation=65)


if __name__ == "__main__":
    M_PL = 2.435e18

    gBLs = np.array([1e-2])
    mZps = np.array([1e6, 8e5, 5e5, 2.5e5])
    labels = [
        r"$m_{\rm Z'} = 1 \times 10^{6}$",
        r"$m_{\rm Z'} = 8 \times 10^{5}$",
        r"$m_{\rm Z'} = 5 \times 10^{5}$",
        r"$m_{\rm Z'} = 2.5 \times 10^{5}$",
    ]
    tamara_filenames = np.array(
        [
            "h2OmegaGW_SI_gBL=1.00e-02_mZp=1.00e+06.txt",
            "h2OmegaGW_SI_gBL=1.00e-02_mZp=8.00e+05.txt",
            "h2OmegaGW_SI_gBL=1.00e-02_mZp=5.00e+05.txt",
            "h2OmegaGW_SI_gBL=1.00e-02_mZp=2.50e+05.txt",
        ]
    )
    tamara_data = [
        np.loadtxt(
            "./spectra_tamara/" + file,
            skiprows=1,
            unpack=True,
        )
        for file in tamara_filenames
    ]

    fig, ax = plot_compare_sens_curves()
    f_range = np.logspace(-10, 2, 1000)

    # Set colormap for GW curves
    cmap = mpl.colormaps["Blues"]
    colors = [cmap(x) for x in np.linspace(0.5, 1.0, len(mZps) * len(gBLs))]

    ax.set_prop_cycle(cycler(color=colors))
    for gBL in gBLs:
        for i, mZp in enumerate(mZps):
            f_peak_0, omega_peak_0 = get_peaks_from_params(gBL, mZp)
            print(f"{f_peak_0=:.2E}, {omega_peak_0=:.2E}")

            mask = f_range > f_peak_0 * (10 ** (-2))
            ax.plot(
                f_range[mask],
                spectrum_best_fit(f_range, f_peak_0, omega_peak_0)[mask],
                linewidth=2,
                label=labels[i],
            )
    for file in tamara_data:
        _, f, omega_gw = file
        ax.plot(f, omega_gw)
    ax.legend()
    fig.savefig("figures/sens_curves.pdf", format="pdf", backend="pgf")
