import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from load_data import load_gw_spectra
from plot_data import save_figure
from pathlib import Path


def omega_peaks_best_fit(m_over_H):
    c = 0.834
    s = -1.95
    return c * m_over_H**s


def omega_peaks_best_fit_params(m_over_Hs, peaks):
    params, covariance = np.polyfit(np.log10(m_over_Hs), np.log10(peaks), 1, cov=True)
    s, n = params
    s_err, n_err = np.sqrt(np.diag(covariance))
    c = 10**n
    c_err = c * np.log(10) * n_err
    return c, s, c_err, s_err


def main():
    base_dirs = [
        "mH1e2_512",
        "mH1e3_512",
        "mH1e4_512",
        "mH1e5_512",
        "mH1e6_512",
        "mH1e7_512",
        "mH1e8_512",
        "mH1e9_512",
    ]
    input_dirs = [Path("../output") / d for d in base_dirs]
    output_dirs = [Path("./figures") / d for d in base_dirs]

    m_over_Hs = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])

    sims = [
        Simulation(inp, out, m_over_H)
        for inp, out, m_over_H in zip(input_dirs, output_dirs, m_over_Hs)
    ]

    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)

    s, c, s_err, c_err = omega_peaks_best_fit_params(m_over_Hs, peaks)
    print(f"Best-Fit: h^2 Omega_GW = c * (m/H)^s")
    print(f"s = {s:.3f} +/- {s_err:.3f}")
    print(f"c = {c:.2f} +/- {c_err:.2f}")
    # peaks_fit = c * m_over_Hs**s
    peaks_fit = omega_peaks_best_fit(m_over_Hs)
    fig, ax = plt.subplots()
    ax.plot(m_over_Hs, peaks, linestyle="", marker=".", markersize=8)
    ax.plot(m_over_Hs, peaks_fit, linestyle="-")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$m/H$")
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    save_figure(fig, Path("./figures/peaks_mH.pdf"))


if __name__ == "__main__":
    main()
    print("\nAll done!")
