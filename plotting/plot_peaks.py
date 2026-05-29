import numpy as np
import matplotlib.pyplot as plt
from plot_data import load_spectrum
from pathlib import Path


def power_law(x, a, b):
    return a * x ** (-b)


def main():
    base_dirs = [
        "mH1e2/",
        "mH1e3/mu11_lam-10/",
        "mH1e4/",
        "mH1e5/",
        "mH1e6/",
        "mH1e7/",
        "mH1e8/",
        "mH1e9/",
    ]
    input_dirs = [Path("../output") / d for d in base_dirs]

    log_peaks = np.empty(len(base_dirs))
    for i, d_in in enumerate(input_dirs):
        spectra, _, _ = load_spectrum(d_in / "spectra_gws.txt")
        log_peaks[i] = np.log10(np.max(spectra[:, 1]))
    log_mHs = np.array([i for i in range(2, len(base_dirs) + 2)])

    params, covariance = np.polyfit(log_mHs, log_peaks, 1, cov=True)
    m, b = params
    m_err, b_err = np.sqrt(np.diag(covariance))
    print(f"m = {m:.3f} +/- {m_err:.3f}")
    print(f"b = {b:.2f} +/- {b_err:.2f}")
    a = 10**b
    p = m
    print(f"a = {a:.3f} +/- {0}")
    print(f"p = {p:.2f} +/- {b_err:.2f}")
    log_peaks_fit = m * log_mHs + b
    fig, ax = plt.subplots()
    ax.plot(log_mHs, log_peaks, linestyle="", marker=".", markersize=10)
    ax.plot(log_mHs, log_peaks_fit, linestyle="-")
    ax.set_ylabel(r"$\log_{10} \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$\log_{10} m/H$")
    # fig.savefig("./figures/peaks_mH.pdf")


if __name__ == "__main__":
    main()
    print("\nAll done!")
