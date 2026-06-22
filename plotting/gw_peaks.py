import numpy as np
import matplotlib.pyplot as plt
from plot_data import load_spectrum
from pathlib import Path

plt.rcParams.update(
    {
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "axes.labelsize": 20,
        "axes.grid": True,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.constrained_layout.use": True,
    }
)


def main():
    base_dirs = [
        "mH1e2/",
        "mH1e3/mu11_lam-10/",
        "mH1e4_old/",
        "mH1e5/",
        "mH1e6/",
        "mH1e7/",
        "mH1e8/",
        "mH1e9/",
    ]
    input_dirs = [Path("../output") / d for d in base_dirs]

    peaks = np.empty(len(base_dirs))
    for i, d_in in enumerate(input_dirs):
        spectra, _, _ = load_spectrum(d_in / "spectra_gws.txt")
        peaks[i] = np.max(spectra[:, 1])
    log_peaks = np.log10(peaks)
    log_mHs = np.array([i for i in range(2, len(base_dirs) + 2)])
    mHs = 10**log_mHs

    params, covariance = np.polyfit(log_mHs, log_peaks, 1, cov=True)
    s, n = params
    s_err, n_err = np.sqrt(np.diag(covariance))
    c = 10**n
    c_err = c * np.log(10) * n_err
    print(f"Best-Fit: Omega_GW = c * (m/H)^s")
    print(f"s = {s:.3f} +/- {s_err:.3f}")
    print(f"c = {c:.2f} +/- {c_err:.2f}")
    peaks_fit = c * mHs**s
    fig, ax = plt.subplots()
    ax.plot(mHs, peaks, linestyle="", marker=".", markersize=8)
    ax.plot(mHs, peaks_fit, linestyle="-")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$m/H$")
    fig.savefig("./figures/peaks_mH.pdf", backend="pgf")


if __name__ == "__main__":
    main()
    print("\nAll done!")
