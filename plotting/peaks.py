from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation
from load_data import load_gw_spectra, load_scale_factor
from plot_data import save_figure


def omega_peaks_best_fit_params(m_over_Hs, peaks, peak_errs):
    """Weighted fit of Omega_GW^peak = c * (mu/H)^s."""

    log_x = np.log10(m_over_Hs)
    log_y = np.log10(peaks)

    # Propagate Omega uncertainties into log10(Omega).
    log_y_err = peak_errs / (peaks * np.log(10))

    params, covariance = np.polyfit(
        log_x,
        log_y,
        1,
        w=1.0 / log_y_err,
        cov=True,
    )

    s, log10_c = params
    s_err, log10_c_err = np.sqrt(np.diag(covariance))

    c = 10**log10_c
    c_err = c * np.log(10) * log10_c_err

    return c, s, c_err, s_err


def main():
    m_over_Hs = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])

    base_dirs = [f"mH1e{int(np.log10(x))}_512_new" for x in m_over_Hs]

    sims = [
        Simulation(
            Path("../output") / base_dir,
            Path("./figures") / base_dir,
            m_over_H,
        )
        for base_dir, m_over_H in zip(base_dirs, m_over_Hs)
    ]

    omega_peaks = np.empty(len(sims))

    kappa_peaks = np.empty(len(sims))
    kappa_peak_errs = np.empty(len(sims))

    # k range used when locating the peak within each spectrum.
    k_min = 1
    k_max = 150

    # Time/spectra range over which kappa_peak is averaged.
    spec_min = 120
    spec_max = 150

    # Constant uncertainty due to k-space binning.
    binning_err = 0.05

    for i, sim in enumerate(sims):

        _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")

        # --------------------------------------------------------------
        # Overall maximum Omega_GW
        # --------------------------------------------------------------

        max_omega = -np.inf

        for spec in spectra:
            omega = spec["omega_gw"][k_min:k_max]
            max_omega = max(max_omega, np.max(omega))

        omega_peaks[i] = max_omega

        # --------------------------------------------------------------
        # Average peak kappa between spectra indices 80 and 149
        # --------------------------------------------------------------

        kappa_peaks_at_each_time = []

        for idx_spec in range(spec_min, spec_max):
            spec = spectra[idx_spec]

            omega = spec["omega_gw"][k_min:k_max]

            # Index of maximum Omega_GW within this spectrum.
            idx_local = np.argmax(omega)
            idx = k_min + idx_local

            # Peak kappa normalized by the scale factor at this time.
            kappa_peak = spec["kappa"][idx] / a[idx_spec]

            kappa_peaks_at_each_time.append(kappa_peak)

        kappa_peaks_at_each_time = np.asarray(kappa_peaks_at_each_time)

        # Mean peak position.
        kappa_peaks[i] = np.mean(kappa_peaks_at_each_time)

        # Statistical spread over the selected time range.
        std_err = np.std(kappa_peaks_at_each_time)

        # Combine standard deviation and binning error in quadrature.
        kappa_peak_errs[i] = np.sqrt(std_err**2 + binning_err**2)

        print(
            f"mu/H = {sim.m_over_H:.0e}: "
            f"Omega_peak = {omega_peaks[i]:.3e}, "
            f"kappa_peak = {kappa_peaks[i]:.3e} +/- {kappa_peak_errs[i]:.3e}, "
            f"std = {std_err:.3e}"
        )

    # ------------------------------------------------------------------
    # Peak amplitude fit
    # ------------------------------------------------------------------

    # Assume a 10% relative uncertainty on every Omega_GW peak.
    omega_peak_errs = 0.10 * omega_peaks

    c, s, c_err, s_err = omega_peaks_best_fit_params(
        m_over_Hs,
        omega_peaks,
        omega_peak_errs,
    )

    print("\nPeak amplitude fit:")
    print("Omega_GW^peak = c * (mu/H)^s")
    print(f"s = {s:.3f} +/- {s_err:.3f}")
    print(f"c = {c:.3f} +/- {c_err:.3f}")

    omega_fit = c * m_over_Hs**s
    chi2 = np.sum(((omega_peaks - omega_fit) / omega_peak_errs) ** 2)

    dof = len(omega_peaks) - 2  # c and s are fitted
    chi2_red = chi2 / dof

    print(f"chi2 / dof = {chi2:.2f} / {dof}")
    print(f"reduced chi2 = {chi2_red:.2f}")

    fig, ax = plt.subplots()

    ax.errorbar(
        m_over_Hs,
        omega_peaks,
        yerr=omega_peak_errs,
        linestyle="",
        marker=".",
        markersize=5,
        capsize=7,
        label="Simulation",
    )
    ax.plot(
        m_over_Hs,
        omega_fit,
        linestyle="-",
        label=r"Best fit:$h^2 \Omega_\mathrm{GW}^\mathrm{peak} = c (\mu/H)^s$",
    )
    # ax.plot(
    #     m_over_Hs,
    #     c * m_over_Hs ** (-2),
    #     linestyle="--",
    #     label=r"$(\mu/H)^{-2}$",
    # )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mu/H$")
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.legend(frameon=False)

    save_figure(fig, Path("./figures/peaks_OmegaGW.pdf"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Peak frequency / wavenumber
    # ------------------------------------------------------------------

    weights = 1.0 / kappa_peak_errs**2

    kappa_fit = np.sum(weights * kappa_peaks) / np.sum(weights)
    kappa_fit_err = np.sqrt(1.0 / np.sum(weights))

    print(f"\nConstant fit: {kappa_fit:.4f} +/- {kappa_fit_err:.4f}")

    chi2 = np.sum(((kappa_peaks - kappa_fit) / kappa_peak_errs) ** 2)
    dof = len(kappa_peaks) - 1
    chi2_red = chi2 / dof

    print(f"chi2 / dof = {chi2:.2f} / {dof}")
    print(f"reduced chi2 = {chi2_red:.2f}")
    fig, ax = plt.subplots()

    ax.errorbar(
        m_over_Hs,
        kappa_peaks,
        yerr=kappa_peak_errs,
        linestyle="",
        marker=".",
        markersize=8,
        capsize=3,
    )
    ax.axhline(
        kappa_fit,
        linestyle="-",
        label=rf"Constant fit: ${kappa_fit:.3f} \pm {kappa_fit_err:.3f}$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\mu/H$")
    ax.set_ylabel(
        r"$\langle k^{\mathrm{peak}}_{\mathrm{phys}}/\mu\rangle_{\tilde \eta\in [\tilde\eta_\mathrm{sat}, \tilde \eta_\mathrm{end}}]$"
    )
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    save_figure(fig, Path("./figures/peaks_k.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("\nAll done!")
