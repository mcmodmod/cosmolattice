import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import lambertw
from pathlib import Path
from dataclasses import dataclass

from plot_data import save_figure
from load_data import load_single_gw_spectrum, spectrum_peak, load_scale_factor
from simulation import Simulation

M_PL = 2.435e18


@dataclass
class FitResult:
    params: dict
    errors: dict
    covariance: np.ndarray


def x_max(gamma, p):
    arg = gamma / (p + 1.0) * ((p / gamma) * np.exp(gamma)) ** (1.0 / (p + 1.0))

    return (p + 1.0) / gamma * lambertw(arg).real


def S_max(gamma, p):
    xm = x_max(gamma, p)

    return xm**p / (1.0 + xm**p * np.exp(gamma * (xm - 1.0)))


# def gw_template(kappa, kappa_peak, gamma, p):
#     """
#     Parameterization in terms of the actual peak position.
#     """
#
#     xm = x_max(gamma, p)
#
#     # Convert peak position to kappa_s
#     kappa_s = kappa_peak / xm
#
#     x = kappa / kappa_s
#
#     S = x**p / (1.0 + x**p * np.exp(gamma * (x - 1.0)))
#
#     return S / S_max(gamma, p)
def log_S(x, gamma, p):
    log_xp = p * np.log(x)
    return log_xp - np.logaddexp(
        0.0,
        log_xp + gamma * (x - 1.0),
    )


def gw_template(kappa, kappa_peak, gamma, p):
    xm = x_max(gamma, p)
    kappa_s = kappa_peak / xm

    x = kappa / kappa_s

    log_S_value = log_S(x, gamma, p)
    log_S_max = log_S(xm, gamma, p)

    return np.exp(log_S_value - log_S_max)


def fit_template(kappa, omega_gw_star, kappa_max) -> FitResult:

    measured_kappa_peak, omega_peak = spectrum_peak(kappa, omega_gw_star)

    omega_norm = omega_gw_star / omega_peak

    mask = (kappa > 0) & (kappa < kappa_max) & (omega_norm > 0)

    kappa_fit = kappa[mask]
    omega_fit = omega_norm[mask]

    p0 = [
        measured_kappa_peak,
        1.2,
        1.0,
    ]

    bounds = (
        [1e-10, 1e-6, 1e-6],
        [np.inf, np.inf, np.inf],
    )

    def log_template(kappa, kappa_peak, gamma, p):
        return np.log(
            gw_template(
                kappa,
                kappa_peak,
                gamma,
                p,
            )
        )

    popt, pcov = curve_fit(
        log_template,
        kappa_fit,
        np.log(omega_fit),
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )

    errors = np.sqrt(np.diag(pcov))
    names = ["kappa_peak", "gamma", "p"]

    return FitResult(
        params=dict(zip(names, popt)),
        errors=dict(zip(names, errors)),
        covariance=pcov,
    )


def plot_results(
    kappa,
    omega_gw_star,
    fit: FitResult,
    output_file: Path,
    kappa_max=10.0,
):
    fig, ax = plt.subplots()

    # Normalize data by its measured peak
    _, omega_peak = spectrum_peak(kappa, omega_gw_star)
    omega_norm = omega_gw_star / omega_peak

    # ax.plot(
    #     kappa,
    #     omega_norm,
    #     linestyle="",
    #     marker=".",
    #     color="lightgrey",
    #     label="Data",
    # )

    # Data points used in the fit
    mask = (kappa > 0) & (kappa < kappa_max)

    ax.plot(
        kappa[mask],
        omega_norm[mask],
        linestyle="",
        marker=".",
        markersize=10,
        color="blue",
        label="Data used for fit",
    )

    # Best-fit template
    ax.plot(
        kappa[mask],
        gw_template(kappa[mask], **fit.params),
        linewidth=2,
        color="red",
        label="Fit template",
    )

    ax.set(
        # xscale="log",
        # yscale="log",
        xlabel=r"$k_\mathrm{phys}/\mu$",
        ylabel=r"$\Omega_{\mathrm{GW}}/\Omega_{\mathrm{GW}}^{\mathrm{peak}}$",
    )

    # ax.set_ylim(
    #     np.min(omega_norm[omega_norm > 0]),
    #     2.0,
    # )

    ax.legend(frameon=False)

    save_figure(fig, output_file)


def plot_residuals(
    kappa,
    omega_gw_star,
    fit: FitResult,
    output_file: Path,
    kappa_max=10.0,
):
    fig, ax = plt.subplots()

    # Normalize data in exactly the same way as for the fit
    _, omega_peak = spectrum_peak(kappa, omega_gw_star)
    omega_norm = omega_gw_star / omega_peak

    mask = (kappa > 0) & (kappa < kappa_max)

    best_fit = gw_template(
        kappa[mask],
        **fit.params,
    )

    residuals = np.abs(omega_norm[mask] - best_fit) / best_fit

    ax.plot(
        kappa[mask],
        residuals,
        linestyle="",
        marker=".",
        label="Relative residual",
    )

    ax.set(
        # xscale="log",
        # yscale="log",
        xlabel=r"$\kappa$",
        ylabel=r"$|\mathrm{data}-\mathrm{fit}|/\mathrm{fit}$",
    )

    ax.legend(frameon=False)

    save_figure(fig, output_file)


if __name__ == "__main__":
    base_dirs = [
        Path("mH1e3_512_new"),
        Path("mH1e4_512_new"),
        Path("mH1e5_512_new"),
        Path("mH1e6_512_new"),
        Path("mH1e7_512_new"),
        Path("mH1e8_512_new"),
        Path("mH1e9_512_new"),
    ]

    m_over_Hs = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

    sims = [
        Simulation(
            Path("../output") / base_dir,
            Path("./figures") / base_dir,
            m_over_H,
        )
        for base_dir, m_over_H in zip(base_dirs, m_over_Hs)
    ]

    k_peaks, gammas, ps = np.empty(len(sims)), np.empty(len(sims)), np.empty(len(sims))
    k_peaks_err, gammas_err, ps_err = (
        np.empty(len(sims)),
        np.empty(len(sims)),
        np.empty(len(sims)),
    )
    eta = -1
    for i, sim in enumerate(sims):
        _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")

        k_star, omega_gw_star = load_single_gw_spectrum(
            sim.input_dir / "spectra_gws.txt", eta
        )

        # Physical/rescaled momentum used for the fit.
        kappa = k_star / a[eta]

        # Peak of the original spectrum.
        kappa_peak, omega_peak = spectrum_peak(
            kappa,
            omega_gw_star,
        )

        # Set the upper fitting limit in kappa.
        kappa_max = 4.0

        fit = fit_template(
            kappa[1:],
            omega_gw_star[1:],
            kappa_max,
        )
        k_peaks[i] = fit.params["kappa_peak"]
        k_peaks_err[i] = fit.errors["kappa_peak"]
        gammas[i] = fit.params["gamma"]
        gammas_err[i] = fit.errors["gamma"]
        ps[i] = fit.params["p"]
        ps_err[i] = fit.errors["p"]

        std = np.sqrt(np.diag(fit.covariance))
        corr = fit.covariance / np.outer(std, std)

        # print("\nCorrelation matrix:")
        # print(corr)

        plot_results(
            kappa[1:],
            omega_gw_star[1:],
            fit,
            sim.output_dir / "gw_spectrum_fit.pdf",
            kappa_max=kappa_max,
        )

        plot_residuals(
            kappa[1:],
            omega_gw_star[1:],
            fit,
            sim.output_dir / "gw_spectrum_fit_residuals.pdf",
            kappa_max=kappa_max,
        )
        print("\nPeak values:")
        print(f"kappa_peak = {kappa_peak:.3e}")
        print(f"omega_peak = {omega_peak:.3e}")

        print("\nFit template:")
        for name in fit.params:
            print(f"{name} = " f"{fit.params[name]:.4g} ± " f"{fit.errors[name]:.2g}")

    fig, ax = plt.subplots()
    # ax.errorbar(m_over_Hs, k_peaks, yerr=k_peaks_err, linestyle="", marker=".")
    # ax.set_xscale("log")
    ax.errorbar(m_over_Hs, gammas, yerr=gammas_err, linestyle="", marker=".")
    ax.set_xscale("log")
    # ax.errorbar(m_over_Hs, ps, yerr=ps_err, linestyle="", marker=".")
    # ax.set_xscale("log")
    save_figure(fig, Path("./figures/test.pdf"))

    print(
        f"{np.mean(k_peaks)=:.2f}, {np.std(k_peaks)=:.2f}, {np.std(k_peaks) / np.mean(k_peaks)=:.2f}"
    )
    print(
        f"{np.mean(gammas)=:.2f}, {np.std(gammas)=:.2f}, {np.std(gammas) / np.mean(gammas)=:.2f}"
    )
    print(f"{np.mean(ps)=:.2f}, {np.std(ps)=:.2f}, {np.std(ps) / np.mean(ps)=:.2f}")
