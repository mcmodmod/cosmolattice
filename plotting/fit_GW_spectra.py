import numpy as np
from scipy.special import expit
from scipy.optimize import least_squares
from scipy.special import log_expit
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from dataclasses import dataclass
from load_data import load_last_gw_spectrum, spectrum_peak, load_scale_factor
from plot_data import save_figure
from simulation import Simulation

M_PL = 2.435e18


@dataclass
class FitResult:
    params: dict
    errors: dict
    covariance: np.ndarray


# def spectrum_best_fit(
#     f, k_peak, omega_peak, sim=Simulation(Path("../output/mH1e3_512"), Path(), 1e3)
# ):
#     k_star, omega_gw_star = load_last_gw_spectrum(sim.input_dir / "spectra_gws.txt")
#     fit = fit_template_2(k_star, omega_gw_star)
#     return gw_template_2(f, **fit.params, k_peak=k_peak, omega_peak=omega_peak)


def gw_template_1(f, n1, n2, delta, k_peak, omega_peak):
    """From 2306.14856 eq. 2.25"""
    k_tilde = f / k_peak
    return (
        omega_peak
        * k_tilde**n1
        * (0.5 * (1 + k_tilde ** (1 / delta))) ** ((n2 - n1) * delta)
    )


def gw_template_2(f, As, fs, gamma, p, k_peak, omega_peak):
    z = f / (k_peak * fs)
    x = gamma * (z - 1)

    return omega_peak * As * expit(p * np.log(z) - x)


# def gw_template_2(f, As, fs, gamma, p, k_peak, omega_peak):
#     """From 1912.01007 eq. 7"""
#     k_tilde = f / k_peak
#     return omega_peak * (
#         As
#         * (k_tilde / fs) ** p
#         / (1 + (k_tilde / fs) ** p * np.exp(gamma * (k_tilde / fs - 1)))
#     )


def fit_template_1(k_star, omega_gw_star, k_max=1e35) -> FitResult:
    mask = k_star < k_max
    k_fit = k_star[mask]
    omega_fit = omega_gw_star[mask]

    k_peak, omega_peak = spectrum_peak(k_star, omega_gw_star)
    popt, pcov = curve_fit(
        lambda f, n1, n2, delta: gw_template_1(f, n1, n2, delta, k_peak, omega_peak),
        k_fit,
        omega_fit,
    )

    errors = np.sqrt(np.diag(pcov))
    names = ["n1", "n2", "delta"]

    return FitResult(
        params=dict(zip(names, popt)),
        errors=dict(zip(names, errors)),
        covariance=pcov,
    )


def fit_template_2(k_star, omega_gw_star, k_max=1e35) -> FitResult:
    mask = k_star < k_max
    k_fit = k_star[mask]
    omega_fit = omega_gw_star[mask]

    k_peak, omega_peak = spectrum_peak(k_star, omega_gw_star)
    popt, pcov = curve_fit(
        lambda f, As, gamma, p: gw_template_2(f, As, 1, gamma, p, k_peak, omega_peak),
        k_fit,
        omega_fit,
        p0=[1.8, 1.2, 1],
    )

    errors = np.sqrt(np.diag(pcov))
    names = ["As", "gamma", "p"]

    return FitResult(
        params=dict(zip(names, popt)),
        errors=dict(zip(names, errors)),
        covariance=pcov,
    )


# def fit_template_2(k_star, omega_gw_star, k_max=1e35) -> FitResult:
#     k_peak, omega_peak = spectrum_peak(k_star, omega_gw_star)
#
#     # We need strictly positive values for the logarithmic fit.
#     mask = (
#         (k_star < k_max)
#         & (k_star > 0)
#         & (omega_gw_star > 0)
#         & np.isfinite(k_star)
#         & np.isfinite(omega_gw_star)
#     )
#
#     k_fit = k_star[mask]
#     omega_fit = omega_gw_star[mask]
#
#     log_omega_fit = np.log(omega_fit)
#
#     def log_model(theta):
#         # Optimize logarithms so all parameters stay positive.
#         As, fs, gamma, p = np.exp(theta)
#
#         z = k_fit / (k_peak * fs)
#
#         # gw_template_2 is:
#         #
#         # omega_peak * As * expit(
#         #     p * log(z) - gamma * (z - 1)
#         # )
#         #
#         # Evaluate its logarithm directly to avoid numerical
#         # underflow when the spectrum becomes extremely small.
#         u = p * np.log(z) - gamma * (z - 1)
#
#         return np.log(omega_peak) + np.log(As) + log_expit(u)
#
#     def residuals(theta):
#         return log_model(theta) - log_omega_fit
#
#     # Same physical starting point as your original fit.
#     p0 = np.array([1, 1, 1, 1])
#     theta0 = np.log(p0)
#
#     result = least_squares(
#         residuals,
#         theta0,
#         method="trf",
#         x_scale="jac",
#         max_nfev=100_000,
#     )
#
#     # Convert back from log parameters.
#     popt = np.exp(result.x)
#
#     # Approximate covariance matrix.
#     #
#     # First calculate covariance in log-parameter space.
#     n_data = len(k_fit)
#     n_params = len(result.x)
#
#     if n_data > n_params:
#         _, s, VT = np.linalg.svd(result.jac, full_matrices=False)
#
#         threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
#         valid = s > threshold
#
#         cov_log = (VT[valid].T / s[valid] ** 2) @ VT[valid]
#
#         # Reduced chi^2-like scaling of the covariance.
#         residual_variance = 2 * result.cost / (n_data - n_params)
#         cov_log *= residual_variance
#
#         # Transform covariance from log(parameters) to parameters.
#         J_transform = np.diag(popt)
#         pcov = J_transform @ cov_log @ J_transform
#     else:
#         pcov = np.full((n_params, n_params), np.nan)
#
#     errors = np.sqrt(np.diag(pcov))
#     names = ["As", "fs", "gamma", "p"]
#
#     return FitResult(
#         params=dict(zip(names, popt)),
#         errors=dict(zip(names, errors)),
#         covariance=pcov,
#     )


def plot_results(
    k_star,
    omega_gw_star,
    fit2: FitResult,
    output_file: Path,
    k_max=10,
):
    fig, ax = plt.subplots()

    ax.plot(
        k_star,
        omega_gw_star,
        linestyle="",
        marker=".",
        color="lightgrey",
        label="Data",
    )

    mask = k_star < k_max
    ax.plot(
        k_star[mask],
        omega_gw_star[mask],
        linestyle="",
        marker=".",
        markersize=10,
        color="blue",
        label="Data used for fit",
    )

    ax.plot(
        k_star,
        gw_template_2(
            k_star, fs=1, **fit2.params, k_peak=k_peak, omega_peak=omega_peak
        ),
        linewidth=2,
        color="red",
        label="Fit template",
    )

    # ax.vlines(
    #     [k_peak],
    #     1e-19,
    #     omega_peak,
    #     linestyle="--",
    #     label=r"$(k_\mathrm{phys}/\mu)_\mathrm{peak}$",
    # )
    # ax.hlines(
    #     [omega_peak],
    #     0,
    #     10,
    #     linestyle="--",
    #     label=r"$(k_\mathrm{phys}/\mu)_\mathrm{peak}$",
    # )

    ax.set(xscale="log", yscale="log")
    # ax.set(xscale="log")
    ax.set(
        xlabel=r"$k_\mathrm{phys}/\mu$",
        ylabel=r"$h^2\Omega_{\mathrm{GW}}$",
    )
    ax.set_ylim(np.min(omega_gw_star), np.max(omega_gw_star) * 5)

    ax.legend(frameon=False)
    save_figure(fig, output_file)


def plot_residuals(k_star, omega_gw_star, fit, output_file, k_max):
    fig, ax = plt.subplots()

    mask = k_star < k_max
    best_fit = gw_template_2(
        k_star[mask], fs=1, **fit2.params, k_peak=k_peak, omega_peak=omega_peak
    )
    ax.plot(
        k_star[mask],
        np.abs(omega_gw_star[mask] - best_fit) / best_fit,
        linestyle="",
        marker=".",
        label="Data used for fit",
    )

    ax.set(xscale="log", yscale="log")
    # ax.set(
    #     xlabel=r"$k_\mathrm{phys}/\mu$",
    #     ylabel=r"$h^2\Omega_{\mathrm{GW}}$",
    # )
    # ax.set_ylim(1e-10, 1e-5)

    # ax.legend(frameon=False)
    save_figure(fig, output_file)


if __name__ == "__main__":
    base_dirs = [
        Path("mH1e3_512_new"),
        Path("mH1e4_512_new"),
        Path("mH1e5_512_new"),
        Path("mH1e6_512_new"),
        Path("mH1e7_512_new"),
        Path("mH1e8_512_new"),
    ]
    m_over_Hs = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    sims = [
        Simulation(Path("../output") / base_dir, Path("./figures") / base_dir, m_over_H)
        for base_dir, m_over_H in zip(base_dirs, m_over_Hs)
    ]
    for sim in sims:
        output_file = sim.output_dir / "gw_spectrum_fit.pdf"
        _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")

        k_max = 4

        k_star, omega_gw_star = load_last_gw_spectrum(sim.input_dir / "spectra_gws.txt")
        k_star = k_star / a[-1]

        k_peak, omega_peak = spectrum_peak(k_star, omega_gw_star)
        fit2 = fit_template_2(k_star[1:], omega_gw_star[1:], k_max)
        plot_results(k_star, omega_gw_star, fit2, output_file, k_max)
        plot_residuals(
            k_star, omega_gw_star, fit2, sim.output_dir / "fit_residuals_gw.pdf", k_max
        )

        print("Peak values:")
        print(f"k_peak = {k_peak:.3e}")
        print(f"omega_peak = {omega_peak:.3e}")

        print("\nFit template 2:")
        for name in fit2.params:
            print(f"{name} = {fit2.params[name]:.2f} ± {fit2.errors[name]:.2f}")
