import numpy as np
from scipy.special import expit
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from dataclasses import dataclass
from load_data import load_last_gw_spectrum_f, spectrum_peak
from plot_data import save_figure
from simulation import Simulation

M_PL = 2.435e18


@dataclass
class FitResult:
    params: dict
    errors: dict
    covariance: np.ndarray


def spectrum_best_fit(
    f, f_peak, omega_peak, sim=Simulation(Path("../output/mH1e3_512"), Path(), 1e3)
):
    f_star, omega_gw_star = load_last_gw_spectrum_f(
        sim.input_dir / "spectra_gws.txt", sim.mu
    )
    fit = fit_template_2(f_star, omega_gw_star)
    return gw_template_2(f, **fit.params, f_peak=f_peak, omega_peak=omega_peak)


def gw_template_1(f, n1, n2, delta, f_peak, omega_peak):
    """From 2306.14856 eq. 2.25"""
    f_tilde = f / f_peak
    return (
        omega_peak
        * f_tilde**n1
        * (0.5 * (1 + f_tilde ** (1 / delta))) ** ((n2 - n1) * delta)
    )


def gw_template_2(f, As, fs, gamma, p, f_peak, omega_peak):
    f_tilde = f / f_peak
    y = (f_tilde / fs) ** p
    x = gamma * (f_tilde / fs - 1)

    return omega_peak * As * expit(np.log(y) - x)


# def gw_template_2(f, As, fs, gamma, p, f_peak, omega_peak):
#     """From 1912.01007 eq. 7"""
#     f_tilde = f / f_peak
#     return omega_peak * (
#         As
#         * (f_tilde / fs) ** p
#         / (1 + (f_tilde / fs) ** p * np.exp(gamma * (f_tilde / fs - 1)))
#     )


def fit_template_1(f_star, omega_gw_star, f_max=1e35) -> FitResult:
    mask = f_star < f_max
    f_fit = f_star[mask]
    omega_fit = omega_gw_star[mask]

    f_peak, omega_peak = spectrum_peak(f_star, omega_gw_star)
    popt, pcov = curve_fit(
        lambda f, n1, n2, delta: gw_template_1(f, n1, n2, delta, f_peak, omega_peak),
        f_fit,
        omega_fit,
    )

    errors = np.sqrt(np.diag(pcov))
    names = ["n1", "n2", "delta"]

    return FitResult(
        params=dict(zip(names, popt)),
        errors=dict(zip(names, errors)),
        covariance=pcov,
    )


def fit_template_2(f_star, omega_gw_star, f_max=1e35) -> FitResult:
    mask = f_star < f_max
    f_fit = f_star[mask]
    omega_fit = omega_gw_star[mask]

    f_peak, omega_peak = spectrum_peak(f_star, omega_gw_star)
    popt, pcov = curve_fit(
        lambda f, As, fs, gamma, p: gw_template_2(
            f, As, fs, gamma, p, f_peak, omega_peak
        ),
        f_fit,
        omega_fit,
    )

    errors = np.sqrt(np.diag(pcov))
    names = ["As", "fs", "gamma", "p"]

    return FitResult(
        params=dict(zip(names, popt)),
        errors=dict(zip(names, errors)),
        covariance=pcov,
    )


def plot_results(
    f_star, omega_gw_star, fit1: FitResult, fit2: FitResult, output_file: Path
):
    fig, ax = plt.subplots()

    ax.plot(
        f_star,
        omega_gw_star,
        linestyle="",
        marker=".",
        color="grey",
        label="Data",
    )

    ax.plot(
        f_star,
        gw_template_1(f_star, **fit1.params, f_peak=f_peak, omega_peak=omega_peak),
        linewidth=2,
        color="k",
        label="Fit template 1",
    )

    ax.plot(
        f_star,
        gw_template_2(f_star, **fit2.params, f_peak=f_peak, omega_peak=omega_peak),
        linewidth=2,
        color="red",
        label="Fit template 2",
    )

    ax.vlines(
        [f_peak],
        np.min(omega_gw_star),
        omega_peak,
        linestyle="--",
        label=r"$f_\star^\mathrm{peak}$",
    )

    ax.set(xscale="log", yscale="log")
    ax.set(
        xlabel=r"$f_\star [\mathrm{Hz}]$",
        ylabel=r"$h^2\Omega_{\mathrm{GW}, \star}$",
    )

    ax.legend()
    save_figure(fig, output_file)


if __name__ == "__main__":
    base_dir = Path("mH1e3_512")
    input_file = Path("../output") / base_dir / "spectra_gws.txt"
    output_file = Path("./figures") / base_dir / "gw_spectrum_fit.pdf"

    mu = M_PL * 1e-7

    f_star, omega_gw_star = load_last_gw_spectrum_f(input_file, mu)
    f_peak, omega_peak = spectrum_peak(f_star, omega_gw_star)
    fit1 = fit_template_1(f_star, omega_gw_star)
    fit2 = fit_template_2(f_star, omega_gw_star)
    plot_results(f_star, omega_gw_star, fit1, fit2, output_file)

    print("Peak values:")
    print(f"f_peak = {f_peak:.3e}")
    print(f"omega_peak = {omega_peak:.3e}")

    print("\nFit template 1:")
    for name in fit1.params:
        print(f"{name} = {fit1.params[name]:.3f} ± {fit1.errors[name]:.3f}")

    print("\nFit template 2:")
    for name in fit2.params:
        print(f"{name} = {fit2.params[name]:.3f} ± {fit2.errors[name]:.3f}")
