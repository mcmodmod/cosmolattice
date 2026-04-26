import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path

plt.rcParams.update(
    {
        "text.usetex": True,
        "axes.labelsize": 20,
        "axes.grid": True,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.constrained_layout.use": True,
    }
)
M_PL = 2.435 * 10 ** (18)
SAVE = False


def save_figure(fig, path: Path):
    if SAVE:
        fig.savefig(path, format="pdf")
    plt.close(fig)


def mu_over_sqrtlam_from_mH(mH):
    """Takes a ratio m/H at the origin and returns the ratio mu/sqrt(lambda)"""
    return np.sqrt(12) * M_PL / mH


def plot_average_field(save_dir, phiDat, sl=slice(None)):

    eta = phiDat[sl, 0]
    phi = phiDat[sl, 1]

    fig, ax = plt.subplots()
    ax.plot(eta, phi)
    ax.set_xlabel(r"$\tilde\eta$")
    ax.set_ylabel(r"$\langle\tilde\phi\rangle$")

    save_figure(fig, save_dir / "average_field.pdf")


def plot_energy_densities(save_dir, enDat, sl=slice(None)):
    eta = enDat[sl, 0]

    labels = [
        r"$\tilde E_K$",
        r"$\tilde E_G$",
        r"$\tilde E_V$",
        r"$\langle \tilde \rho \rangle$",
    ]

    fig, ax = plt.subplots()
    for i, label in enumerate(labels, start=1):
        ax.plot(eta, enDat[sl, i], label=label)

    ax.set(
        xlabel=r"$\tilde \eta$",
        ylabel=r"$\tilde E$",
        yscale="log",
    )
    ax.legend()

    save_figure(fig, save_dir / "energies.pdf")


def load_spectrum(filename, comment="#"):
    with open(filename) as f:
        # open the file and count how many bins in one spectrum
        n_bins = sum(1 for line in iter(f.readline, "\n") if line[0] != comment)
    data = np.loadtxt(filename, comments=comment)
    n_spectra = len(data) // n_bins

    return data, n_bins, n_spectra


def make_line_collection(spectra, column, n_bins, n_spectra, H, ignore=0):
    segments = [
        np.column_stack(
            [
                spectra[i * n_bins : (i + 1) * n_bins, 0] / H,
                spectra[i * n_bins : (i + 1) * n_bins, column],
            ]
        )
        for i in range(ignore, n_spectra)
    ]

    times = np.arange(ignore, n_spectra)
    norm = Normalize(times.min(), times.max())

    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(times)
    return lc


def plot_spectrum(
    save_path,
    spectra,
    column,
    n_bins,
    n_spectra,
    H,
    xlabel,
    ylabel,
    ignore=0,
    loglog=False,
):
    fig, ax = plt.subplots()

    lc = make_line_collection(spectra, column, n_bins, n_spectra, H, ignore)
    ax.add_collection(lc)
    ax.autoscale()

    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(lc, ax=ax, label=r"$\tilde\eta$")

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    save_figure(fig, save_path)


def plot_spectra_numerical(save_dir, spectra, n_bins, n_spectra):
    plot_spectrum(
        save_dir / "spectra_field_num.pdf",
        spectra,
        1,
        n_bins,
        n_spectra,
        1,
        xlabel=r"$k/\omega_\star$",
        ylabel=r"$\tilde{\Delta}_{\tilde\phi_0}$",
        ignore=120,
    )

    plot_spectrum(
        save_dir / "spectra_derivative_num.pdf",
        spectra,
        2,
        n_bins,
        n_spectra,
        1,
        xlabel=r"$k/\omega_\star$",
        ylabel=r"$\tilde{\Delta}_{\tilde\phi'_0}$",
        ignore=15,
    )

    plot_spectrum(
        save_dir / "occupation_number_num.pdf",
        spectra,
        3,
        n_bins,
        1,
        1,
        xlabel=r"$k/\omega_\star$",
        ylabel=r"$\tilde n_{\tilde\phi}$",
    )


def plot_spectra_physical(save_dir, spectra, n_bins, n_spectra, omega_star, f_star, H):
    spectra = spectra.copy()
    spectra[:, 0] *= omega_star
    spectra[:, 1] *= f_star**2
    spectra[:, 2] *= f_star**2 * omega_star**2

    plot_spectrum(
        save_dir / "spectra_field_phys.pdf",
        spectra,
        1,
        n_bins,
        n_spectra,
        H,
        xlabel=r"$k/H$",
        ylabel=r"$\Delta_{\phi_0}$",
        ignore=120,
    )

    plot_spectrum(
        save_dir / "spectra_derivative_phys.pdf",
        spectra,
        2,
        n_bins,
        n_spectra,
        H,
        xlabel=r"$k/H$",
        ylabel=r"$\Delta_{\phi'_0}$",
        ignore=15,
    )

    plot_spectrum(
        save_dir / "occupation_number_phys.pdf",
        spectra,
        3,
        n_bins,
        1,
        H,
        xlabel=r"$k/H$",
        ylabel=r"$\tilde n_{\tilde\phi}$",
    )


def plot_gw_spectrum(save_dir, spectra, n_bins, n_spectra, H):
    plot_spectrum(
        save_dir / "spectra_gw.pdf",
        spectra,
        1,
        n_bins,
        n_spectra,
        H,
        xlabel=r"$\kappa=k/(H\omega_\star)$",
        ylabel=r"$\Omega_\mathrm{GW}(k,t)$",
        ignore=15,
        loglog=True,
    )


def plot_gw_energies(save_dir, phiDat, sl=slice(None)):

    eta = phiDat[sl, 0]
    rho_frac = phiDat[sl, 1]

    fig, ax = plt.subplots()
    ax.plot(eta, rho_frac)
    ax.set_xlabel(r"$\tilde\eta$")
    ax.set_ylabel(r"$\rho_\mathrm{GW}/\rho_\mathrm{tot}$")

    save_figure(fig, save_dir / "energies_gw.pdf")


def main():
    base_dirs = ["mu_MPle-2/", "mu_MPle-4/", "mu_MPle-6/", "mu_MPle-8/"]

    mH = 100
    r = mu_over_sqrtlam_from_mH(mH)

    mus = np.array([M_PL * 10 ** (-i) for i in range(2, 9, 2)])
    lams = (mus / r) ** 2

    omega_stars = mus
    f_stars = omega_stars / np.sqrt(lams)
    Hs = mus**2 / (np.sqrt(12) * M_PL * np.sqrt(lams))

    input_dirs = [Path("../output/mexhat") / d for d in base_dirs]
    output_dirs = [Path("./figures/mexhat") / d for d in base_dirs]

    for mu, lam in zip(mus, lams):
        print(f"{mu=:.2E}   {lam=:.2E}")

    if SAVE:
        for d, out, omega, f, H in zip(
            input_dirs, output_dirs, omega_stars, f_stars, Hs
        ):
            out.mkdir(parents=True, exist_ok=True)

            plot_average_field(out, np.loadtxt(d / "average_scalar_0.txt"))
            plot_energy_densities(out, np.loadtxt(d / "average_energies.txt"))

            spectra, n_bins, n_spec = load_spectrum(d / "spectra_scalar_0.txt")
            plot_spectra_numerical(out, spectra, n_bins, n_spec)
            plot_spectra_physical(out, spectra, n_bins, n_spec, omega, f, H)

            spectra, n_bins, n_spec = load_spectrum(d / "spectra_gws.txt")
            plot_gw_spectrum(out, spectra, n_bins, n_spec, H)

            plot_gw_energies(out, np.loadtxt(d / "energy_gws.txt"))


if __name__ == "__main__":
    main()
