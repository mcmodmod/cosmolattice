import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from simulation import Simulation
from load_data import (
    load_average_field,
    load_field_rms,
    load_energy_densities,
    load_scale_factor,
    load_scalar_spectra,
    load_gw_spectra,
    load_gw_energies,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pgf.preamble": r"""
        \usepackage[lining,semibold,scaled=1.05]{ebgaramond}
        \usepackage{amsmath}
        \usepackage[vvarbb,subscriptcorrection]{newtxmath}
    """,
        "pgf.texsystem": "pdflatex",
        "axes.labelsize": 20,
        "legend.fontsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.constrained_layout.use": True,
    }
)
SAVE = True


def save_figure(fig, path: Path):
    if SAVE:
        print(f"Saving figure to {path}")
        fig.savefig(path, format="pdf", backend="pgf")
    plt.close(fig)


def plot_scale_factor(sim, sl=slice(None)):
    eta, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")

    eta = eta[sl]
    a = a[sl]

    fig, ax = plt.subplots()
    ax.plot(eta, a)
    ax.set_xlabel(r"$\tilde\eta$")
    ax.set_ylabel(r"$a$")

    save_figure(fig, sim.output_dir / "scale_factor.pdf")


def plot_average_field(sim, sl=slice(None)):

    eta, phi = load_average_field(sim.input_dir / "average_scalar_0.txt")
    eta = eta[sl]
    phi = phi[sl]

    fig, ax = plt.subplots()
    ax.plot(eta, phi)
    ax.set_xlabel(r"$\tilde\eta$")
    ax.set_ylabel(r"$\langle\tilde\varphi\rangle$")

    save_figure(fig, sim.output_dir / "average_field.pdf")


def plot_rms(sim, sl=slice(None)):

    eta, phi = load_field_rms(sim.input_dir / "average_scalar_0.txt")
    eta = eta[sl]
    rms = phi[sl]

    fig, ax = plt.subplots()
    ax.plot(eta, rms)
    ax.set_xlabel(r"$\tilde\eta$")
    ax.set_ylabel(r"$\mathrm{rms}\left(\tilde\varphi\right)$")

    save_figure(fig, sim.output_dir / "field_rms.pdf")


def plot_energy_densities(sim, sl=slice(None)):
    eta, energies = load_energy_densities(sim.input_dir / "average_energies.txt")

    labels_to_data = {
        r"$\langle \tilde E_K \rangle$": energies["E_K"],
        r"$\langle \tilde E_G \rangle$": energies["E_G"],
        r"$\langle \tilde E_V \rangle$": energies["E_V"],
        r"$\langle \tilde \rho \rangle$": energies["rho"],
    }

    fig, ax = plt.subplots()
    for label, data in labels_to_data.items():
        ax.plot(eta, data[sl], label=label)

    ax.set(
        xlabel=r"$\tilde \eta$",
        ylabel=r"Energy density",
    )
    ax.legend(frameon=False)

    save_figure(fig, sim.output_dir / "average_energies.pdf")


def plot_gw_energies(sim):
    eta, rho_frac, rho_GW = load_gw_energies(sim.input_dir / "energy_gws.txt")
    # _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")

    fig, ax = plt.subplots()
    ax.plot(eta, rho_frac)
    ax.set(yscale="log")
    ax.set_xlabel(r"$\tilde\eta$")
    ax.set_ylabel(r"$\rho_\mathrm{GW}/\rho_\mathrm{tot}$")

    save_figure(fig, sim.output_dir / "gw_energies.pdf")


def plot_spectra_num(sim: Simulation, quantity: str):
    """
    Plot one quantity from all time steps with a color gradient.

    Parameters
    ----------
    sim : Simulation
    quantity : str
        One of 'field', 'derivative', 'occupation'.
    """
    spectra = load_scalar_spectra(sim.input_dir / "spectra_scalar_0.txt")
    _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")
    cmap = plt.cm.viridis
    nsteps = len(spectra)
    labels = {
        "field": r"$\Delta_{\varphi}/v_{\mathrm{sim}}^2$",
        "derivative": r"$\tilde\Delta_{\tilde\varphi'}$",
        "occupation": r"$\tilde n_{\tilde k}$",
    }
    fig, ax = plt.subplots()

    for i, spec in enumerate(spectra):
        color = cmap(i / max(nsteps - 1, 1))
        ax.plot(spec["kappa"] / a[i], spec[quantity], color=color)

    ax.set_xlabel(r"$k_{\mathrm{phys}}/\mu$")
    ax.set_ylabel(labels[quantity])
    ax.set(xscale="log", yscale="log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=nsteps - 1))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"$\tilde\eta$")

    filenames = {
        "field": "spectrum_field_num.pdf",
        "derivative": "spectrum_derivative_num.pdf",
        "occupation": "occupation_number_num.pdf",
    }
    save_figure(fig, sim.output_dir / filenames[quantity])


def plot_spectra_phys(sim: Simulation, quantity: str):
    """
    Plot one quantity from all time steps with a color gradient.

    Parameters
    ----------
    sim : Simulation
    quantity : str
        One of 'field', 'derivative', 'occupation'.
    """
    spectra = load_scalar_spectra(sim.input_dir / "spectra_scalar_0.txt")
    cmap = plt.cm.viridis
    nsteps = len(spectra)
    labels = {
        "field": r"$\Delta_{\varphi}$",
        "derivative": r"$\Delta_{\varphi'}$",
        "occupation": r"$\tilde n_{\tilde k}$",
    }
    fig, ax = plt.subplots()

    for i, spec in enumerate(spectra):
        spec["kappa"] = spec["kappa"] * sim.omega_star / sim.H
        match quantity:
            case "field":
                spec[quantity] *= sim.f_star**2
            case "derivative":
                spec[quantity] *= sim.f_star**2 * sim.omega_star**2
        color = cmap(i / max(nsteps - 1, 1))
        ax.plot(spec["kappa"], spec[quantity], color=color)

    ax.set_xlabel(r"$k/H$")
    ax.set_ylabel(labels[quantity])
    ax.set(xscale="log", yscale="log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=nsteps - 1))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"$\tilde\eta$")

    filenames = {
        "field": "spectrum_field_phys.pdf",
        "derivative": "spectrum_derivative_phys.pdf",
        "occupation": "occupation_number_phys.pdf",
    }
    save_figure(fig, sim.output_dir / filenames[quantity])


def plot_gw_spectra(sim: Simulation):
    """
    Plot one quantity from all time steps with a color gradient.

    Parameters
    ----------
    sim : Simulation
    """
    spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
    _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")
    cmap = plt.cm.viridis
    nsteps = len(spectra)
    fig, ax = plt.subplots()

    for i, spec in enumerate(spectra):
        spec["kappa"] = spec["kappa"] / a[i]
        color = cmap(i / max(nsteps - 1, 1))
        ax.plot(spec["kappa"], spec["omega_gw"], color=color)

    ax.set_xlabel(r"$k_\mathrm{phys}/\mu$")
    ax.set_ylabel(r"$h^2\Omega_\mathrm{GW}$")
    ax.set(xscale="log", yscale="log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=nsteps - 1))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"$\tilde\eta$")

    save_figure(fig, sim.output_dir / "gw_spectra.pdf")


def plot_gw_spectrum_redshifted(sim: Simulation):
    spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
    cmap = plt.cm.viridis
    nsteps = len(spectra)
    fig, ax = plt.subplots()

    for i, spec in enumerate(spectra):
        spec["kappa"] = spec["kappa"] * sim.omega_star / sim.H
        color = cmap(i / max(nsteps - 1, 1))
        ax.plot(spec["kappa"], spec["omega_gw"], color=color)

    ax.set_xlabel(r"$k/H$")
    ax.set_ylabel(r"$h^2\Omega_\mathrm{GW}$")
    ax.set(xscale="log", yscale="log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=nsteps - 1))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"$\tilde\eta$")

    save_figure(fig, sim.output_dir / "gw_spectra.pdf")


def main():
    base_dirs = [
        # "mH1e2_512",
        "phi_init_comparison/mH1e3_phi_init_0",
        "mH1e3_512",
        # "mH1e4_512",
        # "mH1e5_512",
        # "mH1e6_512",
        # "mH1e7_512",
        # "mH1e8_512",
        # "mH1e9_512",
    ]
    input_dirs = [Path("../output") / d for d in base_dirs]
    output_dirs = [Path("./figures") / d for d in base_dirs]

    # m_over_Hs = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    m_over_Hs = np.array([1e3, 1e3])

    sims = [
        Simulation(inp, out, m_over_H)
        for inp, out, m_over_H in zip(input_dirs, output_dirs, m_over_Hs)
    ]

    if SAVE:
        print("SAVE flag is True.")
        print("Plotting...")
        for sim in sims:
            sim.output_dir.mkdir(parents=True, exist_ok=True)

            plot_scale_factor(sim)
            plot_average_field(sim)
            plot_energy_densities(sim)

            plot_spectra_num(sim, "field")
            # plot_spectra_num(sim, "derivative")
            # plot_spectra_num(sim, "occupation")

            # plot_spectra_phys(sim, "field")
            # plot_spectra_phys(sim, "derivative")
            # plot_spectra_phys(sim, "occupation")

            plot_gw_spectra(sim)
            plot_gw_energies(sim)


if __name__ == "__main__":
    main()
    print("\nAll done!")
