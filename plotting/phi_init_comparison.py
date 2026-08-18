import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from simulation import Simulation
from load_data import load_gw_spectra, load_scale_factor
from plot_data import save_figure


def main():

    base_dirs = [
        "mH1e3_phi_init_0",
        "mH1e3_phi_init_e3",
        "mH1e3_phi_init_e6",
        "mH1e3_phi_init_e7",
        "mH1e3_phi_init_e8",
        "mH1e3_phi_init_e9",
    ]
    input_dirs = [Path("../output/phi_init_comparison") / d for d in base_dirs]
    m_over_H = 1e3
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H) for input_dir in input_dirs
    ]
    vev = sims[0].vev
    phi_init = [0] + [vev * 10**i for i in [-12, -9, -8, -7, -6]]
    print(phi_init)
    labels = [
        r"$\overline\varphi_\mathrm{i}=0$",
        r"$\overline\varphi_\mathrm{i}=v_\mathrm{sim}\times 10^{-12}$",
        r"$\overline\varphi_\mathrm{i}=v_\mathrm{sim}\times 10^{-9}$",
        r"$\overline\varphi_\mathrm{i}=v_\mathrm{sim}\times 10^{-8}$",
        r"$\overline\varphi_\mathrm{i}=v_\mathrm{sim}\times 10^{-7}$",
        r"$\overline\varphi_\mathrm{i}=v_\mathrm{sim}\times 10^{-6}$",
    ]

    peaks = np.empty(len(sims))
    for i, sim in enumerate(sims):
        spectra = load_gw_spectra(sim.input_dir / "spectra_gws.txt")
        # Find maximum value of omega_gw over all time steps:
        peaks[i] = max(spec["omega_gw"].max() for spec in spectra)
    dOmega = np.abs(peaks[0] - peaks[1]) / peaks[0]
    print(f"{dOmega=:.2E}")
    fig, ax = plt.subplots()
    ax.plot(phi_init / vev, peaks, linestyle="", marker=".", markersize=12)
    ax.hlines([peaks[0]], xmin=1e-10, xmax=1e-5, linestyle="--", color="red")
    ax.set_xscale("log")
    ax.set_ylabel(r"$h^2 \Omega_\mathrm{GW}^\mathrm{peak}$")
    ax.set_xlabel(r"$\overline\varphi_\mathrm{i}/v_\mathrm{sim}$")
    # ax.set_xlim(7e-10, 2e-6)
    save_figure(fig, Path("./figures/phi_init_comparison.pdf"))

    fig, ax = plt.subplots()
    for i, sim in enumerate(sims):
        # Last time step:
        _, a = load_scale_factor(sim.input_dir / "average_scale_factor.txt")
        spectrum = load_gw_spectra(sim.input_dir / "spectra_gws.txt")[-1]
        kappa = spectrum["kappa"] / a[-1]  # * sim.omega_star / sim.H
        ax.plot(
            kappa,
            spectrum["omega_gw"],
            label=labels[i],
        )
    ax.set(xlabel=r"$k_\mathrm{phys}/a$", ylabel=r"$h^2\Omega_\mathrm{GW}$")
    ax.set(xscale="log", yscale="log")
    ax.legend(frameon=False)
    savefile = Path("./figures/phi_init_comparison_spectra.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
