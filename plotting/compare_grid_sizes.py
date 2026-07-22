import matplotlib.pyplot as plt
from pathlib import Path
from simulation import Simulation
from load_data import load_gw_spectra
from plot_data import save_figure


def main():

    base_dir = Path("../output/mH1e4_testing")
    input_dirs = [
        base_dir / dir
        for dir in [
            "mH1e4_VV10_64",
            "mH1e4_LF_128",
            "mH1e4_VV10_256",
            "mH1e4_LF_512",
            "mH1e4_LF_1024",
        ]
    ]
    m_over_H = 1e4
    sims = [
        Simulation(input_dir, Path("./figures"), m_over_H) for input_dir in input_dirs
    ]
    labels = [
        r"$N=64$",
        r"$N=128$",
        r"$N=256$",
        r"$N=512$",
        r"$N=1024$",
    ]
    fig, ax = plt.subplots()
    for i, sim in enumerate(sims):
        spectrum = load_gw_spectra(sim.input_dir / "spectra_gws.txt")[
            -1
        ]  # Last time step
        k_over_H = spectrum["kappa"] * sim.omega_star / sim.H
        ax.plot(
            k_over_H,
            spectrum["omega_gw"],
            label=labels[i],
        )
    ax.set(xlabel=r"$k/H$", ylabel=r"$h^2\Omega_\mathrm{GW}(k,t)$")
    ax.set(xscale="log", yscale="log")
    fig.legend(loc="center")
    savefile = Path("./figures/grid_sizes_gw.pdf")
    save_figure(fig, savefile)


if __name__ == "__main__":
    main()
    print("All done!")
