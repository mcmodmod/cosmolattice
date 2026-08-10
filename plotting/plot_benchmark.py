import io
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from pathlib import Path
from plot_data import save_figure
from load_data import load_scale_factor, load_field_variance, load_average_field


def plot_average_field_with_sqrtvariance(sim, sl=slice(None)):

    eta, var = load_field_variance(sim.input_dir / "average_scalar_0.txt")
    _, phi = load_average_field(sim.input_dir / "average_scalar_0.txt")

    mu_eta = eta[sl]
    var = var[sl]
    phi = phi[sl]

    fig, ax = plt.subplots()
    ax.plot(
        mu_eta,
        np.abs(phi),
        label=r"$\left|\langle\tilde\varphi\rangle\right|/v_{\mathrm{sim}}$",
    )
    ax.plot(
        mu_eta,
        np.sqrt(var),
        label=r"$\sqrt{\langle\varphi^2\rangle - \langle\varphi\rangle^2}/v_{\mathrm{sim}}$",
    )
    # ax.set(yscale="log")
    ax.set_xlabel(r"$\tilde \eta$")
    # ax.set_ylabel(r"$\langle\tilde\varphi\rangle$")
    ax.legend()

    save_figure(fig, sim.output_dir / "average_field_with_sqrtvar.pdf")


def main():
    input_dir = Path("../output/phi_init_comparison/mH1e3_phi_init_0")
    output_dir = Path("./figures/phi_init_comparison/mH1e3_phi_init_0")

    m_over_H = 1e3

    sim = Simulation(input_dir, output_dir, m_over_H)
    time_steps = [0, 17, 13, 75]
    plot_average_field_with_sqrtvariance(sim)


if __name__ == "__main__":
    main()
    print("All done!")
