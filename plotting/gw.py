import numpy as np
import matplotlib.pyplot as plt


def main():
    plt.style.use("classic")
    plt.tight_layout()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.labelsize": 20,
            "legend.fontsize": 13,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )
    # filename = "../output/lphi4_single/spectra_gws.txt"
    # filename = "../output/mexhat/spectra_gws.txt"
    # filename = "./test.txt"
    filename = f"../output/Veff_N64_eta1e5/spectra_gws.txt"
    # filename = f"../output/mexhat/test/spectra_gws.txt"
    print("Using data from", filename, "...")
    # Load the file, split into blocks separated by blank lines
    with open(filename) as f:
        # Read contents and create list of strings of the form:
        # [
        # "κ Ω_GW(k, t) #l\n
        #  κ Ω_GW(k, t) #l\n
        # ...",
        # "κ Ω_GW(k, t) #l\n
        #  κ Ω_GW(k, t) #l\n
        # ...",
        # ...
        # ]
        content = f.read().strip().split("\n\n")
        content = content[-100:]
    cmap = plt.get_cmap("YlOrRd", len(content))
    plt.figure(figsize=(8, 6))

    for j, block in enumerate(content):
        # Use np.loadtxt on each block
        data = np.loadtxt(block.splitlines())
        kappa = data[:, 0]
        omega_gw = data[:, 1]
        plt.plot(kappa, omega_gw, color=cmap(j), label=f"Time step {j+1}")
        # savepath = f"./figures/gif/frame{j:0>5}.png"
        # print("Saving figure to", savepath)
        # plt.savefig(savepath)

    # plt.ylim(10e-41, 10e-33)
    # plt.xlim(0.2, 13)
    plt.xlabel(r"$\kappa=k/\omega_\star$")
    plt.ylabel(r"$\Omega_\mathrm{GW}(k,t)$")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    # title = "V=lam phi**4 - mu phi**2, N=64, dt=0.05, kIR=0.2"
    # plt.title(title)

    # savepath = f"./figures/mexhat_spec_gws_N128_lam1e-{i}.pdf"
    savepath = "./figures/test.pdf"

    print("Saving figure to", savepath, "...")
    plt.savefig(savepath)


if __name__ == "__main__":
    main()
    print("All done!")
