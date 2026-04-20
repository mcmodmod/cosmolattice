import numpy as np
import matplotlib.pyplot as plt

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

SAVE = True
base_dirs = ["mu_MPle-2/", "mu_MPle-4/", "mu_MPle-6/", "mu_MPle-8/"]

plt.figure()
for base_dir in base_dirs:
    directory = "../output/mexhat/" + base_dir
    enDat = np.loadtxt(directory + "average_energies.txt")
    eta = enDat[:, 0]
    E_G = enDat[:, 2]

    lim1 = 0
    lim2 = len(eta) // 2

    plt.plot(eta[lim1:lim2], E_G[lim1:lim2], label=base_dir)


plt.yscale("log")
plt.xlabel(r"$\tilde \eta$")
plt.ylabel(r"$\tilde E_G$")
plt.legend()
if SAVE:
    savefile = "./figures/mexhat/gradient_energies.pdf"
    plt.title(savefile)
    plt.savefig(savefile, format="pdf")
