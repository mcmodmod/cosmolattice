import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "axes.labelsize": 23,
        "axes.grid": True,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.constrained_layout.use": True,
    }
)

SAVE = True
base_dirs = ["mu11_lam-10/", "mu12_lam-8/", "mu13_lam-6/", "mu14_lam-4/"]
labels = [
    "mu=2.44E+14 lam=8.33E-04",
    "mu=2.44E+13 lam=8.33E-06",
    "mu=2.44E+12 lam=8.33E-08",
    "mu=2.44E+11 lam=8.33E-10",
]

plt.figure()
for base_dir, label in zip(base_dirs, labels):
    input_dir = "../output/mH1e3/" + base_dir
    enDat = np.loadtxt(input_dir + "average_energies.txt")
    eta = enDat[:, 0]
    E_G = enDat[:, 2]

    lim1 = 0
    lim2 = len(eta) // 2

    plt.plot(eta[lim1:lim2], E_G[lim1:lim2], label=label)


# plt.yscale("log")
plt.xlabel(r"$\tilde \eta$")
plt.ylabel(r"$\tilde E_G$")
plt.legend()
if SAVE:
    savefile = "./figures/mH1e3/gradient_energies.pdf"
    plt.title(savefile)
    plt.savefig(savefile, format="pdf")
