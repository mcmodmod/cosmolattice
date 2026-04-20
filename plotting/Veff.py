import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "axes.labelsize": 20,
        "legend.fontsize": 13,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }
)

# ---------------------------
# Parameter values
# ---------------------------
g = 0.01
mZp = 1.0e5  # GeV
v_QCD = 0.1  # GeV
M_W = 80.379
G_mu = 1.1663787e-5
M_h = 125.1
pi = np.pi

# Derived parameters
g2_at_mZ = np.sqrt(8 * M_W**2 * G_mu / np.sqrt(2))
lam_h = (g2_at_mZ**2) / (16 * M_W**2) * 2 * (M_h**2)
vh_at_mZ = np.sqrt(4 * M_W**2 / g2_at_mZ**2)

lam_p = 2 * lam_h * (vh_at_mZ * 2 * g / mZp) ** 2
lam_phi = g**4 / pi**2
mu = mZp
vev = mZp / (2 * g)

omega_star = mZp
f_star = mZp


# ---------------------------
# Potential function V(phi)
# ---------------------------
def V(phi):
    return (
        -lam_p * v_QCD**2 / 2.0 * phi**2
        + 0.25 * lam_phi * phi**4
        + (3.0 / (64.0 * pi**2))
        * (2 * g * phi) ** 4
        * (np.log((2 * g * phi / mu) ** 2) - 5.0 / 6.0)
    )


def V_num(phi_tilde):
    return 1 / (omega_star * f_star) ** 2 * (V(phi_tilde * f_star))


# ---------------------------
# Plotting physical potential
# ---------------------------
phis = np.linspace(0.1, vev * 1.5, 800)
plt.figure(figsize=(8, 6))
plt.plot(phis, V(phis), label="Physical potential: " + r"$V(\phi)$")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$V(\phi)$")
plt.grid(True)
plt.tight_layout()
plt.legend(loc="upper left")
plt.savefig("./figures/Veff_phys.pdf", format="pdf")


# ---------------------------
# Plotting numerical potential
# ---------------------------
tilde_phis = phis / f_star
plt.figure(figsize=(8, 6))
plt.plot(
    tilde_phis,
    V_num(tilde_phis),
    label="Numerical potential: " + r"$\tilde V(\tilde\phi)$",
)
plt.xlabel(r"$\tilde\phi$")
plt.ylabel(r"$\tilde V(\tilde\phi)$")
plt.grid(True)
plt.tight_layout()
plt.legend(loc="upper left")
plt.savefig("./figures/Veff_num.pdf", format="pdf")
