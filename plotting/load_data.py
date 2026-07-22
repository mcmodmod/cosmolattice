import io
import numpy as np
from pathlib import Path

M_PL = 2.435e18
h = 0.67
GEV_TO_HZ = 2.417989e23


def _frequency_from_kappa(kappa, mu):
    """
    Convert dimensionless kappa = k / omega_star
    to physical source frequency f_star [Hz].
    """
    omega_star = mu
    return kappa * omega_star / (2 * np.pi) * 2.417989e23


def _wavenumber_from_kappa(kappa, mu):
    """
    Convert dimensionless kappa to physical wave number k_star.
    """
    omega_star = mu
    return kappa * omega_star


def _read_blocks(filename):
    """
    Read a block-formatted spectrum file.

    Returns
    -------
    list[np.ndarray]
        Each array has shape (Nbins, Ncolumns).
    """
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    return [
        np.loadtxt(io.StringIO(block)) for block in text.split("\n\n") if block.strip()
    ]


def _last_block(filename):
    """Return only the final time block."""
    return _read_blocks(filename)[-1]


def load_scale_factor(filename):
    data = np.loadtxt(filename)
    eta = data[:, 0]
    a = data[:, 1]
    return eta, a


def load_average_field(filename):
    data = np.loadtxt(filename)
    eta = data[:, 0]
    phi = data[:, 1]
    return eta, phi


def load_energy_densities(filename):
    data = np.loadtxt(filename)
    eta = data[:, 0]
    E_K = data[:, 1]
    E_G = data[:, 2]
    E_V = data[:, 3]
    rho = data[:, 4]
    return eta, {"E_K": E_K, "E_G": E_G, "E_V": E_V, "rho": rho}


def load_gw_energies(filename):
    data = np.loadtxt(filename)
    eta = data[:, 0]
    rho_GW_over_rho_tot = data[:, 1]
    rho_GW = data[:, 2]
    return eta, rho_GW_over_rho_tot, rho_GW


def load_scalar_spectra(filename):
    """
    Load all scalar spectra using dimensionless kappa.

    Returns
    -------
    list[dict]

    Each element corresponds to one simulation time and contains

        "kappa"       : dimensionless wave number
        "field"       : field spectrum
        "derivative"  : derivative spectrum
        "occupation"  : occupation number
    """

    blocks = _read_blocks(filename)

    spectra = []

    for block in blocks:
        spectra.append(
            {
                "kappa": block[:, 0],
                "field": block[:, 1],
                "derivative": block[:, 2],
                "occupation": block[:, 3],
            }
        )

    return spectra


def load_gw_spectra(filename):
    """
    Load all scalar spectra using dimensionless kappa.

    Returns
    -------
    list[dict]

    Each element corresponds to one simulation time and contains

        "kappa"       : dimensionless wave number
        "omega_gw"       : GW spectrum
    """

    blocks = _read_blocks(filename)

    spectra = []

    for block in blocks:
        spectra.append(
            {
                "kappa": block[:, 0],
                "omega_gw": h**2 * block[:, 1],
            }
        )

    return spectra


def load_last_gw_spectrum(filename):
    """
    Last spectrum.

    Returns
    -------
    kappa
    h^2 Omega_GW
    """
    data = _last_block(filename)

    kappa = data[:, 0]
    omega = h**2 * data[:, 1]

    return kappa, omega


def load_last_gw_spectrum_f(filename, mu):
    """
    Last spectrum as (f_star, Omega_GW).
    """
    kappa, omega = load_last_gw_spectrum(filename)
    f = _frequency_from_kappa(kappa, mu)
    return f, omega


def load_last_gw_spectrum_k(filename, mu):
    """
    Last spectrum as (k_star, Omega_GW).
    """
    kappa, omega = load_last_gw_spectrum(filename)
    k = _wavenumber_from_kappa(kappa, mu)
    return k, omega


def load_all_gw_spectra(filename):
    """
    Returns every spectrum.

    Returns
    -------
    kappa : (Nt,Nk)
    omega : (Nt,Nk)
    """

    blocks = _read_blocks(filename)

    kappa = np.array([b[:, 0] for b in blocks])
    omega = np.array([h**2 * b[:, 1] for b in blocks])

    return kappa, omega


def load_all_gw_spectra_f(filename, mu):
    """
    Returns every spectrum in frequency space.
    """

    kappa, omega = load_all_gw_spectra(filename)
    f = _frequency_from_kappa(kappa, mu)

    return f, omega


def load_all_gw_spectra_k(filename, mu):
    """
    Returns every spectrum in wave-number space.
    """

    kappa, omega = load_all_gw_spectra(filename)
    k = _wavenumber_from_kappa(kappa, mu)

    return k, omega


def spectrum_peak(x, omega):
    """
    Return the peak position of a spectrum.

    Parameters
    ----------
    x : ndarray
        kappa, frequency or wave number.
    omega : ndarray

    Returns
    -------
    x_peak
    omega_peak
    """

    idx = np.argmax(omega[1:]) + 1  # skip first point

    return x[idx], omega[idx]


if __name__ == "__main__":

    mH = 1e3
    mu = M_PL * 1e-7
    file = "../output/mH1e3_512/spectra_scalar_0.txt"
