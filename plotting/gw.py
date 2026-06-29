import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_data import load_spectrum, plot_gw_spectrum
from get_params_from_mH import lam_from_mu


def main():
    M_PL = 2.435 * 10 ** (18)
    mH = 10**4
    mu = 2.435e10
    lam = lam_from_mu(mu, mH)
    omega_star = mu
    f_star = omega_star / np.sqrt(lam)
    H = mu**2 / (np.sqrt(12) * M_PL * np.sqrt(lam))

    input_file = Path("../output/mH1e4/mH1e4_VV10_256/spectra_gws.txt")
    out_dir = Path("./figures")
    spectra, n_bins, n_spec = load_spectrum(input_file)
    plot_gw_spectrum(out_dir, spectra, n_bins, n_spec, H / omega_star)


if __name__ == "__main__":
    main()
    print("All done!")
