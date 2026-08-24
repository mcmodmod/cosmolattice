import pickle
from pathlib import Path
from plot_data import save_figure

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from GW_spectrum_SNR import GW_analysis
from PLIs.GW_detectors import GW_detectors
from Veff_Daniel import EffectivePotential

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

YEAR = 365 * 24 * 3600
DAY = 24 * 3600

# Parameter-space grid
x_min = np.log10(5e4)
x_max = np.log10(4e7)
y_min = np.log10(1e-4)
y_max = np.log10(2e-1)
g_values = np.logspace(y_min, y_max, 100)
m_values = np.logspace(x_min, x_max, 100)  # m_Z' in GeV

# Output


# ---------------------------------------------------------------------------
# Experimental sensitivities
# ---------------------------------------------------------------------------

# Standard detector sensitivity curves
ska_spec = GW_detectors("ska").spectral_density
lisa_spec = GW_detectors("lisa").spectral_density
bbo_spec = GW_detectors("bbo").spectral_density
et_spec = GW_detectors("et").spectral_density
decigo_spec = GW_detectors("decigo").spectral_density

# muARES sensitivity curve
with open("PLIs/muares_spec.p", "rb") as file:
    f_muares, omega_muares = pickle.load(file)

muares_spec = interp1d(
    f_muares,
    omega_muares,
    bounds_error=False,
    fill_value=np.inf,
)


# All experiment-specific information is collected here.
#
# "auto" selects GW_analysis.SNR_auto().
# Otherwise GW_analysis.SNR() is used.
experiments = {
    "SKA": {
        "sensitivity": ska_spec,
        "t_obs": 20 * YEAR,
        "fmin": 1 / (20 * YEAR),
        "fmax": 0.1 / (14 * DAY),
        "threshold": 4,
        "auto": True,
    },
    "LISA": {
        "sensitivity": lisa_spec,
        "t_obs": 4 * YEAR,
        "fmin": 1e-7,
        "fmax": 1e-1,
        "threshold": 10,
        "auto": True,
    },
    "BBO": {
        "sensitivity": bbo_spec,
        "t_obs": 4 * YEAR,
        "fmin": 1e-3,
        "fmax": 1,
        "threshold": 10,
        "auto": False,
    },
    "ET": {
        "sensitivity": et_spec,
        "t_obs": 5 * YEAR,
        "fmin": 1,
        "fmax": 10,
        "threshold": 5,
        "auto": False,
    },
    r"$\mu$ARES": {
        "sensitivity": muares_spec,
        "t_obs": 7 * YEAR,
        "fmin": np.min(f_muares),
        "fmax": np.max(f_muares) * 0.01,
        "threshold": 10,
        "auto": True,
    },
    "DECIGO": {
        "sensitivity": decigo_spec,
        "t_obs": 4 * YEAR,
        "fmin": 1e-3,
        "fmax": 1,
        "threshold": 10,
        "auto": False,
    },
}


# ---------------------------------------------------------------------------
# Compute SNRs
# ---------------------------------------------------------------------------

# One SNR grid for each experiment.
shape = (len(g_values), len(m_values))

m_over_H_grid = np.full(shape, np.nan)
T_rh_grid = np.full(shape, np.nan)
H_star_grid = np.full(shape, np.nan)

snr_grids = {name: np.full(shape, np.nan) for name in experiments}

for i, g_bl in enumerate(g_values):

    print(f"Computing row {i + 1}/{len(g_values)}: g_B-L = {g_bl:.3e}")

    for j, m_zprime in enumerate(m_values):
        # print(f"Computing column {j + 1}/{len(m_values)}: m_Z' = {m_zprime:.3e}")
        # Compute the physical model quantities once at this parameter point.
        model = EffectivePotential(g_bl, m_zprime, vh_qcd=0.1)
        model.interpolations()

        m_over_H = model.m_over_H()
        T_rh = model.find_T_vac()
        H_star = model.Hubble(T_rh)
        m_over_H_grid[i, j] = m_over_H
        T_rh_grid[i, j] = T_rh
        H_star_grid[i, j] = H_star

        gw = GW_analysis(
            m_over_H=m_over_H,
            H_star=H_star,
            T_rh=T_rh,
        )

        # Evaluate the same GW signal for every experiment.
        for name, experiment in experiments.items():

            if experiment["auto"]:
                snr = gw.SNR_auto(
                    experiment["t_obs"],
                    experiment["sensitivity"],
                    experiment["fmin"],
                    experiment["fmax"],
                )
            else:
                snr = gw.SNR(
                    experiment["t_obs"],
                    experiment["sensitivity"],
                    experiment["fmin"],
                    experiment["fmax"],
                )

            snr_grids[name][i, j] = snr

np.savez_compressed(
    "SNR_scan_2.npz",
    m_zprime=m_values,
    g_bl=g_values,
    m_over_H=m_over_H_grid,
    T_rh=T_rh_grid,
    H_star=H_star_grid,
    snr_ska=snr_grids["SKA"],
    snr_lisa=snr_grids["LISA"],
    snr_bbo=snr_grids["BBO"],
    snr_et=snr_grids["ET"],
    snr_muares=snr_grids[r"$\mu$ARES"],
    snr_decigo=snr_grids["DECIGO"],
)
