from pathlib import Path
import numpy as np

from Veff_Daniel import EffectivePotential


def m_over_H(g_bl, m_zprime):
    veff = EffectivePotential(g_bl, m_zprime, vh_qcd=0.1)
    veff.interpolations()
    return veff.m_over_H()


# -------------------------------------------------------------------------
# Grid settings
# -------------------------------------------------------------------------

x_min = 5e4
x_max = 4e7
y_min = 1e-4
y_max = 2e-1

# Increase these for the final high-resolution calculation.
n_m = 75
n_g = 75

m_grid = np.logspace(
    np.log10(x_min),
    np.log10(x_max),
    n_m,
)

g_grid = np.logspace(
    np.log10(y_min),
    np.log10(y_max),
    n_g,
)

M, G = np.meshgrid(m_grid, g_grid)

MH = np.empty_like(M)


# -------------------------------------------------------------------------
# Compute m/H on the grid
# -------------------------------------------------------------------------

total_points = n_m * n_g
counter = 0

for i in range(n_g):
    for j in range(n_m):
        print(f"{(i,j)=}")
        MH[i, j] = m_over_H(
            G[i, j],
            M[i, j],
        )

        counter += 1

        # Simple progress output every 100 calculations.
        if counter % 100 == 0 or counter == total_points:
            print(
                f"Completed {counter}/{total_points} "
                f"({100 * counter / total_points:.1f}%)"
            )


# -------------------------------------------------------------------------
# Save result
# -------------------------------------------------------------------------

output_dir = Path("./data")
output_dir.mkdir(parents=True, exist_ok=True)

savefile = output_dir / "m_over_H_grid.npz"

np.savez_compressed(
    savefile,
    m_grid=m_grid,
    g_grid=g_grid,
    m_over_H=MH,
)

print(f"\nSaved m/H grid to:")
print(savefile)
