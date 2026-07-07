import h5py
import matplotlib.pyplot as plt

filename = "../output/mexhat_DATE_d22_m6_y2026_TIME_h18_m52_s33.h5"

with h5py.File(filename, "r") as f:
    field = f["scalar_0(x)"][:]

nx, ny, nz = field.shape

# Central slice
slice2d = field[:, :, nz // 2]

# Coordinates
extent = [0, nx, 0, ny]

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(
    slice2d.T,  # transpose so x is horizontal
    origin="lower",
    extent=extent,
    cmap="viridis",  # change if desired
    aspect="equal",
)

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label(r"$\phi$")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(r"$z=L/2$")

plt.tight_layout()

# plt.savefig("scalar_slice.pdf", bbox_inches="tight")
# plt.savefig("scalar_slice.png", dpi=400, bbox_inches="tight")

plt.show()
