import h5py
import matplotlib.pyplot as plt
from plot_data import save_figure
from pathlib import Path

filenames = [
    Path("../output/mH1e3/") / file
    for file in [
        "mexhat_DATE_d13_m8_y2026_TIME_h24_m51_s23.h5",
        "mexhat_DATE_d13_m8_y2026_TIME_h24_m58_s32.h5",
        "mexhat_DATE_d14_m8_y2026_TIME_h1_m7_s37.h5",
        "mexhat_DATE_d14_m8_y2026_TIME_h1_m16_s19.h5",
        "mexhat_DATE_d14_m8_y2026_TIME_h1_m31_s9.h5",
    ]
]

for i, filename in enumerate(filenames):
    # Read the 3D scalar field
    with h5py.File(filename, "r") as f:
        scalar = f["scalar_0(x)"][:]

    # Take a slice through the middle of the volume
    middle = scalar.shape[0] // 2
    slice_2d = scalar[middle, :, :]

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the slice
    im = ax.imshow(slice_2d, origin="lower", cmap="viridis")

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Scalar field value")

    # Labels and title
    ax.set_xlabel(r"$y$")
    ax.set_xlabel(r"$z$")
    save_figure(fig, Path(f"./figures/hdf5_{i}.pdf"))
