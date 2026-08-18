import h5py
import matplotlib.pyplot as plt
from plot_data import save_figure
from load_data import load_average_field
from pathlib import Path
import numpy as np
from math import log10, floor


def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{name}/")


def format_number(x):
    if abs(x) < 0.1 or abs(x) > 100:
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    else:
        return f"{x:.2g}"


if __name__ == "__main__":
    filenames = [
        Path("../output/mH1e3_512_hdf5/snapshots") / file
        for file in [
            "mH1e3_512_t0.h5",
            "mH1e3_512_t5.h5",
            "mH1e3_512_t10.h5",
            "mH1e3_512_t20.h5",
            "mH1e3_512_t50.h5",
            # "mH1e3_512_t100.h5",
            "mH1e3_512_t150.h5",
        ]
    ]
    times = [0, 5, 10, 20, 50, 150]

    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 8.5))

    # Loop over files and axes
    for filename, time, ax in zip(filenames, times, axes.flat):
        with h5py.File(filename, "r") as f:
            dataset = f["scalar_0(x)"]

            if not isinstance(dataset, h5py.Dataset):
                raise TypeError("'scalar_0(x)' is not an HDF5 dataset")

            middle = dataset.shape[0] // 2
            slice_2d = dataset[middle, :, :]

        # Symmetric color scale for this snapshot
        limit = max(abs(slice_2d.min()), abs(slice_2d.max()))
        # limit = round(limit, -int(floor(log10(abs(limit)))))
        im = ax.imshow(
            slice_2d,
            origin="lower",
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
        )

        ax.set_title(rf"$\tilde \eta = {time}$", fontsize=15)
        ax.set_xlabel(r"$y$", fontsize=15)
        ax.set_ylabel(r"$z$", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a smaller colorbar
        cbar = fig.colorbar(
            im,
            ax=ax,
            shrink=0.8,
            pad=0.03,
        )

        cbar.set_ticks([-limit, 0, limit])
        cbar.set_ticklabels(
            [
                format_number(-limit),
                "0",
                format_number(limit),
            ]
        )
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_title(
            r"$\tilde\varphi$",
            fontsize=16,
            # labelpad=-10,
        )

    save_figure(fig, Path("figures/hdf5.pdf"))

    _, phi = load_average_field(Path("../output/mH1e3_512_new/average_scalar_0.txt"))
    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 8.5))

    # Loop over files and axes
    for filename, time, ax in zip(filenames, times, axes.flat):
        with h5py.File(filename, "r") as f:
            dataset = f["scalar_0(x)"]

            if not isinstance(dataset, h5py.Dataset):
                raise TypeError("'scalar_0(x)' is not an HDF5 dataset")

            middle = dataset.shape[0] // 2
            if time == 0:
                slice_2d = np.abs(dataset[middle, :, :] - phi[0])
            else:
                slice_2d = np.abs(dataset[middle, :, :] - phi[time - 1])

        # Symmetric color scale for this snapshot
        limit = max(abs(slice_2d.min()), abs(slice_2d.max()))
        im = ax.imshow(
            slice_2d,
            origin="lower",
            cmap="PuRd",
        )

        ax.set_title(rf"$\tilde \eta = {time}$", fontsize=15)
        ax.set_xlabel(r"$y$", fontsize=15)
        ax.set_ylabel(r"$z$", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        # Add a smaller colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label(
            r"$\tilde\varphi$",
            fontsize=14,
            labelpad=-10,
        )

        cbar.set_ticks([0, limit])
        cbar.set_ticklabels(
            [
                "0",
                format_number(limit),
            ]
        )
        cbar.ax.tick_params(labelsize=10)

    save_figure(fig, Path("figures/hdf5_differences.pdf"))
