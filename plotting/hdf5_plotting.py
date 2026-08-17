import h5py
import matplotlib.pyplot as plt
from plot_data import save_figure
from pathlib import Path


def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{name}/")


filenames = [
    Path("../output/mH1e3_512_hdf5/snapshots") / file
    for file in [
        "mH1e3_512_t0.h5",
        "mH1e3_512_t5.h5",
        "mH1e3_512_t10.h5",
        "mH1e3_512_t50.h5",
        "mH1e3_512_t150.h5",
    ]
]

ts = [0, 5, 10, 50, 150]
for i, filename in enumerate(filenames):
    # Read the 3D scalar field
    with h5py.File(filename, "r") as f:
        # f.visititems(print_structure)
        dataset = f["scalar_0(x)"]

        if not isinstance(dataset, h5py.Dataset):
            raise TypeError("'scalar_0(x)' is not an HDF5 dataset")

        scalar = dataset[:]

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
    save_figure(fig, Path(f"./figures/hdf5_{ts[i]}.pdf"))
