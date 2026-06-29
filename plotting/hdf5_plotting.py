import h5py
import pyvista as pv

filename = "../output/mexhat_DATE_d22_m6_y2026_TIME_h18_m52_s33.h5"

with h5py.File(filename, "r") as f:
    field = f["scalar_0(x)"][:]

    def print_structure(name, obj):
        print(name, obj.shape if hasattr(obj, "shape") else "")

    f.visititems(print_structure)
grid = pv.ImageData()
grid.dimensions = field.shape

grid["scalar"] = field.flatten(order="F")

plotter = pv.Plotter()
plotter.add_volume(grid, cmap="viridis")
plotter.show()
