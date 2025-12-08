from fenics import *
import numpy as np
import json
import os
from sys import argv
import matplotlib.pyplot as plt

path_to_mesh = os.path.expanduser(argv[1])
out_file= os.path.expanduser(argv[2])

mesh = Mesh()
with XDMFFile(path_to_mesh + "_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile(path_to_mesh + "_mf.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)



# Define spatial coordinates
x = SpatialCoordinate(mesh)
A = assemble(Constant(1) * dx(domain=mesh))

# Compute centroid
centroid = [0, 0]
centroid[0] = assemble(x[0] * dx(domain=mesh)) / A
centroid[1] = assemble(x[1] * dx(domain=mesh)) / A

# Shifted coordinates
x0 = x[0] - centroid[0]
x1 = x[1] - centroid[1]

# Central second moments
T_xx = assemble(x0 * x0 * dx(domain=mesh)) / A
T_xy = assemble(x1 * x0 * dx(domain=mesh)) / A
T_yx = T_xy
T_yy = assemble(x1 * x1 * dx(domain=mesh)) / A

# Print tensor components
print("T_xx:", T_xx)
print("T_xy:", T_xy)
print("T_yx:", T_yx)
print("T_yy:", T_yy)

# Moment tensor
T = np.zeros((2,2), dtype='float')
T[0,0] = T_xx
T[0,1] = T_xy
T[1,0] = T_yx
T[1,1] = T_yy

# Optional: compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(T)  # For symmetric tensors

# Create a dict of properties to save
shape_props = {
    "area": A,
    "centroid": centroid,
    "moment_tensor": T.tolist(),  # convert NumPy array to nested list
    "eigenvalues": eigvals.tolist(),
    "eigenvectors": eigvecs.tolist()
}


# Save to JSON
with open(out_file + '.json', "w") as f:
    json.dump(shape_props, f, indent=4)
    