### Set up the environments for running the full workflow ###

# Environment set up for FEniCS simulations
mamba create -n fenics fenics numpy scipy matplotlib

# Envronment set-up for analysis of fenics simulations
mamba create -n fenics_analysis  imageio vtk=9.2.6 pyvista=0.41.1 python=3.11 fenics ffmpeg matplotlib numpy=1.26 opencv scipy
mamba activate fenics_analysis 
pip install feret==1.3.1

# easier for mamba to install things in two steps, so we install the complex stuff first and then the easy stuff, this will probably still take a while to run

# Environment set-up for mesh-making
mamba create -n fenics_mesh  python-gmsh opencv meshio numpy

### Then to run a full workflow we would do the following ###

mamba activate fenics-mesh
python meshing_cell_masks.py # make mesh
mamba deactivate

mamba activate fenics
mpirun -np 2 python cell_10_sim.py # run simulation in parallel with 2 processors
mamba deactivate

mamba activate fenics_analysis
python example_code/analysing_cell_sim.py # analyse simulation
mamba deactivate