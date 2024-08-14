### Set up the environments for running the full workflow ###

# Environment set up for FEniCS simulations
conda create -n fenics -c conda-forge fenics numpy scipy matplotlib

# Envronment set-up for analysis of fenics simulations
conda create -n fenics_analysis -c conda-forge imageio vtk=9.2.6 pyvista=0.41.1 python=3.11
conda activate fenics_analysis 
conda install -c condo-forge  numpy scipy matplotlib
# easier for conda to install things in two steps, so we install the complex stuff first and then the easy stuff, this will probably still take a while to run

# Environment set-up for mesh-making
conda create -n fenics_mesh -c conda-forge python-gmsh opencv meshio numpy

### Then to run a full workflow we would do the following ###

conda activate fenics-mesh
python meshing_cell_masks.py # make mesh
conda deactivate

conda activate fenics
mpirun -np 2 python cell_10_sim.py # run simulation in parallel with 2 processors
conda deactivate

conda activate fenics_analysis
python example_code/analysing_cell_sim.py # analyse simulation
conda deactivate