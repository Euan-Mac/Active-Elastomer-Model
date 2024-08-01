# Active-Elastomer-Model
Code used in our active elastomer in LECs paper for the modelling/theory side.

## Main files
We have two main files:
  - active_elastomer_model.py: Contains all the classes and functions needed to set up and run FEniCS simulations of the active elastomer model as described in the paper.
  - fields_methods.py: Contains key functions to analyze the output of the simulations, allowing tracking of contractile regions, computation of correlation functions, etc.

Two example scripts demonstrate how to use the active elastomer model:
  - rect_sim.py: Runs a periodic boundary simulation of an active elastomer on a square domain.
  - cell_10_sim.py: Runs a simulation with a real cell geometry and weak mixed boundary conditions (see manuscript).

Two examp,e scripts use the functions in fields_methods.py to analyze the output of the simulations:
  - analysing_rect_sim.py: Analyzes the output of the periodic boundary condition square simulation.
  - analysing_cell_sim.py: Analyzes the output of the cellular data simulation.

## Installation Instructions

We recommend first installing Anaconda, which allows us to use separate Python environments for running and analyzing the simulations. Note that if installing on Apple Silicon, you will need to install with the x86_64 architecture.

### Environment Setup

Create two separate environments:
  - FEM Simulations Environment, required packages (available from conda-forge):
    - fenics
    - numpy
    - scipy
    - matplotlib (only used for some advanced functions, so you may get away without this)
  - Processing Output Environment, required packages (available from conda-forge):
    - numpy
    - scipy
    - matplotlib
    - pyvista
    - imageio

## Mesh Generation

The active elastomer class can generate basic geometric meshes (lines, squares, and rectangles) internally, but not more complex shapes. To construct meshes of real cellular geometries, we use gmsh. A mask representing a real larval epithelial cell is given in the meshes directory. We also provide a script called meshing_cell_masks.py, which converts this mask into a FEniCS-compatible mesh.

To save the user from needing to do this, we also provide the output files from this script in the meshes directory (this is used to run the cell_10_sim.py simulation). Alternatively, users can use the meshing script to make the mesh themselves, with different settings, or adapt it to other images. To do this, a third conda environment will be needed with gmsh and OpenCV installed. The full list of required packages from conda-forge is:
  - python-gmsh (Note: the conda package named gmsh does not include the Python interface; you need python-gmsh)
  - opencv
  - meshio
  - numpy

We have provided a bash script called env_set_up.bash to help set up your conda environment.
