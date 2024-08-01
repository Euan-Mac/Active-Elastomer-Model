# Active-Elastomer-Model
Code used in our active elastomer in LECs paper for the modelling/theory side.

Here we have two main files entitled active_elastomer_model.py and fields_methods.py. The first contains all the classes and functions neede dto set-up and run fenics simulations of the active elastomer model as we do in the paper. We give two example scripts on ho to use this model in the files rect_sim.py and cell_12_sim.py respectively. One runs a periodic boundaries simulations of an active elastomer on a square domain, while the other runs a simulation with a real cell geometry and weak mixed boundary condtions (see manuscirpt). 

The other files fields_methods.py contains all the key functions used to analyse the output of the simulation, allowing things like the tracking of contracile regions, the computation of correlation functions, etc. Two smaller scripts are set-up to use these functions to analyse the output of the two above simulations, these are analysing_rect_sim.py for the PBC sqaure, and analysing_cell_sim.py for the cellular data. These produce a series of key plots and measurments in some output directories similar, to the plots used in the manuscirpt. 

To run these codes we reccomend first installing anaconda (https://www.anaconda.com/download), which allows us to use seperate python evironments or running and analysing the simulations. Note that if installing on apple silicon, you will need to install with the  x86_64 architecture. 

Once anaconda is installed we reccomened creating two seperate envronemnts. The first will be for running the FEM simulations in fenics and will need the following packages which are available to install from conda forge.
  - fenics
  - numpy
  - scipy
  - matplotlib (only used when some advanced functions are called so you may get away without this).

The second environment is used purely for prcoessing the output of the simulations and requires the following pakcages
  - numpy
  - scipy
  - matplotlib
  - pyvista
  - imageio.
  
