import active_elastomer_model as ag

out_file="./rect_sim_out" # place to save output

T=500 # final time
dt = 5E-2 # time step

Bulk=1 # Bulk modulus
mu=Bulk # Shear modulus
eta_b=1 # bulk viscosity
eta_s=eta_b # shear viscosity
zeta1=-35 # activity parameter - note this is negative because in the code the activity is defined with a mnius sign compared to the paper

k=0.25 # turnover rate
rho0=1 # rest density
a=-1 # strain dependent unbinding parameter
D=1 # diffusion constant

mesh_ob=ag.rect_mesh(20,20,50,50,Periodic=True) # create a rectangular mesh, change periodic to False if you want to use Dirichlet boundary conditions
# Create an instance of the active elastomer model class used to do the simulation
s=ag.acitve_elastomer_model(mesh_ob,dt=dt,B=Bulk,zeta=zeta1,
k=k,rho0=rho0,mu=mu,eta_s=eta_s,eta_b=eta_b,a=a,D=D,rand_seed=1)

s.intialise_function_space(degree=1,elements=["CG","CG"]) # intialise the FEM function space (use CG elements of degree 1)

# uncomment the following lines to use Dirichlet boundary conditions on displacement field, and no-flux boundary conditions on the myosin field
# bd_all=ag.Boundary_all() # create a boundary object that is just the boundry of the entire domain
# s.set_Drichelet_BC((0,0),bd_all,0) # create a Dirichlet boundary condition for the displacement field over the entire boundary

s.steady_states_init_conds() # Set the intial conditions to be the steady state solution + noise

s.create_weak_from(theta=1) # Create the weak form of the PDEs

# Create an outputter object to save the results, saves results every freq units of time.
outputter=ag.processer(s,freq=0.1,save_dir=out_file,extra_info=False) # Note we create the output directory and if it already exists we will overwrite it

s.evolve_time(T,do_print=True) # Evolve the system in time for total time T, printing progress to the screen

