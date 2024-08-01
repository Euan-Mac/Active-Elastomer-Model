# Classes designed to run a fenics model of an active elastomer.
# Model involes a vector displacement field u and  a scalar concentration/density field phi.
# The model is desinged to use either external gmsh mesh files or to generate a simple mesh internally.

# Import libraries
import numpy as np
from numpy.random import rand
from time import time
from ufl import Index, nabla_div, exp, div, grad, indices, PermutationSymbol
from fenics import *
import sys


############### Small classes for the model ###############

# Simple fenics subdomain class which gets the entire boundary of the mesh as the subdomain
class Boundary_all(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


# Class for 2D PBCs, assuming a rectangle with (0,0) at one cornet  and opposite corner at (PP[0],PP[1])
class PeriodicBoundary2D(SubDomain):
    """
    A class representing a periodic boundary condition in 2D.

    Parameters:
    - PP: A tuple representing the size of the domain in the x and y directions.
    """

    def __init__(self, PP) -> None:
        super().__init__()
        self.PP = PP

    def inside(self, x, on_boundary):
        """
        Check if a point is inside the periodic boundary.

        Parameters:
        - x: A tuple representing the coordinates of the point.
        - on_boundary: A boolean indicating if the point is on the boundary.

        Returns:
        - A boolean indicating if the point is inside the periodic boundary.
        """
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], self.PP[1])) or
                          (near(x[0], self.PP[0]) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        """
        Map a point to its periodic image.

        Parameters:
        - x: A tuple representing the coordinates of the point.
        - y: A tuple to store the coordinates of the mapped point.
        """
        if near(x[0], self.PP[0]) and near(x[1], self.PP[1]):
            y[0] = x[0] - self.PP[0]
            y[1] = x[1] - self.PP[1]
        elif near(x[0], self.PP[0]):
            y[0] = x[0] - self.PP[0]
            y[1] = x[1]
        else:  # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - self.PP[1]


# Class for 1D PBCs, assuming a line with 0 at one end and PP at the other
class PeriodicBoundary1D(SubDomain):
    """
    A class representing a periodic boundary condition in 1D.

    Parameters:
    - PP: The periodicity length.

    Methods:
    - inside(x, on_boundary): Determines if a point is inside the boundary.
    - map(x, y): Maps a point from the original domain to the periodic domain.
    """

    def __init__(self, PP) -> None:
        super().__init__()
        self.PP = PP

    def inside(self, x, on_boundary):
        return (near(x[0], 0.0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - self.PP



class scalar_noisy_homogenous(UserExpression):
    """
    A class representing a 1D function which takes a homogeneous value + noise. 
    Used to set the initial conditions for the concentration field.

    Args:
        mean (float, optional): The mean value of the expression. Defaults to 1.

    Attributes:
        mean (float): The mean value of the expression.

    Methods:
        eval(values, x): Evaluates the expression at a given point.
        value_shape(): Returns the shape of the expression's value.

    """

    def __init__(self, mean=1, *args, **kwargs):
        self.mean = mean
        self.__init_subclass__
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        """
        Evaluates the expression at a given point.

        Args:
            values (numpy.ndarray): Array to store the evaluated value.
            x (float): The point at which to evaluate the expression.

        Returns:
            None
        """
        noise = 0.01 * (rand() - 0.5)
        if self.mean != 0:
            noise = self.mean * noise
        values[0] = self.mean + noise

    def value_shape(self):
        """
        Returns the shape of the expression's value.

        Returns:
            tuple: The shape of the expression's value.
        """
        return ()


class vector_noisy_homogenous(UserExpression):
    """
    A class representing a vector-valued noisy homogeneous function.
    Used to set the initial conditions for the displacement field.

    Args:
        mean (float): The mean value of the noise (default: 0).
        d (int): The dimension of the vector (default: 2).
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        d (int): The dimension of the vector.
        mean (float): The mean value of the noise.

    Methods:
        eval(values, x): Evaluates the function at a given point.
        value_shape(): Returns the shape of the function's values.

    """

    def __init__(self, mean=0, d=2, *args, **kwargs):
        self.d = d
        self.mean = mean
        self.__init_subclass__
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        for v in range(self.d):
            noise = 0.01 * (rand() - 0.5)
            if self.mean != 0:
                noise = self.mean * noise
            values[v] = self.mean + noise

    def value_shape(self):
        return (self.d,)

########### Mesh classes ###############
    
class mesh:
    """
    Represents a mesh object used in the active elastomer model. 
    Contains an actual mesh object (ie mesh.mesh) as well as other key info.

    Attributes:
        mpi (MPI.Comm): MPI communicator for parallel computing.
        PeriodicBoundary (None or object): Periodic boundary object.
        Periodic (bool): Flag indicating if the mesh has periodic boundaries.
        space_dim (None or int): Dimension of the mesh.
        mesh (None or object): FEniCS mesh object.
        mf (None or object): FEniCS mesh function for facet markers.
        cf (None or object): FEniCS mesh function for cell markers.
        bounds (list): List of boundary objects attached to the mesh.
    """

    def __init__(self):
        mpi_comm = MPI.comm_world
        num_processes = mpi_comm.size
        self.mpi = mpi_comm
        self.PeriodicBoundary = None
        self.Periodic = False
        self.space_dim = None
        self.mesh = None
        self.mf = None
        self.cf = None
        self.bounds = []



    def find_physical_boundary_verts(self):
            """
            Finds the indices of the vertices on the physical boundary of the mesh.

            Returns:
                np.array: Array of indices of the boundary vertices.
            """
            all_coords=self.mesh.coordinates()
            boundary_vertices_coords = BoundaryMesh(self.mesh, 'exterior').coordinates()

            boundary_vertices=[]
            for vert in vertices(self.mesh):
                vert_x=all_coords[vert.index(),0]
                vert_y=all_coords[vert.index(),1]
                for coord in boundary_vertices_coords:
                    if vert_x==coord[0] and vert_y==coord[1]:
                        boundary_vertices.append(vert.index())
            
            return np.array(boundary_vertices)
    
    def sort_boundary_vertices(self,boundary_vertices):
        """
        Sorts the boundary vertices of a mesh in a closed loop.

        Args:
            boundary_vertices (list): List of boundary vertices.
            mesh: The mesh object.

        Returns:
            tuple: A tuple containing the sorted boundary vertices and their corresponding coordinates.
        """
        mesh=self.mesh
        all_coords = mesh.coordinates()
        prev_vert = boundary_vertices[0]
        sorted_verts = [prev_vert]
        closed_loop = False

        while not closed_loop:
            stack = []
            for edge in edges(mesh):
                p1 = edge.entities(0)[0]
                p2 = edge.entities(0)[1]
                if p1 == prev_vert and p2 not in sorted_verts and p2 in boundary_vertices:
                    stack.append(p2)
                elif p2 == prev_vert and p1 not in sorted_verts and p1 in boundary_vertices:
                    stack.append(p1)

            if len(stack) == 0:
                closed_loop = True
            elif len(stack) == 1:
                sorted_verts.append(stack[0])
                prev_vert = stack[0]
            elif len(stack) > 1:
                min_dist = np.inf
                ind = 0
                for vert in stack:
                    new_x, new_y = all_coords[vert, 0], all_coords[vert, 1]
                    prev_x, prev_y = all_coords[prev_vert, 0], all_coords[prev_vert, 1]
                    dist = np.sqrt((new_x - prev_x) ** 2 + (new_y - prev_y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        ind = vert
                sorted_verts.append(ind)
                prev_vert = ind

        sorted_coords = mesh.coordinates()[sorted_verts, :]
        return sorted_verts, sorted_coords
    
    def find_and_sort_boundary_vertices(self):
        boundary_vertices=self.find_physical_boundary_verts()
        sorted_verts,sorted_coords=self.sort_boundary_vertices(boundary_vertices)
        return sorted_verts,sorted_coords
    
    def split_into_segements(self,num_segs=2,show_segs=False,wwindow_size=1):
        """
        Splits the boundary into segments. Segements are split at points of high curvature.


        Args:
            sorted_bound_inds (list): List of sorted boundary indices.
            num_segs (int, optional): Number of segments to split the boundary into. Defaults to 2.

        Returns:
            list: List of lists of boundary indices for each segment.
        """
        sorted_bound_inds,sorted_coords=self.find_and_sort_boundary_vertices()
        coords=self.mesh.coordinates()[sorted_bound_inds,:]
        curvature=np.roll(coords,-1,axis=0)+np.roll(coords,1,axis=0)-2*coords # compute curvature
        curvature=np.linalg.norm(curvature,axis=1) # compute norm of curvature
        
        # average curvature over a few points
        from scipy.signal import convolve
        smooth_curvature=convolve(curvature,np.ones(wwindow_size)/wwindow_size,mode="same")


        
        # sort points by curvature
        sorted_curv_inds=np.argsort(smooth_curvature)
        sorted_curv_inds=sorted_curv_inds[::-1]
        
        # want to reject corners which are too close together
        found_cuts=[sorted_curv_inds[0]]
        for ind in sorted_curv_inds:
            fail=False
            for ind2 in found_cuts:
                if abs(ind-ind2)<5:
                    fail=True
                    break
            if not fail:
                found_cuts.append(ind)
                
            if len(found_cuts)==num_segs:
                break

        # go through cuts (points of high curvature) list indices between cuts so they are in the same segment
        segs=[[] for i in range(num_segs)]
        point=found_cuts[0]
        done_cuts=[]
        ind=point
        n=0
        while len(done_cuts)<len(found_cuts):
            other_cuts=[i for i in found_cuts if i!=ind]
            while point not in other_cuts:
                segs[n].append(point)
                point=(point+1)%len(sorted_curv_inds)
            
            done_cuts.append(point)
            ind=point
            n+=1

        # show segements if required
        if show_segs:
            import matplotlib.pyplot as plt
            for n,seg in enumerate(segs):
                print(f"Segment {n}: {seg}")
                plt.plot(coords[seg,0],coords[seg,1],label=f"Segment {n}")
                # equal aspect ratio
                plt.gca().set_aspect('equal', adjustable='box')
                plt.legend()
            plt.show()
        return segs
    
    def vertex_list_to_edge_list(self,vertex_list):
        """
        Converts a list of vertices to a list of edges.

        Args:
            vertex_list (list): List of vertices.

        Returns:
            list: List of edges.
        """
        e_list=[]
        for n,v in enumerate(vertex_list):
            v_next=vertex_list[(n+1)%len(vertex_list)]
            vert_ob=Vertex(self.mesh,v)
            for e in edges(vert_ob):
                if v_next in e.entities(0):
                    e_list.append(e.index())
                    break
        
        return e_list

    def manually_mark_mark_edge_list(self,edge_list,mark):
        """
        Manually marks a list of edges.

        Args:
            edge_list (list): List of edges.
            mark (int): The marker to assign to the edges.
        """
        for e in edge_list:
            self.mf.array()[e]=mark

    def show_mesh_with_mf(self):
        """
        Shows the mesh with the mesh function.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        
        for e in edges(self.mesh):
            p1=e.entities(0)[0]
            p2=e.entities(0)[1]
            c1=self.mesh.coordinates()[p1,:]
            c2=self.mesh.coordinates()[p2,:]
            if self.mf[e.index()]==1:
                plt.plot([c1[0],c2[0]],[c1[1],c2[1]],color="r",ls="-")
            elif self.mf[e.index()]==2:
                plt.plot([c1[0],c2[0]],[c1[1],c2[1]],color="b",ls="-")
            elif self.mf[e.index()]==0:
                plt.plot([c1[0],c2[0]],[c1[1],c2[1]],color="k",ls="-")
        plt.show()

# Derived class for rectangular meshes
class rect_mesh(mesh):
    """
    A class representing a rectangular mesh.

    Args:
        Lx (float): Length of the mesh in the x-direction.
        Ly (float): Length of the mesh in the y-direction.
        Nx (int): Number of divisions in the x-direction.
        Ny (int): Number of divisions in the y-direction.
        Periodic (bool, optional): Whether the mesh has periodic boundary conditions. Defaults to False.
    """

    def __init__(self, Lx, Ly, Nx, Ny, Periodic=False):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.Periodic = Periodic
        self.space_dim = 2
        if self.Periodic:
            self.PeriodicBoundary = PeriodicBoundary2D((Lx, Ly))
        else:
            self.PeriodicBoundary = None
        self.mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), Nx, Ny)
        self.mf = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())


# Derived class for square meshes
class square_mesh(mesh):
    """
    A class representing a square mesh.

    Args:
        L (float): Length of the mesh in the x and y directions.
        N (int): Number of divisions in the x and y directions.
        Periodic (bool, optional): Whether the mesh has periodic boundary conditions. Defaults to False.
    """

    def __init__(self, L, N, Periodic=False):
        self = rect_mesh(L, L, N, N, Periodic)


# Derived class for 1D meshes
class line_mesh(mesh):
    """
    Represents a line mesh in 1D.

    Args:
        L (float): Length of the line mesh.
        N (int): Number of elements in the line mesh.
        Periodic (bool, optional): Whether the line mesh is periodic. Defaults to False.
    """
    def __init__(self, L, N, Periodic=False):
        super().__init__()
        self.L = L
        self.N = N
        self.Periodic = Periodic
        self.space_dim = 1
        if self.Periodic:
            self.PeriodicBoundary = PeriodicBoundary1D(L)
        else:
            self.PeriodicBoundary = None
        self.mesh = IntervalMesh(N, 0, L)
        self.mf = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.cf = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
  


# Class for gmsh meshes
class gmsh_mesh(mesh):
    """
    A class representing a Gmsh mesh.

    Args:
        path_to_mesh (str): The path to the Gmsh mesh files.

        Must be saved in the following format:
        - path_to_mesh_mesh.xdmf: The mesh file.
        - path_to_mesh_mf.xdmf: The mesh function file.

        Should have been converted from a gmsh msh file
        to and xdmf file before being passed to this class.

    Attributes:
        space_dim (int): The spatial dimension of the mesh.
        mesh (Mesh): The mesh object.
        mf (cpp.mesh.MeshFunctionSizet): The mesh function.
        cf (MeshFunction): The cell function.

    """
    def __init__(self, path_to_mesh):
        super().__init__()
        self.space_dim = 2
        mesh = Mesh()
        with XDMFFile(path_to_mesh + "_mesh.xdmf") as infile:
            infile.read(mesh)
        mvc = MeshValueCollection("size_t", mesh, 1)
        with XDMFFile(path_to_mesh + "_mf.xdmf") as infile:
            infile.read(mvc, "name_to_read")
        mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
        self.mesh = mesh
        self.mf = mf
        # self.cf = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())


########### Output Class ###############
class processer:
    """
    A class that processes and saves data for an active elastomer model.

    Args:
        model (object): The active elastomer model.
        freq (float, optional): The update frequency for saving data. Defaults to 1E-2.
        save_dir (str, optional): The directory to save the results. Defaults to "./results".
        extra_info (bool, optional): Flag to indicate whether to save extra information. Defaults to True.
    """

    def __init__(self, model, freq=1E-2, save_dir="./results", extra_info=True):
        from os.path import join, expanduser
        from shutil import rmtree
        from pathlib import Path
        from os import makedirs as mkdir

        self.update_interval = freq
        save_dir = expanduser(save_dir) # expand ~ to home directory
        self.save_dir = save_dir
        self.counter = 0 

        # delete old data and create new directories, makes sure we are not overwriting as this can cause issues
        print("Deleting old data...")
        sys.stdout.flush()
        rmtree(save_dir, ignore_errors=True)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # output - basic info we always output
        u_file = XDMFFile(join(save_dir, "u.xdmf"))
        u_file.parameters["flush_output"] = True
        phi_file = XDMFFile(join(save_dir, "rho_b.xdmf"))
        phi_file.parameters["flush_output"] = True

        self.u_file = u_file
        self.phi_file = phi_file
        self.v_div_file = XDMFFile(join(save_dir, "v_div.xdmf"))
        self.v_div_file.parameters["flush_output"] = True
        self.sources_file = XDMFFile(join(save_dir, "sources.xdmf"))
        self.sources_file.parameters["flush_output"] = True
        self.vel_file = XDMFFile(join(save_dir, "velocity.xdmf"))
        self.vel_file.parameters["flush_output"] = True

        # output mesh information and connectivity, useful for determining topology of the mesh in analysis
        mesh_file = XDMFFile(join(save_dir, "mesh.xdmf"))
        mesh_file.parameters["flush_output"] = True
        cell_file = XDMFFile(join(save_dir, "cell.xdmf"))
        cell_file.parameters["flush_output"] = True
        self.mesh_file = mesh_file
        self.cell_file = cell_file
        self.mesh_file.write(model.mesh.mesh)
        self.cell_file.write(model.mesh.mf)

        self.tensor_space = TensorFunctionSpace(model.mesh.mesh, "CG", 1)
        
        self.CG_file = XDMFFile(join(save_dir, "CG.xdmf"))
        self.CG_file.parameters["flush_output"] = True
        self.CG_file.parameters["functions_share_mesh"] = True
        self.CG_file.parameters["rewrite_function_mesh"] = False
        

        self.extra_info = extra_info
        self.model = model  # model and output are recursively referencing each other, not sure if this is good practice
        model.output = self

        # output extra information if required
        if extra_info:
            stress_file = XDMFFile(join(save_dir, "stress.xdmf"))
            stress_file.parameters["flush_output"] = True
            self.stress_file = stress_file
       
            self.passive_energy_file = XDMFFile(join(save_dir, "passive_energy.xdmf"))
            self.passive_energy_file.parameters["flush_output"] = True
            self.active_energy_file = XDMFFile(join(save_dir, "active_energy.xdmf"))
            self.active_energy_file.parameters["flush_output"] = True
            self.full_energy_file = XDMFFile(join(save_dir, "full_energy.xdmf"))
            self.full_energy_file.parameters["flush_output"] = True
            self.strain_file = XDMFFile(join(save_dir, "strain.xdmf"))
            self.strain_file.parameters["flush_output"] = True
            self.e_stress_file = XDMFFile(join(save_dir, "elastic_stress.xdmf"))
            self.e_stress_file.parameters["flush_output"] = True
            self.v_stress_file = XDMFFile(join(save_dir, "viscous_stress.xdmf"))
            self.v_stress_file.parameters["flush_output"] = True
            self.a_stress_file = XDMFFile(join(save_dir, "active_stress.xdmf"))
            self.a_stress_file.parameters["flush_output"] = True
            self.von_mises_file = XDMFFile(join(save_dir, "von_mises.xdmf"))
            self.von_mises_file.parameters["flush_output"] = True
            self.div_part_file = XDMFFile(join(save_dir, "div_part_v.xdmf"))
            self.div_part_file.parameters["flush_output"] = True
            self.curl_part_file = XDMFFile(join(save_dir, "curl_part_v.xdmf"))
            self.curl_part_file.parameters["flush_output"] = True
            self.u_div_file = XDMFFile(join(save_dir, "u_div.xdmf"))
            self.u_div_file.parameters["flush_output"] = True
            self.u_curl_file = XDMFFile(join(save_dir, "u_curl.xdmf"))
            self.u_curl_file.parameters["flush_output"] = True
            self.v_curl_file = XDMFFile(join(save_dir, "v_curl.xdmf"))
            self.v_curl_file.parameters["flush_output"] = True
            

            # helmholtz hodge decomposition
            # may be an issue with the boundary conditions here
            pot_el = FiniteElement("CG", self.model.mesh.mesh.ufl_cell(), 1)
            potential_space = FunctionSpace(self.model.mesh.mesh, MixedElement([pot_el, pot_el]))
            bound_all = Boundary_all()
            self.HH_bc = DirichletBC(potential_space.sub(1), Constant(0), bound_all)
            potentials = TrialFunction(potential_space)
            VP, SP = split(potentials)
            test_pots = TestFunction(potential_space)
            VQ, SQ = split(test_pots)

            u = (self.model.u_new - self.model.u_old) / self.model.dt
            i = Index()
            F1 = Dx(u[i], i) * SQ * dx + Dx(SP, i) * Dx(SQ, i) * dx
            F2 = Dx(u[1], 0) * VQ * dx - Dx(u[0], 1) * VQ * dx - Dx(VP, i) * Dx(VQ, i) * dx
            F = F1 + F2

            self.HH_a = lhs(F)
            self.HH_L = rhs(F)
            self.potentials = Function(potential_space)

    def write_params(self, dt, start_time=0):
        """
        Write the parameters of the model to a JSON file. 
        Used for reproducibility.

        Args:
            dt (float): The time step size.
            start_time (float, optional): The start time of the simulation. Defaults to 0. 
            This is here in case we do a equilibration run before the actual simulation.
            Which will probably not need the data saved.
        """
        import json
        from datetime import datetime
        from os.path import join

        world_comm = MPI.comm_world  # parallelization settings
        world_size = world_comm.Get_size()
        self.world_size = int(world_size)

        now = datetime.now() # current date and time
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        para = {}
        para["date"] = dt_string
        para["start_time"] = str(start_time)
        para["dt"] = str(dt)
        para["freq"] = str(self.update_interval) # freq is how often we save data

        # convert key info into dict
        for property, value in vars(self.model).items():
            if not property.startswith('__') and not callable(property) and type(value) in [float, int, str, bool]:
                para[property] = str(value)

        # write to json file
        with open(join(self.save_dir, "parameters.json"), "w") as outfile:
            json.dump(para, outfile)

    def out_data(self, ti):
        """
        Write data to output files.

        Parameters:
        - ti (float): Current time.

        Returns:
        None
        """
        # from os.path import join
        
        step_int = self.update_interval // self.model.dt # how many steps per save

        check = self.counter == step_int # check if we should save data, note that counter is incremented in evolve_time in model class
        if not check: # if not return 
            return
        
        self.counter = 0 # reset counter

        u_out = self.model.func_new.split()[0] # get displacement fields into a format we can write to file
        phi_out = self.model.func_new.split()[1]
        u_out.rename("u", "u") # rename fields
        phi_out.rename("rho_b", "rho_b")
        self.u_file.write(u_out, ti) # write to file
        self.phi_file.write(phi_out, ti)


        vel = (self.model.u_new - self.model.u_old) / Constant(self.model.dt) # compute velocity
        velocity = project(vel, self.model.function_space.sub(0).collapse()) # project onto a scalar function space, so we can write to file
        velocity.rename("velocity", "velocity") # rename
        self.vel_file.write(velocity, ti) # write to file

        # compute turnover rate and write to file
        div_v = div(vel)
        kr = Constant(self.model.k)
        a = Constant(self.model.a)
        rho0 = Constant(self.model.rho0)
        strain = self.model.strain_component(self.model.u_new)
        i = Index()
        sources = kr * phi_out * exp(a * strain[i, i]) + kr * rho0

        # write out sources and div_v
        dv = project(div_v, self.model.function_space.sub(1).collapse())
        ss = project(sources, self.model.function_space.sub(1).collapse())
        ss.rename("sources", "sources")
        dv.rename("div_v", "div_v")
        self.sources_file.write(ss, ti)
        self.v_div_file.write(dv, ti)
        
         # cauchy green strain tensor
        F = Identity(2) + grad(self.model.u_new)
        E = 0.5 * (F.T * F - Identity(2))
        TrE=E[i,i]
        detE=E[0,0]*E[1,1]-E[0,1]*E[1,0]
        lam1=TrE/2+sqrt(TrE**2/4-detE)
        lam2=TrE/2-sqrt(TrE**2/4-detE)
        
        CGs=project(E, self.tensor_space)
        l1s=project(lam1, self.model.function_space.sub(1).collapse())
        l2s=project(lam2, self.model.function_space.sub(1).collapse())
        
        l1s.rename("lambda1", "lambda1")
        l2s.rename("lambda2", "lambda2")
        CGs.rename("CG", "CG")
                
        self.CG_file.write(CGs,ti)
        self.CG_file.write(l1s,ti)
        self.CG_file.write(l2s,ti)

        # write out al the extra info
        # everything hs to be computed, then projected onto an appropriate function space, then written to file
        if self.extra_info:
            i, j = indices(2)

            elastic_stress = self.model.stress_func(Constant(self.model.B), Constant(self.model.mu),
                                                    self.model.u_new)
            viscous_stress = self.model.stress_func(Constant(self.model.eta_b), Constant(self.model.eta_s),
                                                    self.model.u_new - self.model.u_old)/Constant(self.model.dt)
            active_stress = self.model.active_stress_func(self.model.phi_new, Constant(self.model.zeta))
            passive_stess = elastic_stress + viscous_stress
            total_stress = elastic_stress + viscous_stress + active_stress
            total_stress_comp = total_stress[i, j]
            von_mis = sqrt(total_stress_comp * total_stress_comp * 3 / 2)
            passive_energy = strain[i, j] * passive_stess[i, j]
            active_energy = strain[i, j] * active_stress[i, j]
            total_energy = active_energy + passive_energy
            div_u = div(self.model.u_new)
            curl_u = Dx(self.model.u_new[1], 0) - Dx(self.model.u_new[0], 1)
            curl_v = Dx(vel[1], 0) - Dx(vel[0], 1)
            
           

            es = project(elastic_stress, self.tensor_space)
            vs = project(viscous_stress, self.tensor_space)
            as_ = project(active_stress, self.tensor_space)
            vm = project(von_mis, self.model.function_space.sub(1).collapse())
            pe = project(passive_energy, self.model.function_space.sub(1).collapse())
            ae = project(active_energy, self.model.function_space.sub(1).collapse())
            te = project(total_energy, self.model.function_space.sub(1).collapse())
            s = project(strain, self.tensor_space)
            ts = project(total_stress, self.tensor_space)
            du = project(div_u, self.model.function_space.sub(1).collapse())
            cu = project(curl_u, self.model.function_space.sub(1).collapse())
            cv = project(curl_v, self.model.function_space.sub(1).collapse())

            es.rename("elastic_stress", "elastic_stress")
            vs.rename("viscous_stress", "viscous_stress")
            as_.rename("active_stress", "active_stress")
            vm.rename("von_mis", "von_mis")
            pe.rename("passive_energy", "passive_energy")
            ae.rename("active_energy", "active_energy")
            te.rename("total_energy", "total_energy")
            s.rename("strain", "strain")
            ts.rename("stress", "stress")
            du.rename("div_u", "div_u")
            cu.rename("curl_u", "curl_u")
            cv.rename("curl_v", "curl_v")

            self.passive_energy_file.write(pe, ti)
            self.active_energy_file.write(ae, ti)
            self.full_energy_file.write(te, ti)
            self.strain_file.write(s, ti)
            self.e_stress_file.write(es, ti)
            self.v_stress_file.write(vs, ti)
            self.a_stress_file.write(as_, ti)
            self.von_mises_file.write(vm, ti)
            self.stress_file.write(ts, ti)

            self.u_div_file.write(du, ti)
            self.u_curl_file.write(cu, ti)
            self.v_curl_file.write(cv, ti)

            # helmholtz hodge decomposition, solve, project and write to file

            solve(self.HH_a == self.HH_L, self.potentials, self.HH_bc)

            # write out potentials
            V, P = self.potentials.split()
            div_part = project(grad(P), self.model.function_space.sub(0).collapse())
            div_part.rename("D", "D")

            curl_part_exp = as_vector([Dx(V, 1), -Dx(V, 0)])
            curl_part = project(curl_part_exp, self.model.function_space.sub(0).collapse())
            curl_part.rename("C", "C")

            self.div_part_file.write(div_part, ti)
            self.curl_part_file.write(curl_part, ti)


########### Main class ###############
class acitve_elastomer_model:
    """
    A class representing an active elastomer model.

    Parameters:
    - mesh (object): The mesh class representing the geometry.
    - dt (float): The time step size (default: 1E-2).
    - rand_seed (int): The random seed for numpy random generator (default: 1).
    - B (float): The bulk modulus (default: None).
    - mu (float): The shear modulus (default: None).
    - eta_b (float): The bulk viscosity (default: None).
    - eta_s (float): The shear viscosity (default: None).
    - zeta (float): The active stress coefficient (default: None). Note that this is defined with a sign flip compared to manuscript.
    - D (float): The diffusion coefficient (default: None).
    - k (float): The rate of myosin turnoer (default: None).
    - tau (float): The timescale of myosin turnover (default: None).
        Should only specify one of k or tau.
    - a (float): Dimensionless constant for feedback due to strain dependent unbinding of myosin (default: None).
    - rho0 (float): The preferred myosin density (default: None).
    """

    def __init__(self, mesh, dt=1E-2, rand_seed=1, B=None, mu=None, eta_b=None,
                 eta_s=None, zeta=None, D=None, k=None, tau=None, a=None, rho0=None):

        self.mesh = mesh

        self.function_space = None # mixed function space
        self.func_new = None # functions for current time step
        self.func_old = None # functions for previous time step
        self.tests = None # test functions

        self.phi_new = None # myosin at current time step
        self.phi_old = None # myosin at previous time step
        self.phi_test = None # test function for myosin
        self.u_new = None # displacement at current time step
        self.u_old = None # displacement at previous time step
        self.u_test = None # test function for displacement

        self.dri_bcs = [] # list of dirichlet boundary conditions
        self.residual = None # residual of the weak form
        self.solver = None # solver for the weak form

        self.spring_const = 0 # spring constant for the boundary

        self.dt = dt # time step size
        self.time = 0 # current time
        self.N = 0  # current time step

        self.B = B 
        self.mu = mu 
        self.eta_b = eta_b
        self.eta_s = eta_s
        self.zeta = zeta # again note that this is defined with a sign flip compared to manuscript
        self.D = D
        self.gamma = 1 # external friction coefficient, set to 1 automatically as we usually use that system of units. But here so that user can change it if they want 
        
        if k is None: # set k or tau
            self.k = 1 / tau
        elif tau is None:
            self.k = k
        else:
            raise ValueError("Either k or tau must be specified")
        
        self.a = a
        self.rho0 = rho0

        self.rand_seed = rand_seed # random seed for numpy random generator
        np.random.seed(self.rand_seed)
        
        self.MPI = MPI.comm_world # MPI communicator for parallel computing

        self.output = None # output object which we will create later
        set_log_level(30) # set log level to 30 so that we don't get any output from fenics

    # Spring constant for the boundary when we want springlike boundary conditions, note that myosin boundaries will notbe moved.
    # So this setting is not reccomended. Also note that any point which have dri_bcs will not have the spring-like ones applied. 
    def set_spring_const(self, k):
        self.spring_const = k

    # Set the initial conditions for the model, to homogenous steady states + noise
    def steady_states_init_conds(self):
        if self.rho0 == 0 or self.rho0 is None:
            phi_mean = 1
        else:
            phi_mean = self.rho0
        u_mean = 0
        if self.mesh.space_dim == 1:
            IC_u = scalar_noisy_homogenous(mean=u_mean) # create instances of expressin classes for the initial conditions
        else:
            IC_u = vector_noisy_homogenous(mean=u_mean)
        IC_phi = scalar_noisy_homogenous(mean=phi_mean)
        self.set_initial_conds(IC_u, IC_phi) # use below unction to assign functions to the created expressions

    # See above
    def set_initial_conds(self, u0_f, phi0_f):
        u0 = interpolate(u0_f, self.function_space.sub(0).collapse())
        phi0 = interpolate(phi0_f, self.function_space.sub(1).collapse())
        assign(self.func_old, [u0, phi0])
        assign(self.func_new, [u0, phi0])



    def set_Drichelet_BC(self, bound_val, bound_region, field_ind):
            """
            Set Dirichlet boundary conditions for a given field.

            Parameters:
            bound_val (float): The value of the boundary condition. Eg 1 for a scalar field or [0,0] for a vector field.
            bound_region (dolfin.MeshFunction): The fenics subdomain representing the boundary.
            field_ind (int): The index of the field. 0 for displacement, 1 for myosin.
            Returns:
            None
            """

            bcs_in = DirichletBC(self.function_space.sub(field_ind), bound_val, bound_region) # create fenics boundary condition object
            self.dri_bcs.append(bcs_in) # add to list of boundary conditions

    def set_Drichelet_BC_mf(self, bound_val, bound_region, field_ind):
            """
            Set Dirichlet boundary conditions for a given field. Using the mesh function to specify the boundary. 
            Instead of a subdomain.

            Parameters:
            bound_val (float): The value of the boundary condition. Eg 1 for a scalar field or [0,0] for a vector field.
            bound_region (dolfin.MeshFunction): The integar value (of the mf) which we want to assign the boundary condition to.
            field_ind (int): The index of the field. 0 for displacement, 1 for myosin.
            Returns:
            None
            """

            bcs_in = DirichletBC(self.function_space.sub(field_ind), bound_val, self.mesh.mf, bound_region)
            self.dri_bcs.append(bcs_in)

    def intialise_function_space(self, degree=1, elements=["CG", "CG"]):
            """
            Initializes the function space for the active elastomer model.

            Parameters:
            - degree (int): The degree of the finite element basis functions. Default is 1.
            - elements (list): The type of finite elements to use for each component of the function space. 
              The first element corresponds to the displacement field, and the second element corresponds to the phase field.
              Default is ["CG", "CG"].

            Returns:
            None
            """
          
            # create elements for the function space
            P1 = VectorElement(elements[0], self.mesh.mesh.ufl_cell(), degree, dim=self.mesh.space_dim)
            P2 = FiniteElement(elements[1], self.mesh.mesh.ufl_cell(), degree)

            if self.mesh.space_dim == 1: # if 1D mesh, we need to use a scalar element for both as paraview doesn't like 1D vector fields
                E = MixedElement([P2, P2])
            else:
                E = MixedElement([P1, P2]) # otherwise use vector element for displacement and scalar for myosin

            if self.mesh.Periodic: # if we have periodic boundary conditions, we need to use a constrained function space, uses classes at top of file
                self.function_space = FunctionSpace(self.mesh.mesh, E, constrained_domain=self.mesh.PeriodicBoundary)
            else:
                self.function_space = FunctionSpace(self.mesh.mesh, E)

            func_new = Function(self.function_space) # create functions for current and previous time steps
            func_old = Function(self.function_space)
            self.func_new = func_new 
            self.func_old = func_old

            self.u_new, self.phi_new = split(func_new) # split functions into displacement and myosin
            self.u_old, self.phi_old = split(func_old)

            self.tests = TestFunction(self.function_space) # create test functions
            self.u_test, self.phi_test = split(self.tests)

    # Defines which timestep to use for the RHS of the weak form,
    # theta=0.5 is Crank-Nicolson, theta=1 is backward Euler, theta=0 is forward Euler.
    # 1 is recommended for stability
    def star_func(self, new_func, old_func, theta):
        return theta * new_func + (1 - theta) * old_func

    # Defines the strain tensor
    def strain_component(self, u_star):
        i, j = indices(2)
        strain_tens = 0.5 * (Dx(u_star[i], j) + Dx(u_star[j], i))
        strain_tensor = as_tensor(strain_tens, (i, j))
        return strain_tensor

    # Defines the stress tensor for elastic or viscous stress
    def stress_func(self, B, mu, u_star):
        i, j, k = indices(3)
        delta = Identity(self.mesh.space_dim)
        strain = self.strain_component(u_star)
        s_ij = B * Dx(u_star[k], k) * delta[i, j] + 2 * mu * (strain[i, j] - 0.5 * delta[i, j] * Dx(u_star[k], k))
        s = as_tensor(s_ij, (i, j))
        return s

    # Defines the active stress tensor
    def active_stress_func(self, phi_star, s): # note this is defined with a sign flip compared to the manuscript
        i, j = indices(2)
        delta = Identity(self.mesh.space_dim)
        act_str_ij = - delta[i, j] * s * phi_star / (1 + phi_star)
        act_str = as_tensor(act_str_ij, (i, j))
        return act_str

    # Defines the weak form of the model
    def create_weak_from(self, theta=1):

        # We multiply by dt so do not divide dot by dt
        phi_dot = self.phi_new - self.phi_old # time derivative of myosin
        u_dot = self.u_new - self.u_old # time derivative of displacement

        # We use the theta method to define the RHS of the weak form
        phi_star = self.star_func(self.phi_new, self.phi_old, theta)
        u_star = self.star_func(self.u_new, self.u_old, theta)

        # Define the constants, note that we use the Constant class from fenics for speed
        B = Constant(self.B)
        mu = Constant(self.mu)
        eta_b = Constant(self.eta_b)
        eta_s = Constant(self.eta_s)
        zeta = Constant(self.zeta) # note that this is defined with a sign flip compared to the manuscript
        D = Constant(self.D)
        kr = Constant(self.k)
        a = Constant(self.a)
        rho0 = Constant(self.rho0)
        delta_t = Constant(self.dt)
        gamma = Constant(self.gamma)

        # Define indices for einsum notation
        i, j, k = indices(3)


        ds = Measure('ds', domain=self.mesh.mesh, subdomain_data=self.mesh.mf)
        dx = Measure('dx', domain=self.mesh.mesh, subdomain_data=self.mesh.mf)
        n = FacetNormal(self.mesh.mesh)
        h = CellDiameter(self.mesh.mesh)

        # residual for myosin
        phi_current = phi_star * u_dot[k] - delta_t * D * Dx(phi_star, k) # advection + diffusion term 
        phi_res = (
                phi_dot * self.phi_test * dx
                - phi_current * Dx(self.phi_test, k) * dx
        ) 

        ep = Dx(u_star[k], k) # trace of strain tensor
        source_terms = -delta_t * kr * (phi_star * exp(a * ep) - rho0) * self.phi_test # terms from turnover

        # add turnover terms to residual
        if self.k != 0: 
            phi_res += -source_terms * dx

        elastic_stress = self.stress_func(B, mu, u_star)
        viscous_stress = self.stress_func(eta_b, eta_s, u_dot)
        active_stress = self.active_stress_func(phi_star, zeta)

        stress_tens = delta_t * elastic_stress + viscous_stress + delta_t * active_stress # stress tensor

        # residual for displacement
        u_res = (
                gamma * u_dot[i] * self.u_test[i] * dx
                + stress_tens[i, j] * Dx(self.u_test[i], j) * dx
        )

        # add spring-like boundary conditions, these will only be applied to points which do not have dirichlet boundary conditions
        if self.spring_const != 0:
            u_res += delta_t * self.spring_const * u_star[k] * self.u_test[k] * ds

        # total residual
        self.residual = u_res + phi_res


    # Solve the weak form to get the displacement and myosin at the next time step
    def evolve_time(self, T, do_print=True):
        world_comm = self.MPI # parallelization settings
        my_rank = world_comm.Get_rank() # get rank of current process, so that we don't print multiple times, etc
        

        # Write out simulation parameters to file for reproducibility
        self.output.write_params(self.dt, start_time=self.time) 
        out_time = time()
        out_step = 0

        # Load key info into local variables for speed
        bcs = self.dri_bcs
        res = self.residual
        func_new = self.func_new

        dx = Measure('dx', domain=self.mesh.mesh, subdomain_data=self.mesh.mf)
        mesh_area = assemble(Constant(1) * dx) # compute area of mesh
        
        for step in range(int(T / self.dt) + 1):  # (for lop better than while loop in parallel )
            t = step * self.dt # current time
            self.output.counter += 1 # increment counter for output object
            self.func_old.vector()[:] = func_new.vector() # set prev function to current function

            if len(bcs) == 0: # if no boundary conditions, solve without bcs
                solve(res == 0, func_new)
            else:
                solve(res == 0, func_new, bcs)

            # Write out data. Do this every time step and processor class will hand theoutput frequency part
            if self.output is not None:
                self.output.out_data(t + self.time)

            # Print out info if required
            if t % 1 < 0.1 * self.dt and do_print and t > 0:

         
                total_myo = assemble(self.phi_new * dx) # compute total myosin

                if my_rank == 0: # only print if we are on the first process, to avoid multiple prints
                    print(f"Time: {t + self.time:n}, Run time per step: {(time() - out_time) / (step - out_step):.3f}, Myo per unit area: {float(total_myo)/float(mesh_area): .3f}")
                    out_time = time() # reset time and step
                    out_step = step 

                sys.stdout.flush() # flush stdout to avoid issues with printing in parallel

        self.time += T # At end of simulation, update time, in case we want to run another simulation after this one