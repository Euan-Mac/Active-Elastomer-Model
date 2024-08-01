import cv2
import numpy as np
import gmsh
from os.path import join

# Function to compute shape tensor, and diagonalize it to get the principal axes
def compute_shape_tensor(coords):
    xs=coords[0,:] # load points
    ys=coords[1,:] 
    
    if len(xs)==2: # check points are correct size and shape
        Exception("Need more than 2 points to compute shape tensor - probably cords need transposed")
    
    xsN=xs-np.mean(xs) # subtract mean to centre points
    ysN=ys-np.mean(ys)
    
    # compute shape tensor
    Txx=np.sum(xsN**2)/len(xsN)
    Tyy=np.sum(ysN**2)/len(ysN)
    Txy=np.sum(xsN*ysN)/len(xsN)
    T=np.zeros((2,2))
    T[0,0]=Txx
    T[1,1]=Tyy
    T[0,1]=Txy
    T[1,0]=Txy

    # diagonalize shape tensor
    vals,vecs=np.linalg.eigh(T)

    return vals,vecs

# load q binary mask with opencv
def load_mask(mask_path):
    mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    mask=mask/255
    return mask

# get boundary of cell from mask with opencv
def get_boundary(mask):
    mask=mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundary=contours[0]
    return boundary

def load_scale(scale_path,scale_index):
    import json

    with open(scale_path,"r") as f:
        data=json.load(f)
    scale=data[scale_index]
    return scale

# Takes in a mesh built with gmsh, and rebuilds it using meshio ( easier to convert to xdmf)
def create_meshio_mesh(mesh, cell_type, prune_z=True):
    import meshio 
    cells = mesh.get_cells_type(cell_type) 

    cell_data = mesh.get_cell_data("gmsh:physical", cell_type) # load physical groups, these are used to tag the location of the lampellapodium
    points = mesh.points[:,:2] if prune_z else mesh.points # make sure we only have x,y coordinates
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]}) # create meshio mesh, details are to do with way xdmf files are written
    return out_mesh 

# Converts a gmsh mesh to xdmf format, by going through meshio using the create_meshio_mesh function
def convert_mesh(in_file,out_file):
        import meshio 
        from os.path import join
        
        msh = meshio.read(in_file+".msh") # read in mesh from gmsh file
        line_mesh = create_meshio_mesh(msh, "line", prune_z=True) # create line mesh representing boundary
        triangle_mesh = create_meshio_mesh(msh, "triangle", prune_z=True) # create triangle mesh representing cell interior

        meshio.write(join(out_file + "_mf.xdmf"), line_mesh) # write line mesh to xdmf file
        meshio.write(join(out_file + "_mesh.xdmf"), triangle_mesh) # write triangle mesh to xdmf file

     
mask_file=join("meshes","10_masks_0000.tif") # location of mask we want to mesh
out_file="./meshes/cell_mesh_new"
mask=load_mask(mask_file)
im_scale=118.63 # this is the size of the image in real units (microns), used to rescale each mask, so that the sizes are the same ratio as for real images

lc=0.4 # characteristic length scale for meshing
lc2=0.2 # characteristic length scale for meshing close to the boundary

boundary=get_boundary(mask) # locate closed counteour of cell mask
rescale_fac=0.25 # rescale system size (same for all cells, just makes domain a sensible size)

# rescale boundary
boundary=boundary*im_scale*rescale_fac/1024 # rescale boundary to match image size
boundary=(boundary-np.mean(boundary,axis=0)).squeeze() # centre boundary

mesh_coords=np.zeros_like(boundary) # invert y axis (to do with image coordinates, vs gmsh coordinates)
mesh_coords[:,0]=boundary[:,0]
mesh_coords[:,1]=-boundary[:,1]

# create msh
gmsh.initialize() # use gmsh library to create mesh
gmsh.model.add("mesh") # create model

# add all points on mesh boundary as isolated points
for i in range(len(mesh_coords)):
    x,y=mesh_coords[i,:]
    gmsh.model.geo.addPoint(x,y,0,lc,i+1)
  

it=np.argmin(mesh_coords[:,1]) # find index of point with smallest y value, as the lampellapodium is typically (we sometimes have to fchange this)

segs=[[],[]] # create two segments, one for the lampellapodium, one for the rest of the cell

for i in range(len(mesh_coords)):
    
    if i<it: # if we are before the lampellapodium
        segs[0].append(i+1) # add point to segment 0
    else: # if we are after the lampellapodium
        segs[1].append(i+1) # add point to segment 1
        
segs[0].append(segs[1][0]) # ensure boundary forms a closed loop
segs[1].append(1)

inds=[ [s-1 for s in seg] for seg in segs] # makes one continuous list of the indexes all all points in the segments
   

c1=gmsh.model.geo.addBSpline( segs[0],1) # fit a b-spline to both segments
c2=gmsh.model.geo.addBSpline( segs[1],2)

# add line loop
gmsh.model.geo.addCurveLoop([c1,c2],1) #  define a curve loop, which is a closed loop of curves, used to define boundary of enclosed region

# add plane surface
gmsh.model.geo.addPlaneSurface([1],1) # define a plane surface, which is a region enclosed by a curve loop

gmsh.model.geo.synchronize() # synchronize model (gmsh details)

# add physical surface
gmsh.model.addPhysicalGroup(2,[1],0) # athis just makes sure that the surface of the mesh is physical, as otherwsie fenics will not be able to read it

# add physical line
gmsh.model.addPhysicalGroup(1,[c1],0) # this is used to tag the lampellapodium, so that we can identify it in fenics, and apply BC to it
gmsh.model.addPhysicalGroup(1,[c2],1) # this is used to tag the rest of the cell, so that we can identify it in fenics, and apply a different BC to it


# This basically makes sure mesh is good resolution, and that we have a fine mesh near the boundary, where there may be more fine detail
gmsh.model.mesh.field.add("Distance", 1) 
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [c1,c2])
gmsh.model.mesh.field.setNumber(1, "Sampling", mesh_coords.shape[0]*10)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", lc2)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.01)
gmsh.model.mesh.field.setNumber(2, "DistMax", 1)

# set meshing options - generlly good practice to set these
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", lc2)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

# save all the meshing options
gmsh.model.geo.synchronize()
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# generate mesh
mesh=gmsh.model.mesh.generate(2)

# write mesh to file
gmsh.write(join(out_file+".msh"))

gmsh.fltk.run() # run gmsh gui to see mesh

convert_mesh(out_file,out_file) # convert mesh to xdmf format
