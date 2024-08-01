import numpy as np



##################################################################################################
# This section contains functions for loading data from an active elastomer simulation
##################################################################################################

# Loads a vector field from a fenics xdmf file, takes the divergence of the field and outputs
def XDMF_conversion_div(filename, name, dt=1, do_print=False, max_time=1E12):
    """
    Convert XDMF file to divergence fields.

    Args:
        filename (str): The path to the XDMF file.
        name (str): The name of the field to within the xdmf file. (There can be multiple fields in one file)
        dt (float, optional): The time step used in the simulation. Defaults to 1.
        do_print (bool, optional): Whether to print progress messages. Defaults to False.
        max_time (float, optional): The maximum time value to process up to (files do not dontain 
            information on last time so if this is too large you will get repeated fields). Defaults to 1E12.

    Returns:
        tuple: A tuple containing the coordinates and divergence of the field. As a function of time.
    """
    import pyvista as pv

    reader = pv.get_reader(filename)
    all_fields = []
    time_val = 0
    success = True
    data = reader.read()
    data2 = data.compute_derivative(name, divergence=True)
    field_prev = data2.get_array("divergence") # take divergence of field
    if len(field_prev.shape) > 1:
        field_prev = field_prev.transpose(1, 0)

    while success:
        reader.set_active_time_value(time_val)
        data = reader.read()
        data2 = data.compute_derivative(name, divergence=True)
        time_val += dt

        if do_print:
            print(f"Data at t step {time_val} found in file: " + filename, end='\r')
        field = data2.get_array("divergence")
        if len(field.shape) > 1:
            field = field.transpose(1, 0)
        if np.all(field_prev == field) and max_time == 1E12:
            print("Same field found twice")
            success = False
        elif time_val >= max_time:
            success = False
        else:
            all_fields.append(field)
            field_prev = np.copy(field)

    all_fields = np.array(all_fields)
    coords = data.points
    return coords, all_fields

# Loads a scalar field from a fenics xdmf file - works almost exactly the same as the above function
def XDMF_conversion(filename,name,dt=1,do_print=False,max_time=1E12):
        import pyvista as pv

        reader=pv.get_reader(filename)
        all_fields=[]
        time_val=0
        success=True
        data=reader.read()
        field_prev=data.get_array(name)*0
        if len(field_prev.shape)>1:
            field_prev=field_prev.transpose(1,0)

        while success:
            reader.set_active_time_value(time_val)
            time_val+=dt
            data=reader.read()
        
            if do_print:
                print(f"Data at t step {time_val} found in file: " + filename, end='\r')
            field = data.get_array(name)
            if len(field.shape)>1:
                field=field.transpose(1,0)
            if np.all(field_prev==field) and max_time==1E12:
                print("Same field found twice")
                success=False
            elif time_val>=max_time:
                success=False
            else:
                all_fields.append(field)
                field_prev=np.copy(field)
        
        all_fields=np.array(all_fields)
        coords=data.points
        return coords, all_fields
# Load a vector field from a fenics xdmf file
def v_XDMF_conversion(filename,name,dt=1,do_print=False,max_time=1E12):
    import pyvista as pv

    reader=pv.get_reader(filename)
    all_fields=[]
    time_val=0
    success=True
    data=reader.read()
    init_dat=data.get_array(name)
    field_prev=np.zeros((init_dat.shape[1],init_dat.shape[0]))

    while success:
        reader.set_active_time_value(time_val)
        time_val+=dt
        data=reader.read()
        
        if do_print:
            print(f"Data at t step {time_val} found in file: " + filename)
        field = data.get_array(name)
        field = field.transpose(1,0)
        if np.all(field_prev==field) and max_time==1E12:
            print("Same field found twice")
            success=False
        elif time_val>=max_time:
            success=False
        else:
            all_fields.append(field)
            field_prev=np.copy(field)
        
    all_fields=np.array(all_fields)
    coords=data.points

    return coords, all_fields

# load parameters files which simulations write out
# these are written in json format
# contains key info like time step, start time, parameters for the simulation3
def load_AE_parameters(params_file):
    import json
    with open(params_file) as f:
        params = json.load(f)
    return params

# Helper function to use the above functions to load a vector or scalar field from a fenics xdmf file using the above functions
def load_fenics_field_xdmf(data_direc, field_name, assigned_name, do_print=True, max_time=1E12):
    """
    Load a Fenics field from an XDMF file.

    Parameters:
    - data_direc (str): The directory where the data is located.
    - field_name (str): The name of the field to load.
    - assigned_name (str): The name to assign to the loaded field.
    - do_print (bool): Whether to print progress information. Default is True.
    - max_time (float): The maximum time to load. Default is 1E12.

    Returns:
    - times (ndarray): An array of time values.
    - coords (ndarray): An array of coordinate values.
    - fields (ndarray): An array of field values.
    """
    import glob
    from os.path import join, expanduser

    params = load_AE_parameters(join(data_direc, 'parameters.json'))
    file_name = expanduser(join(data_direc, field_name + ".xdmf"))
    output_gap = float(params['freq'])

    load_func = lambda x: XDMF_conversion(x, assigned_name, dt=output_gap, do_print=do_print, max_time=max_time)
    coords, fields = load_func(file_name)

    num_times = fields.shape[0]
    start_time = float(params['start_time'])

    times = np.arange(start_time, start_time + output_gap * num_times, output_gap)

    return times, coords, fields


# Helper function to use the above functions to load a the divergence of a vector field from a fenics xdmf file using the above functions
def load_field_div(data_direc,field_name,assigned_name,do_print=True,max_time=1E12):
    import glob
    from os.path import join, expanduser
    
    params=load_AE_parameters(join(data_direc, 'parameters.json'))
    file_name=expanduser(join(data_direc, field_name+".xdmf"))
    output_gap = float(params['freq'])
    
    load_func=lambda x: XDMF_conversion_div(x,assigned_name,dt=output_gap,do_print=do_print,max_time=max_time)
    
    coords,fields=load_func(file_name)
    start_time=float(params['start_time'])
    
    times=np.arange(start_time,start_time+fields.shape[0],1)
    
    return times,coords,fields
    
##################################################################################################
# This section contains functions for dealing with meshes from fenics #
##################################################################################################

# Function to load a mesh from a fenics xdmf file, then output the points on the boundary
def get_msh_boundary_points(filename):
    import pyvista as pv
   
    # Load mesh and number of points
    reader=pv.get_reader(filename)
    mesh_dat=reader.read()
    N=mesh_dat.number_of_points
    
    point_counts=np.zeros(N,dtype=int) # an array to count how many times each point is connected to by a cell
    for i in range(mesh_dat.number_of_cells): # loop over all cells
        cell=mesh_dat.get_cell(i)
        for j in range(cell.n_points): # loop over all points in the cell
            point=cell.points[j] # get the point 
            point_mask=np.all(mesh_dat.points==point,axis=1) # relate to point itself to an index
            point_counts[point_mask]+=1 # increment the count of the point

    bound_mask=point_counts<4 # points on the boundary are connected to less than 4 cells
    bound_pts=mesh_dat.points[bound_mask,:] # get the points on the boundary
    return bound_pts

# Get the connectivity of a mesh from a fenics xdmf file
def get_msh_conec(filename,PBC=False):
    import pyvista as pv 
    
    # Load mesh and number of points then iterate over all points and get the points connected to it
    # This gives us an array where the nth element is an array of the points connected to the nth point
    reader=pv.get_reader(filename) 
    mesh_dat=reader.read()
    N=mesh_dat.number_of_points
    connecs=[mesh_dat.point_neighbors(i) for i in range(N)]
    
    
    if PBC: # if periodic boundary conditions are used, we need to add the points connected via the PBC
        points=mesh_dat.points
        x_max=np.max(points[:,0]) # get points sitting on the boundary
        x_min=np.min(points[:,0])
        y_max=np.max(points[:,1])
        y_min=np.min(points[:,1])
        
        min_x_points_mask=(points[:,0]==x_min) # get the points on the left and bottom boundaries
        min_y_points_mask=(points[:,1]==y_min)
        for p in range(N): # loop over all points
            if min_x_points_mask[p]: # if on left
                # look for point with same y and sitting on x_max, and add to connectivity
                for q in range(N):
                    if points[q,0]==x_max and points[q,1]==points[p,1]:
                        connecs[p].append(q)
                        connecs[q].append(p)
            if min_y_points_mask[p]: # if on bottom
                # look for point with same x and sitting on y_max, and add to connectivity
                for q in range(N):
                    if points[q,1]==y_max and points[q,0]==points[p,0]:
                        connecs[p].append(q)
                        connecs[q].append(p)    
        
    return connecs

##################################################################################################
# General functions to run on unsturcutred arrays which are useful #
##################################################################################################

# ray casting algorithm to check if point is in hull - chapt gpt code
def point_in_hull(x,y,polygon):
    n = len(polygon)
    odd_nodes = False
    yi = polygon[:, 1]
    yj = np.roll(yi, -1)  # Equivalent to "shifting" the y-values
    xi = polygon[:, 0]
    xj = np.roll(xi, -1)
    condition = ((yi < y) & (yj >= y)) | ((yj < y) & (yi >= y))
    intersect = xi + (y - yi) / (yj - yi + np.finfo(float).eps) * (xj - xi) < x # draw straight line from point to infinity and count number of intersections
    odd_nodes = np.count_nonzero(condition & intersect) % 2 == 1
    return odd_nodes

# If we we have a time series of masks, we want the intersection masks at all times
def put_image_stack_in_rect(im_stack,Xs,Ys):
    overall_mask=np.all(im_stack.mask,axis=0) # get the overall mask
    max_x=np.max(Xs[~overall_mask]) # max and min x and y values of the points not in the overall mask
    min_x=np.min(Xs[~overall_mask])
    max_y=np.max(Ys[~overall_mask])
    min_y=np.min(Ys[~overall_mask])

    # get the points in the rectangle formed by the above points
    inside_inds=(min_x<=Xs) & (Xs<=max_x) & (min_y<=Ys) & (Ys<=max_y) 
    new_im=im_stack[:,inside_inds]
    new_Xs=Xs[inside_inds]
    new_Ys=Ys[inside_inds]
    num_x=np.unique(new_Xs).shape[0] # get the number of unique x and y values which are inside the rectangle
    num_y=np.unique(new_Ys).shape[0]
    
    # reshape the image stack into a rectangular shape
    new_im=new_im.reshape((new_im.shape[0],num_x,num_y)) 
    new_Xs=new_Xs.reshape((num_x,num_y))
    new_Ys=new_Ys.reshape((num_x,num_y))
    return new_im,new_Xs,new_Ys

# Takes an image and scale it down by a factor
def rescale_image(im,factor=8):
    from scipy.signal import convolve
    averging_box=np.ones((factor,factor)) # kernel we use to average the image
    av_frames=convolve(im,averging_box,mode="same")/factor**2 # convolve the image with the kernel to do a local spatial average
    av_frames=av_frames[::factor,::factor] # take every factorth point in x and y (It now contains the average of factor^2 points)
    return av_frames # return the rescaled image

# Just applied previous function to a time dependent image stack
def rescale_im_stack(im_stack,factor=8):
    im0=rescale_image(im_stack[0,:,:],factor=factor)
    sizex=im0.shape[0]
    sizey=im0.shape[1]
    im_stack_out=np.zeros((im_stack.shape[0],sizex,sizey))
    for i in range(im_stack.shape[0]):
        im_stack_out[i,:,:]=rescale_image(im_stack[i,:,:],factor=factor)

    return im_stack_out

# Function to compute the shape tensor of a set of points,
# diagonalise it and return the eigenvalues and eigenvectors in order of size
def compute_shape_tensor(coords):
    xs=coords[0,:] # get the x and y coordinates of the points
    ys=coords[1,:]
    
    if len(xs)==2:
        Exception("Need more than 2 points to compute shape tensor - probably coords need transposed")
    
    xsN=xs-np.mean(xs) # subtract the mean from the x and y coordinates to centre the points
    ysN=ys-np.mean(ys)
    
    Txx=np.sum(xsN**2)/len(xsN) # compute the elements of the shape tensor
    Tyy=np.sum(ysN**2)/len(ysN)
    Txy=np.sum(xsN*ysN)/len(xsN)
    T=np.zeros((2,2))
    T[0,0]=Txx
    T[1,1]=Tyy
    T[0,1]=Txy
    T[1,0]=Txy

    vals,vecs=np.linalg.eigh(T) # diagonalise the shape tensor

    norm_vecs=np.zeros((2,2)) # normalise the eigenvectors
    norm_vecs[:,0]=vecs[:,0]/np.linalg.norm(vecs[:,0])
    norm_vecs[:,1]=vecs[:,1]/np.linalg.norm(vecs[:,1])

    vecs1=norm_vecs[:,0] 
    vecs2=norm_vecs[:,1]
    vals1=vals[0]
    vals2=vals[1]

    if vals1>vals2:
        return vals1,vals2,vecs1,vecs2
    else:
        return vals2,vals1,vecs2,vecs1

##################################################################################################
# Functions for computing correlation functions etc #
##################################################################################################
"""
Generally in this section we assume fields at 2d arrays for scalr fields, where the first index is time, and the second is space.
"""

# Function to compute time correlation function between two fields, with errors computed using bootstrapping
# Most be evenly spaced in time
def bootstrap_time_correl_FT(field1,field2,times,N_boot=1000,do_print=True):
    valid_points1=~np.isnan(field1) # Find points in space which are valid for both fields at all times
    valid_points2=~np.isnan(field2)
    mask=np.all(valid_points1,axis=0) | np.all(valid_points2,axis=0)
    
    field_valid1=field1[:,mask] # Get the values of the fields which are at these points
    field_valid2=field2[:,mask]
    num_valid_points=field_valid1.shape[1]

    for i in range(N_boot): # Loop over the number of bootstraps
        rand_inds=np.random.choice(np.arange(num_valid_points),num_valid_points,replace=True) # Randomly select some points
        f1=field_valid1[:,rand_inds] # Get the values of the fields at these points
        f2=field_valid2[:,rand_inds]
        corls_now,dts=average_time_correl_FT(f1,f2,times) # Compute the time correlation function of these two spatially resampled fields
        if i==0:
            corls=np.zeros((N_boot,corls_now.shape[0])) # Initialise the array to store the correlation functions
        corls[i,:]=corls_now # Store the correlation function

        if do_print: # Print progress
            print(f"Bootstrapping time correlation {100*(i+1)/N_boot:.1f}%",end='\r')
    
    corls_out=np.nanmean(corls,axis=0) # output is average over bootstraps
    corls_std=np.nanstd(corls,axis=0) # standard deviation of the bootstraps
    corls_std_err=corls_std/np.sqrt(N_boot) # standard error of the bootstraps
    return corls_out,corls_std_err,dts # return the correlation function and its error

# function to compute the time correlation function of a field on an unstructured grid using the FFT
# Points must be evenly spaced in time
def average_time_correl_FT(field1,field2,times):
    from scipy.signal import correlate,correlation_lags

    valid_points1=~np.isnan(field1) # Find points in space which are valid for both fields at all times
    valid_points2=~np.isnan(field2)
    mask=np.all(valid_points1,axis=0) | np.all(valid_points2,axis=0)
    f1=field1[:,mask]
    f2=field2[:,mask] 

    norm1=np.nanstd(f1,axis=0) # Normalise field correctly to get correlstion 1 at delta time 0
    norm2=np.nanstd(f2,axis=0)
    vals1=(f1-np.nanmean(f1,axis=0))/norm1
    vals2=(f2-np.nanmean(f2,axis=0))/norm2
    
    dts=correlation_lags(field1.shape[0],field2.shape[0])*np.mean(np.diff(times)) # get the time lags

    corls=np.zeros((len(dts),f1.shape[1]))
    for i in range(f1.shape[1]): # loop over all points in space
        if np.any(np.isnan(vals1[:,i])) or np.any(np.isnan(vals2[:,i])): # check field is not nan at any time
            corls[:,i]=np.nan
            continue

        corls_now=correlate(vals1[:,i],vals2[:,i],mode='full',method='fft') # compute the correlation function using scipy
        corls[:,i]=corls_now # store the correlation function

    corls_out=np.nanmean(corls,axis=1) # average over all points in space

    norm_mat=correlate(np.ones_like(f1[:,0]),np.ones_like(f2[:,0]),mode='full',method='fft') # make sure the correlation function is normalised correctly
    corls_out=corls_out/norm_mat

    return corls_out,dts


# function to compute the spatial correlation function of a field on a meshgrid using the FFT
def average_correl_FT(field1,field2,coords,do_print=True):
    from scipy.signal import correlate,correlation_lags

    xs=np.unique(coords[:,0])
    ys=np.unique(coords[:,1])
    dx=np.mean(np.diff(xs))
    dy=np.mean(np.diff(ys))
    dxs=correlation_lags(field1.shape[1],field2.shape[1])*dx
    dys=correlation_lags(field1.shape[2],field2.shape[2])*dy

    corls=np.zeros((field1.shape[0],len(dxs),len(dys)))


    for t in range(field1.shape[0]):
        vals1=field1[t,:,:]-np.nanmean(field1[t,:,:])
        vals2=field2[t,:,:]-np.nanmean(field2[t,:,:])
        vals1=vals1/np.nanstd(vals1)
        vals2=vals2/np.nanstd(vals2)
        corls_now=correlate(vals1,vals2,mode='full',method='fft')
        # normalise correlation function
        corls[t,:,:]=corls_now
        if do_print:
            print(f"Completed {t/field1.shape[0]*100:.2f}% of times",end='\r')

    # compute correct normilisation matrix    
    dxs_grid,dys_grid=np.meshgrid(dxs,dys,indexing='ij')
    corls_out=np.mean(corls,axis=0)
    norms=correlate(np.ones_like(field1[0,:,:]),np.ones_like(field2[0,:,:]),mode='full',method='fft')
    corls_norm=corls_out/norms
        
    return corls_norm,dxs_grid,dys_grid

# Takes a scalar field on an unstructured grid puts it on a meshgrid
def interpolate_to_rect(field,coords,do_print=True):
    from scipy.interpolate import griddata

    xs=np.linspace(np.min(coords[:,0]),np.max(coords[:,0]),50) # get the x and y coordinates of the meshgrid
    ys=np.linspace(np.min(coords[:,1]),np.max(coords[:,1]),50) 
    Xs,Ys=np.meshgrid(xs,ys,indexing='ij')

    field_rect=np.zeros((field.shape[0],len(xs),len(ys))) # create an array to store the interpolated field
    for t,f in enumerate(field): # loop over all times
        field_rect[t,:,:]=griddata(coords[:,0:2],f,(Xs,Ys),method="linear") # interpolate the field to the meshgrid
        if do_print:
            print(f"Completed {100*t/len(field):.2f}% of interpolations",end="\r")
    
    return field_rect,Xs,Ys

# Takes in an array mask, as well as the information about a rectangle, and returns the mask with the rectangle filled in
# and returns a new mask of the same size as the input mask, with the rectangle filled in
def sides_2_mask(sides,mask):
    Lx,Ly,centrex,centrey=sides # unpack inputs
    left_edge = max(0, centrex - Lx // 2)
    right_edge = min(mask.shape[0], centrex + (Lx + 1) // 2)
    bottom_edge = max(0, centrey - Ly // 2)
    top_edge = min(mask.shape[1], centrey + (Ly + 1) // 2)

    left_edge = int(left_edge) # convert to integers, to use as indices
    right_edge = int(right_edge)
    bottom_edge = int(bottom_edge)
    top_edge = int(top_edge)
    
    mask_blank=np.zeros_like(mask) # create blank mask
    mask_blank[left_edge:right_edge,bottom_edge:top_edge]=1 # fill in rectangle
    return mask_blank # return filled in mask
    
# a cost function used below to assign a small cost for having a too small mask or a giant one for having a too large mask
def objective_function(rect_params, mask):
    
    now_mask=sides_2_mask(rect_params,mask) # convert rectangle to mask using sides_2_mask function

    inside_mask = np.sum(mask[now_mask]) # sum of rectangular mask inside actual image mask
    outside_mask = np.sum(~mask[now_mask]) # sum of rectangular mask outside actual image mask


    inside_value = 1 # value inside mask, +1 as we want to maximise number of pixels inside mask
    outside_value = -1000 # value outside mask, -1000 as we want to massively punich any pixels outside mask
    return inside_mask * inside_value + outside_mask * outside_value # return objective function value

# function to take a mask, and find the largest rectangle which fits inside the mask
def find_largest_rectangle(mask):
    from scipy.optimize import differential_evolution

    # Initialize the mask, we use full image as the initial guess
    Lx, Ly = mask.shape

    # Use scipy differential evolution algorithm to minimise the objective function
    # Inputs are Lx and Ly, the dimensions of the rectangle, as well as centrex and centrey, the coordinates of the centre of the rectangle
    result = differential_evolution(lambda x: -objective_function(x, mask), [(1, Lx), (1, Ly), (1, Lx), (1, Ly)])
    
    # Convert rectangle to mask
    rect_mask=sides_2_mask(result.x,mask)
    
    return rect_mask,result

def compute_correlation_unstruc(field,coords):
    f_sq,X_sq,Y_sq=interpolate_to_rect(field,coords,do_print=True) # interpolate the field to a meshgrid
    # Find where this interpolation is valis and turn it into a mask, note that griddata returns nan for points outside the convex hull
    big_mask=np.isnan(f_sq[0,:,:]) 
    
    # Find the largest rectangle which fits inside the mask
    rect_mask,_=find_largest_rectangle(~big_mask)

    # Find the points which are inside the rectangle
    Xs_valid=X_sq[rect_mask]
    Ys_valid=Y_sq[rect_mask]
    un_x=np.unique(Xs_valid)
    un_y=np.unique(Ys_valid)
    f_valid=f_sq[:,rect_mask].reshape((f_sq.shape[0],len(un_x),len(un_y)))

    # Flatten the field and coordinates
    Xs_sq_flat=Xs_valid.flatten()
    Ys_sq_flat=Ys_valid.flatten()
    coords_valid=np.array([Xs_sq_flat,Ys_sq_flat]).T

    corls,dXs,dYs=average_correl_FT(f_valid,f_valid,coords_valid) # compute the correlation function of the field
    
    return corls,dXs,dYs # return the correlation function

# Compute the Half Width at Half Maximum of a 1D correlation function
def compute_HWHM(rs,corls):
    from scipy.interpolate import CubicSpline

    dr_interp=np.linspace(np.min(rs),np.max(rs),2000) # interpolate the correlation function so that we don't miss the half maximum
    cs=CubicSpline(rs,corls)    
    corls_interp=cs(dr_interp)

    try:
        where_half=np.abs(corls_interp-0.5)<0.01 # find where the correlation function is within 0.01 of 0.5
        xs_where_half=dr_interp[where_half] # get the x values where this is true
        min_where_half=np.min(np.abs(xs_where_half)) # get the minimum of the closest x values to 0, 
        # as this will be the first point where the correlation function is within 0.01 of 0.5
    except:
        min_where_half=np.nan # if this fails, return nan as the HWHM is not defined

    return min_where_half

##################################################################################################
# Code for looking for clusters (or contracted regions) in a scalr field #
##################################################################################################


class pulse: # class to store information about a cluster of points we is a function of time
    
        def __init__(self,times,label,coords):
            self.times=times # store the times the field is defined from
            self.label=label # label of the cluster
            self.coords=coords # coordinates of points in the cluster
            self.all_inds=[None for t in times] # indices of the points in the cluster
            self.time_valid=[False for t in times] # whether the cluster exists at each time

        def conver_time_to_ind(self,time): # convert time to index
            return self.times.tolist().index(time)
        
        def check_valid(self,time): # check if the cluster exists at a given time
            t=self.conver_time_to_ind(time)
            return self.time_valid[t]
        
        def count_number_valid(self): # count the number of times the cluster exists
            return np.sum(self.time_valid)
        
        def add_ind(self,time,ind): # add an index to the cluster at a given time
            t=self.conver_time_to_ind(time)
            self.all_inds[t]=ind
            self.time_valid[t]=True

        def get_coords(self,time): # get the coordinates of the points in the cluster at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                return self.coords[self.all_inds[t],:]
            else:
                return np.asarray([np.nan,np.nan])
            
        def get_centroid(self,time): # get the centroid of the cluster at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                coords_now=self.get_coords(time)
                return np.mean(coords_now,axis=0)
            else:
                return np.asarray([np.nan,np.nan,np.nan])
        
        def get_radius(self,time): # get the maximum distance of a point in the cluster from the centroid at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                return np.max(np.linalg.norm(self.get_coords(time)-self.get_centroid(time),axis=1))
            else:
                return np.nan
        
        def get_shape_tensor(self,time): # get the shape tensor of the cluster at a given time and return the eigenvalues and eigenvectors
            import fields_methods as fm
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                vals1,vals2,vects1,vects2=fm.compute_shape_tensor(self.get_coords(time).T)
                return vals1,vals2,vects1,vects2
            else:
                return np.nan,np.nan,np.asarray([np.nan,np.nan]),np.asarray([np.nan,np.nan])

        def get_eigvals(self,time): # get the eigenvalues of the shape tensor at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                vals1,vals2,vects1,vects2=self.get_shape_tensor(t)
                return vals1,vals2
            else:
                return np.nan,np.nan
            
        def get_eigvecs(self,time): # get the eigenvectors of the shape tensor at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                vals1,vals2,vects1,vects2=self.get_shape_tensor(time)
                return vects1,vects2
            else:
                return np.asarray([np.nan,np.nan]),np.asarray([np.nan,np.nan])
        
        def get_eccentricity(self,time): # get the eccentricity of the cluster at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                vals1,vals2,vects1,vects2=self.get_shape_tensor(time)
                return np.sqrt(1-vals2/vals1)
            else:
                return np.nan
        
        def get_aspect_ratio(self,time): # get the feret aspect ratio of the cluster at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                points=self.get_coords(time)
                points_arr=np.asarray(points)
                # print(points_arr.shape)
                f_AR=feret_aspect(points_arr[:,0:2])
            else:
                f_AR=np.nan
            return f_AR
        
        def get_orientation(self,time): # get the orientation angle of the largest eigenvector of the shape tensor at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                vals1,vals2,vects1,vects2=self.get_shape_tensor(time)
                return np.arctan2(vects1[1],vects1[0])
            else:
                return np.nan
            
        def get_cluster_size(self,time): # get the number of points in the cluster at a given time
            t=self.conver_time_to_ind(time) 
            if self.time_valid[t]:
                return len(self.all_inds[t])
            else:
                return np.nan
        
        def spread(self,time): # get the spread of the points in the cluster at a given time
            t=self.conver_time_to_ind(time)
            if self.time_valid[t]:
                coords_now=self.get_coords(time)
                std_xy=np.nanmean(coords_now**2,axis=0)-np.nanmean(coords_now,axis=0)**2
                return np.sqrt(np.sum(std_xy**2))
            else:
                return np.nan
    
        def time_av_spread(self): # get the time averaged spread of the points in the cluster
            return np.nanmean([self.spread(t) for t in self.times])
        
        def time_av_radius(self): # get the time averaged radius of the cluster
            return np.nanmean([self.get_radius(t) for t in self.times])
        
        def time_av_eccentricity(self): # get the time averaged eccentricity of the cluster
            return np.nanmean([self.get_eccentricity(t) for t in self.times])

        def time_av_aspect_ratio(self): # get the time averaged aspect ratio of the cluster
            return np.nanmean([self.get_aspect_ratio(t) for t in self.times])
        
        def time_av_orientation(self): # get the time averaged orientation of the cluster
            return np.nanmean([self.get_orientation(t) for t in self.times])
        
        def time_av_pulse_size(self): # get the time averaged number of points in the cluster
            return np.nanmean([self.get_cluster_size(t) for t in self.times])
            
        
# Class to store a set of pulses which are each functions of time
class pulse_set:
    def __init__(self,times,coords,field,all_pulses):
        self.times=times # store the times associatedto the field which we get the pulses from
        self.coords=coords # store the coordinates of the field which we get the pulses from
        self.field=field # store the field which we get the pulses from
        self.pulse_list=all_pulses # store the pulses
        self.pulse_labels=[p.label for p in self.pulse_list] # store the labels of the pulses
        self.mergers=[[] for t in times] # store the mergers events which occur at each time

    def prune_short_pulses(self,min_length=5): # Function to remove pulses which persist for less than a given number of time steps
        for p in self.pulse_list:
            N=p.count_number_valid()
            if N<min_length:
                for t,time in enumerate(self.times):
                    p.time_valid[t]=False

    def get_pulse(self,label): # get a pulse by its label
        for p in self.pulse_list:
            if p.label==label:
                return p

    def get_num_pulses_time_dep(self): # get the number of pulses at each time
        Ns=[]
        for time in self.times:
            num_pulses=len([p for p in self.pulse_list if p.check_valid(time)])
            Ns.append(num_pulses)
        return Ns
    
    def all_individual_pulses(self): # get the number of unique pulses which exist across all times
        return len(self.pulse_list)
        
    # find when two pulses merge, given by when one dissapears while close to another
    def check_for_mergers(self,tol=1): 
        for t_ind in range(len(self.times)-1): # iterate over all times
            t=self.times[t_ind] # get the time now
            next_time=self.times[t_ind+1] # get the next time
            
            for p in self.pulse_list: # go through all pulses which exist at the current time and not at the next time
                if p.check_valid(t) and not p.check_valid(next_time):
                    for p2 in self.pulse_list: # go through all pulses which exist at this time and the next time
                        if p2.check_valid(t) and p2.check_valid(next_time):
                            
                            overlap=check_min_overlap(p.get_coords(t),p2.get_coords(next_time)) # check if the pulses overlap
                            if overlap<tol: # if they do, store the merger event
                                self.mergers[t_ind].append((p.label,p2.label))
    
    def get_merger_info(self,tol=1): # method to run the merger calculation and summarise the results
        self.check_for_mergers(tol=tol) # run the merger calculation
        
        num_mergers_in_time=[len(mergers) for mergers in self.mergers] # get the number of mergers at each time
        num_mergers=np.sum(num_mergers_in_time) # get the total number of mergers
        
        pulses_which_merge=set() # get the pulses which merge
        for mergers in self.mergers:
            for merger in mergers:
                pulses_which_merge.add(merger[0])
                pulses_which_merge.add(merger[1])
                
        return num_mergers,pulses_which_merge,num_mergers_in_time
    
    def get_init_site(self,p_label): # get the site where a pulse first appears
        pulse=self.get_pulse(p_label)
        valid_times=[t for t in self.times if pulse.check_valid(t)]
        first_time=valid_times[0]
        return pulse.get_centroid(first_time)
    
    def get_all_init_sites(self): # get the initiation site of all pulses
        init_sites=np.zeros((self.all_individual_pulses(),3))
        for p in self.pulse_list:
            init_sites[p.label:]=self.get_init_site(p.label)
        return init_sites
    
    # get com velcoties of all pulses at all times
    def get_vel_matrix(self):
        vel_mat=np.zeros((self.all_individual_pulses(),self.times.shape[0],3))
        for p in self.pulse_list:
            vel_mat[p.label,:,:]=self.get_velocities(p.label)
        vel_mat[vel_mat==0]=np.nan
        return vel_mat
        
    # info on individual pulses, all all times for a series of quantities
    
    def get_centroids(self,p_label): 
        pulse=self.get_pulse(p_label)
        centroids=[pulse.get_centroid(t)  if pulse.check_valid(t) else np.asarray([np.nan,np.nan,np.nan]) for t in self.times]
        return centroids

    def get_eccentricities(self,p_label): 
        pulse=self.get_pulse(p_label)
        eccentricities=[pulse.get_eccentricity(t)  if pulse.check_valid(t) else np.nan for t in self.times]
        return eccentricities
    
    def get_aspect_ratios(self,p_label):
        pulse=self.get_pulse(p_label)
        aspect_ratios=[pulse.get_aspect_ratio(t)  if pulse.check_valid(t) else np.nan for t in self.times]
        return aspect_ratios
    
    def get_orientations(self,p_label):
        pulse=self.get_pulse(p_label)
        orientations=[pulse.get_orientation(t)  if pulse.check_valid(t) else np.nan for t in self.times]
        return orientations
    
    def get_cluster_sizes(self,p_label):
        pulse=self.get_pulse(p_label)
        cluster_sizes=[pulse.get_cluster_size(t)  if pulse.check_valid(t) else np.nan for t in self.times]
        return cluster_sizes
    
    def get_spreads(self,p_label):
        pulse=self.get_pulse(p_label)
        spreads=[pulse.spread(t)  if pulse.check_valid(t) else np.nan for t in self.times]
        return spreads
    
    def get_velocities(self,p_label):
        pulse=self.get_pulse(p_label)
        t_prev=self.times[0]
        vels=np.zeros((self.times.shape[0],3))
        for i,t in enumerate(self.times):
            if i==0:
                vel=np.asarray([np.nan,np.nan,np.nan])
            if pulse.check_valid(t) and pulse.check_valid(t_prev):
                vel=(pulse.get_centroid(t)-pulse.get_centroid(t_prev))/(t-t_prev)
            else:
                vel=np.asarray([np.nan,np.nan,np.nan])
            t_prev=t
            vels[i,:]=vel
        return vels
    
    # info on time dependent values of series of quantities averaged over all pulses
    
    def get_pulse_averaged_centroids(self):
        all_pos=np.zeros((self.times.shape[0],self.all_individual_pulses(),3))
        for p in self.pulse_list:
            all_pos[:,p.label,:]=self.get_centroids(p.label)
        return np.nanmean(all_pos,axis=1)
    
    def get_pulse_averaged_eccentricities(self):
        all_ecc=np.zeros((self.times.shape[0],self.all_individual_pulses()))
        for p in self.pulse_list:
            all_ecc[:,p.label]=self.get_eccentricities(p.label)
        return np.nanmean(all_ecc,axis=1)
    
    def get_pulse_averaged_aspect_ratios(self):
        all_ar=np.zeros((self.times.shape[0],self.all_individual_pulses()))
        for p in self.pulse_list:
            all_ar[:,p.label]=self.get_aspect_ratios(p.label)
        return np.nanmean(all_ar,axis=1)
    
    def get_pulse_averaged_orientations(self):
        all_or=np.zeros((self.times.shape[0],self.all_individual_pulses()))
        for p in self.pulse_list:
            all_or[:,p.label]=self.get_orientations(p.label)
        return np.nanmean(all_or,axis=1)
    
    def get_pulse_averaged_cluster_sizes(self):
        all_cs=np.zeros((self.times.shape[0],self.all_individual_pulses()))
        for p in self.pulse_list:
            all_cs[:,p.label]=self.get_cluster_sizes(p.label)
        return np.nanmean(all_cs,axis=1)
    
    def get_pulse_averaged_spreads(self):
        all_sp=np.zeros((self.times.shape[0],self.all_individual_pulses()))
        for p in self.pulse_list:
            all_sp[:,p.label]=self.get_spreads(p.label)
        return np.nanmean(all_sp,axis=1)
    
    def get_pulse_averaged_velocities(self):
        all_vel=np.zeros((self.times.shape[0],self.all_individual_pulses(),3))
        for p in self.pulse_list:
            all_vel[:,p.label]=self.get_velocities(p.label)
        return np.nanmean(all_vel,axis=1)
    
    def get_pulse_averaged_eigvals(self):
        all_eig=np.zeros((self.times.shape[0],self.all_individual_pulses(),2))
        for p in self.pulsxe_list:
            all_eig[:,p.label]=self.get_eigvals(p.label)
        return np.nanmean(all_eig,axis=1)
    
    def get_pulse_averaged_eigvecs(self):
        all_eig=np.zeros((self.times.shape[0],self.all_individual_pulses(),2,2))
        for p in self.pulse_list:
            all_eig[:,p.label]=self.get_eigvecs(p.label)
        return np.nanmean(all_eig,axis=1)
    
    def get_pulse_averaged_shape_tensor(self):
        all_eig=np.zeros((self.times.shape[0],self.all_individual_pulses(),2,2))
        for p in self.pulse_list:
            all_eig[:,p.label]=self.get_shape_tensor(p.label)
        return np.nanmean(all_eig,axis=1)

    # time averaged value for a series of quantities for an individual pulse
    
    def get_time_av_spreads(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_spread()
    
    def get_time_av_eccentricities(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_eccentricity()
    
    def get_time_av_aspect_ratios(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_aspect_ratio()
    
    def get_time_av_orientations(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_orientation()
    
    def get_time_av_pulse_size(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_pulse_size()
    
    def get_time_av_radii(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_radius()
    
    def get_time_av_eigvals(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_eigvals()
    
    def get_time_av_eigvecs(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_eigvecs()
    
    def get_time_av_shape_tensor(self,p_label):
        pulse=self.get_pulse(p_label)
        return pulse.time_av_shape_tensor()
    
    # info averaged over time and all pulses
    
    def time_pulse_avergaed_centroids(self):
        return np.nanmean(self.get_pulse_averaged_centroids(),axis=0)
    
    def time_pulse_averaged_eccentricities(self):
        return np.nanmean(self.get_pulsed_ecentricities(),axis=0)
    
    def time_pulse_averaged_aspect_ratios(self):
        return np.nanmean(self.get_pulse_averaged_aspect_ratios(),axis=0)
    
    def time_pulse_averaged_orientations(self):
        return np.nanmean(self.get_pulse_averaged_orientations(),axis=0)
    
    def time_pulse_averaged_cluster_sizes(self):
        return np.nanmean(self.get_pulse_averaged_cluster_sizes(),axis=0)
    
    def time_pulse_averaged_spreads(self):
        return np.nanmean(self.get_pulse_averaged_spreads(),axis=0)
    
    def time_pulse_averaged_velocities(self):
        return np.nanmean(self.get_pulse_averaged_velocities(),axis=0)
    
    def time_pulse_averaged_eigvals(self):
        return np.nanmean(self.get_pulse_averaged_eigvals(),axis=0)
    
    def time_pulse_averaged_eigvecs(self):
        return np.nanmean(self.get_pulse_averaged_eigvecs(),axis=0)
    
    def time_pulse_averaged_shape_tensor(self):
        return np.nanmean(self.get_pulse_averaged_shape_tensor(),axis=0)
    
    # time averaged info for all pulses, for a bunch of quantities
    def all_pulse_info(self):
        infos=[]
        for p in self.pulse_list:
            
            p_rad=p.time_av_radius()
            p_ecc=p.time_av_eccentricity()
            p_ar=p.time_av_aspect_ratio()
            p_or=p.time_av_orientation()
            p_size=p.time_av_pulse_size()
            p_spread=p.time_av_spread()

            all_vels=self.get_velocities(p.label)
            all_mags=np.linalg.norm(all_vels,axis=1)
            all_thetas=np.arctan2(all_vels[:,1],all_vels[:,0])
            p_mag=np.nanmean(all_mags)
            p_dir=np.nanmean(all_thetas)
            
            info_dict={"label":p.label, "radius":p_rad, "eccentricity":p_ecc,
                       "aspect_ratio":p_ar, "orientation":p_or, "size":p_size, 
                       "spread":p_spread, "vel_dir":p_dir, "vel_mag":p_mag}
                
            infos.append(info_dict)
        return infos

# Soow field with located clusters on top
# Make series of frames and save as gif
def animate_field_and_pulses(times,coords,field,all_pulses,save_dir):
    import imageio
    from shutil import rmtree
    from os.path import join
    from os import makedirs
    import matplotlib.pyplot as plt
    
    pulse_dir=join(save_dir,"pulse_tracking")
    makedirs(pulse_dir,exist_ok=True)
    
    max_f=np.max(field)
    min_f=np.min(field)

    frames=[]
    pulses_prev=[]
    colors_p={}

    unique_label=np.unique([p.label for p in all_pulses])
    colors=[plt.cm.jet(i) for i in np.linspace(0,1,len(unique_label))]
    np.random.shuffle(colors)
    color_dict={l:c for l,c in zip(unique_label,colors)}

    for ind,(t,f) in enumerate(zip(times,field)):
        fig,ax=plt.subplots(1,1,figsize=(10,10))
        ax.tripcolor(coords[:,0],coords[:,1],f,cmap='gray',vmin=min_f,vmax=max_f)
        pulses_now=[p for p in all_pulses if p.check_valid(t)]
        for p in pulses_now:
            ax.scatter(p.get_coords(t)[:,0],p.get_coords(t)[:,1],color=color_dict[p.label])
            ax.text(p.get_centroid(t)[0],p.get_centroid(t)[1],f"{p.label}",color='r')
        ax.set_title(f"Frame={ind},Num pulses={len(pulses_now)}")
        ax.set_aspect('equal')
        fig.tight_layout()

        # plt.show()

        fname=join(pulse_dir,f"pulse_tracking_{ind:d}.png")
        plt.savefig(fname)
        print(f"saved frame {fname}")
        frames.append(imageio.imread(fname))
        plt.close()
    
    imageio.mimsave(join(save_dir,"pulse_tracking.gif"),frames,format='GIF',fps=10)

    rmtree(pulse_dir)
    return

# Compute COM of a cluster
def cluster_COM(clust,Xs,Ys):
    xs=[Xs[i,j] for i,j in clust]
    ys=[Ys[i,j] for i,j in clust]
    if len(xs)==0:
        Exception("Empty cluster")
    comx=np.nanmean(xs)
    comy=np.nanmean(ys)
    return comx,comy

# Function to take fenics data and compute the clusters of points which are above a certain threshold
def cluster_fenics_dat(field,connecs,num_std=2,min_cluster_size=5,pulses="all",mean=None,std=None):
    
    if mean is None: # compute the mean and standard deviation of the field
        mean_val=np.mean(field)
    else:
        mean_val=mean
    if std is None:
        std_val=np.std(field)
    else:
        std_val=std
    
    # look for where the field is above a certain threshold
    # ( eg abs(field)>something or field<something )
    mask=np.zeros_like(field,dtype=bool)
    excluded=np.copy(mask) # array to tells us which point should not be added as they are already in a cluster
    if pulses=="high":
        mask[field-mean_val>num_std*std_val]=True 
    elif pulses=="low":
        mask[field-mean_val<-num_std*std_val]=True
    else:
        mask[np.abs(field-mean_val)>num_std*std_val]=True

    clusters=[] # store the clusters

    for i in range(len(field)): # loop over all points
        if mask[i] and not excluded[i]: # if the point is above the threshold and not already in a cluster
            clusters.append([i]) # start a new cluster
            excluded[i]=True # exclude the point from being added to another cluster
            stack=[i] # points to search through
            while len(stack)>0: # go through all points we need to search through
                curr=stack.pop() # get the current point
                for j in connecs[curr]: # go through all points connected to the current point
                    if mask[j] and not excluded[j]: # if the point is above the threshold and not already in a cluster
                        clusters[-1].append(j) # add the point to the cluster
                        excluded[j]=True # exclude the point from being added to another cluster
                        stack.append(j) # add the point to the stack to search through
 
    clusters_out=[c for c in clusters if len(c)>min_cluster_size] # remove clusters that are too small
    return clusters_out

# Do clustering in a time dependent way
def cluster_time_fenics_dat(field,connecs,**kwargs):
    clust_ts=[]
    mean_val=np.mean(field)
    std_val=np.std(field)
    for t,f in enumerate(field):
        clusters=cluster_fenics_dat(f,connecs,**kwargs,mean=mean_val,std=std_val)
        print(f"t={t}, num clusters={len(clusters)}")
        clust_ts.append(clusters)
    return clust_ts

# Function to associate clusters in time (determine when two clusters at different times are the same pulse which has moved)
def track_pulses(clust_ts,coords,times,dist_tol):
    pulses=[] # store the pulses
    
    clusts0=clust_ts[0] # clusters at time 0
    label=0 
    for c0 in clusts0: # go through all clusters at time 0
        this_pulse=pulse(times,label,coords) # make a pulse object for each cluster
        this_pulse.add_ind(times[0],c0) 
        pulses.append(this_pulse) # add the pulse to the list
        label+=1
        
    # go through all times, and get al the clusters we found at that time
    for t,clusts in enumerate(clust_ts):

        if t==0:
            continue
        
        prev_time=times[t-1] # get the previous time
        time_now=times[t] # get the current time
        unclaimed_pulses=[p for p in pulses if p.check_valid(prev_time)] # get all pulses which exist at the previous time
        unclaimed_clusts=[c for c in clusts] # get all clusters at the current time
        counter=0

        for c in clusts: # go through all clusters at the current time
            for p in unclaimed_pulses: # go through all pulses which exist at the previous time
                dist=check_min_overlap(coords[c,:],p.get_coords(prev_time)) # check if the cluster and pulse overlap
                if dist<dist_tol: # if they do, add the cluster to the pulse
                    p.add_ind(time_now,c) # add the cluster to the pulse at the current time
                    unclaimed_pulses.remove(p) # remove the pulse from the list of unclaimed pulses
                    unclaimed_clusts.remove(c) # remove the cluster from the list of unclaimed clusters
                    counter+=1
                    break

        for c in unclaimed_clusts: # go through all unclaimed clusters - these must be new ones which have appeared
            this_pulse=pulse(times,label,coords) # make a new pulse object
            this_pulse.add_ind(time_now,c) # add the cluster to the pulse
            pulses.append(this_pulse) # add the pulse to the list
            label+=1

    return pulses
 
def check_min_overlap(coords1,coords2): # given two clusters of points, check the distance between the two closest points
    coords1_b=coords1.reshape((1,coords1.shape[0],coords1.shape[1]))
    coords2_b=coords2.reshape((coords2.shape[0],1,coords2.shape[1]))
    dists=np.linalg.norm(coords1_b-coords2_b,axis=2)
    return np.min(dists)

def initation_site_analysis(sites,bounds,tol=2,do_rotate=True):
    """
    Function to analyse the location of initiation sites of pulses.
    sites: array of coordinates of initiation sites
    bounds: array of coordinates of boundary points
    tol: tolerance for counting number of sites which are considered to be at the same location
    do_rotate: whether to rotate the points so that the top point is at the top of the plot
    """
    
    COM=np.mean(bounds,axis=0) # compute the centre of mass of the boundary points
    bounds=bounds-COM # shift the boundary points so that the COM is at the origin
    sites=sites-COM # shift the sites so that the COM is at the origin
    
    drs=bounds.reshape((bounds.shape[0],1,2))-bounds.reshape((1,bounds.shape[0],2)) # compute distances between all pairs of boundary points
    drs=np.sqrt(np.sum(drs**2,axis=2))
    
    # find the pair of boundary points which are furthest apart, compute the angle of these points wrt the y axis
    max_drs=np.argmax(drs) # find the maximum distance between boundary points
    max_inds=np.unravel_index(max_drs,drs.shape) # get the indices of the pair of boundary points
    p1=bounds[max_inds[0],:] # get the boundary points
    p2=bounds[max_inds[1],:]
    top_point_ind=max_inds[np.argmax([p1[1],p2[1]])]    
    angle=-np.arctan2(bounds[top_point_ind,0],bounds[top_point_ind,1])
    
    # rotate points such that the two furthest points are aligned in the y direction
    if do_rotate:
        rot_mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        sites_rot=np.sum(rot_mat.reshape(1,2,2)*sites.reshape(sites.shape[0],2,1),axis=1)
        bounds_rot=np.sum(rot_mat.reshape(1,2,2)*bounds.reshape(bounds.shape[0],2,1),axis=1)
    else:
        sites_rot=np.copy(sites)
        bounds_rot=np.copy(bounds)
    
    # Matrix to store the number of sites in each region of the cell
    pos_count_mat=np.zeros((3,3))
    mean_coords=np.zeros((3,3,2))
    
    height_third=(np.max(bounds_rot[:,1])-np.min(bounds_rot[:,1]))/3 # seperate the cell into three vertical
    top_third=np.max(bounds_rot[:,1])-height_third
    bottom_third=np.min(bounds_rot[:,1])+height_third
    
    upper_count=0 # also counters to say whether init site is left/right or top/bottom
    lower_count=0
    left_count=0
    right_count=0
    
    
    for site_ind in range(sites_rot.shape[0]): # go through all sites
        site=sites_rot[site_ind,:] # get the site
        
        # multis-step process to determine which left/right region of the cell the site is in
        points_to_left=bounds_rot[bounds_rot[:,0]<site[0],:] # get the boundary points to the left of the site
        points_to_right=bounds_rot[bounds_rot[:,0]>site[0],:] # get the boundary points to the right of the site
        y_dists_left=np.abs(points_to_left[:,1]-site[1]) # compute the vertical distance between the site and the boundary points to the left
        y_dists_right=np.abs(points_to_right[:,1]-site[1]) # compute the vertical distance between the site and the boundary points to the right
        closest_lp=points_to_left[np.argmin(y_dists_left),:] # get the closest point (in the y direction) to the left of the site
        closest_rp=points_to_right[np.argmin(y_dists_right),:] # get the closest point (in the y direction) to the right of the site
        width=np.abs(closest_rp[0]-closest_lp[0]) # compute the width of the cell at this particular height
        left_third=closest_lp[0]+width/3 # divide the cell into three horizontal regions
        right_third=closest_rp[0]-width/3
        
        if site[1]<bottom_third: # determine which vertical region the site is in
            row=2
        elif site[1]>top_third:
            row=0
        else:
            row=1
            
        if site[0]<left_third: # determine which horizontal region the site is in
            col=0
        elif site[0]>right_third:
            col=2
        else:
            col=1
            
        pos_count_mat[row,col]+=1 # increment the count of sites in this region
        # (want want to roughly know where the centre of mass of the sites in each region is)
        mean_coords[row,col,:]+=site # add the coordinates of the site to the mean coordinates of the region 

        
        if site[1]>0: 
            upper_count+=1
        else:
            lower_count+=1
        if site[0]>closest_lp[0]+0.5*width:
            right_count+=1
        else:
            left_count+=1
    
    # useful for plotting the initation sites
    # We define a sclar for each init site which is the number of sites within a certain distance of it
    # This will be a colour for the site in a plot
    cols_ar=np.zeros((len(sites_rot)))
    for ind,site in enumerate(sites_rot):
        dists=np.sqrt((sites_rot[:,0]-site[0])**2+(sites_rot[:,1]-site[1])**2)
        N=np.sum(dists<tol)
        cols_ar[ind]=N
        
    # Output the dictionary of all the summarised information
    outputs={"pos_count_mat":pos_count_mat,"mean_coords":mean_coords,"pulse_neighbours":cols_ar,"sites_rot":sites_rot,"bounds_rot":bounds_rot,
            "upper_count":upper_count,"lower_count":lower_count,"left_count":left_count,"right_count":right_count}   
    return outputs


def feret_aspect(points): # function to compute the feret aspect ratio of a set of points - written by chatGPT
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(points,qhull_options="QJ") # compute the convex hull of the points
    hull_points = points[hull.vertices]

    min_feret = float('inf')
    max_feret = 0
    num_hull_points = len(hull_points)

    for i in range(num_hull_points):
        for j in range(i + 1, num_hull_points):

            p1 = hull_points[i]
            p2 = hull_points[j]
            edge_vector = p2 - p1 # compute vector which points along the edge of the convex hull

            perp_vector = np.array([-edge_vector[1], edge_vector[0]]) # get orthogonal vector to the edge vector (in 2d there is only 1)
            
            perp_vector /= np.linalg.norm(perp_vector) # normalise the vector, this defines some given caliper orientation
            projections = np.dot(hull_points - p1, perp_vector) # project all points onto the caliper orientation (Ie how far along the caliper are they)

            min_distance = np.min(projections) # get the minimum distance along the caliper
            max_distance = np.max(projections) # get the maximum distance along the caliper
            feret_diameter = max_distance - min_distance # compute the feret diameter

            min_feret = min(min_feret, feret_diameter) # check if our current feret diameter is the smallest or largest so far
            max_feret = max(max_feret, feret_diameter)

    return max_feret / min_feret # return the aspect ratio

# animate scalar field using tripcolor
def animate_tripfield(times,coords,field,save_loc,title="Myosin Field"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=400)
    ax.set_aspect("equal")
    ax.set_xlim([np.min(coords[:,0]),np.max(coords[:,0])])
    ax.set_ylim([np.min(coords[:,1]),np.max(coords[:,1])])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    
    def animate(i):
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim([np.min(coords[:,0]),np.max(coords[:,0])])
        ax.set_ylim([np.min(coords[:,1]),np.max(coords[:,1])])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        c=ax.tripcolor(coords[:,0],coords[:,1],field[i,:],cmap="viridis",vmin=np.min(field),vmax=np.max(field))
        return ax
    
    anim = animation.FuncAnimation(fig, animate, frames=field.shape[0], interval=200, blit=False)
    anim.save(save_loc,dpi=400,fps=30)
    plt.close(fig)