
import numpy as np
from sys import argv
import matplotlib.pyplot as plt

def run_CC_analysis(CR_masks, smallest_alowed_pix = 1):
    import cv2

    labels = np.zeros_like(CR_masks, dtype=int) 
    labels_filtered = np.zeros_like(CR_masks, dtype=int)
    for t,CR in enumerate(CR_masks):
        CR_uint8 = (CR*255).astype(np.uint8)
        N, labels_now = cv2.connectedComponents(CR_uint8, connectivity=8)
        # print(f"Time {t}: found {N} connected components")
        labels[t,:,:] = labels_now

        # remove small components
        for l in np.unique(labels_now):
            if l == 0:
                continue
            mask_l = labels_now == l
            if np.sum(mask_l) < smallest_alowed_pix:
                labels_now[mask_l] = 0
        # relabel after removing small components
        unique_labels = np.unique(labels_now)
        for i, l in enumerate(unique_labels):
            if l == 0:
                continue
            labels_filtered[t,:,:][labels_now == l] = i
    return labels, labels_filtered


def modify_labels_for_PBCs(labels):
    """
    Merge labels that connect across periodic boundaries in 2D label maps over time.
    Handles left-right and top-bottom connections.
    """
    T, H, W = labels.shape

    for t in range(T):
        lbl = labels[t]
        unique_lbls = np.unique(lbl)
        if 0 in unique_lbls:
            unique_lbls = unique_lbls[unique_lbls != 0]

        # Store merges in a dict: parent[label] = merged_into_label
        parent = {l: l for l in unique_lbls}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # --- Left-right connections ---
        left_edge = lbl[:, 0]
        right_edge = lbl[:, -1]
        for l in np.unique(left_edge[left_edge > 0]):
            if l in unique_lbls:
                # Check for any label that matches at right edge
                overlap = np.unique(right_edge[left_edge == l])
                for o in overlap:
                    if o > 0 and o != l:
                        union(l, o)

        # --- Top-bottom connections ---
        top_edge = lbl[0, :]
        bottom_edge = lbl[-1, :]
        for l in np.unique(top_edge[top_edge > 0]):
            overlap = np.unique(bottom_edge[top_edge == l])
            for o in overlap:
                if o > 0 and o != l:
                    union(l, o)

        # --- Apply merges ---
        merged_lbl = lbl.copy()
        for l in unique_lbls:
            merged_lbl[lbl == l] = find(l)

        # --- Reassign consecutive labels for clarity ---
        new_labels = np.zeros_like(merged_lbl)
        uniq = np.unique(merged_lbl)
        uniq = uniq[uniq != 0]
        for new_id, old_id in enumerate(uniq, start=1):
            new_labels[merged_lbl == old_id] = new_id

        labels[t] = new_labels

    return labels

def get_times(filename):
    import vtk
    reader = vtk.vtkXdmf3Reader()
    reader.SetFileName(filename)
    reader.UpdateInformation()
    info = reader.GetOutputInformation(0)
    TIME_STEPS_KEY = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    n_steps = info.Length(TIME_STEPS_KEY)
    return np.array([info.Get(TIME_STEPS_KEY, i) for i in range(n_steps)])

def load_point_data_timeseries(filename, equil_time, field):
    import pyvista as pv
    reader = pv.get_reader(filename)
    times = get_times(filename)
    out_times = times[times > equil_time]
    data_per_time = []
    integrated_version = []
    for t in times:
        if t < equil_time:
            continue
        
        print(f"Loading time {t:.2f}")
        reader.set_active_time_value(t)
        mesh = reader.read()
        
        if field not in mesh.point_data:
            raise KeyError(f"Field '{field}' not in point_data.")

        # Interpolate point data to cell data
        data = mesh.point_data[field]
        data_per_time.append(data)
        integral = mesh.integrate_data()[field]
        area = mesh.compute_cell_sizes(length=False, volume=False, area=True).cell_data["Area"].sum()
        integrated_version.append(integral / area)

    return out_times, np.stack(data_per_time, axis=0), mesh

def mask_bound_2_coords(mask, Xs, Ys):
    from skimage import measure

    pad_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    Xs_pad = np.pad(Xs, pad_width=1, mode='edge')
    Ys_pad = np.pad(Ys, pad_width=1, mode='edge')

    contours = measure.find_contours(pad_mask.astype(np.uint8), 0.5)
    
    if len(contours) == 0:
        return np.array([]), np.array([])
    # Find the longest contour
    longest_contour = max(contours, key=len)


    x_inds = longest_contour[:, 0].astype(int)
    y_inds = longest_contour[:, 1].astype(int)
    x_coords = Xs_pad[x_inds, y_inds]
    y_coords = Ys_pad[x_inds, y_inds]
    return x_coords, y_coords

# Takes a scalar field on an unstructured grid puts it on a meshgrid
def interpolate_to_rect(field, times, mesh_ob, do_print=True, N=50, ):
    from scipy.interpolate import griddata
    
    coords = mesh_ob.points[:,0:2] # get the x and y coordinates of the points in the unstructured grid

    # make dims square
    min_x = np.min(coords[:,0])
    max_x = np.max(coords[:,0])
    min_y = np.min(coords[:,1])
    max_y = np.max(coords[:,1])
    overall_min  = min(min_x, min_y)
    overall_max  = max(max_x, max_y)

    xs = np.linspace(overall_min, overall_max, N)
    ys = np.linspace(overall_min, overall_max, N)

    Xs,Ys = np.meshgrid(xs, ys, indexing='ij')

    # make a mask of the region inside the original mesh
    tris = np.array([c.point_ids for c in mesh_ob.cell])
    mask = triangle_mask(coords, tris, Xs, Ys)

    field_rect=np.zeros((field.shape[0], Xs.shape[0], Xs.shape[1])) # create an array to store the interpolated field
    for t,f in enumerate(times): # loop over all times
        f=field[t,:,].squeeze()
        f_rect_now=griddata(coords[:,0:2], f, (Xs,Ys), method="linear") # interpolate the field to the meshgrid
        f_rect_now[np.logical_not(mask)]=np.nan # mask out points outside the original mesh
        field_rect[t,:,:]=f_rect_now
        if do_print:
            print(f"Completed {100*t/len(field):.2f}% of interpolations",end="\r")
    
    return field_rect, Xs, Ys, mask


def interpolate_to_rect_PBC(field, times, mesh_ob, do_print=True, N=50):
    from scipy.interpolate import griddata

    coords = mesh_ob.points[:,0:2] # get the x and y coordinates of the points in the unstructured grid

    # check min max dims
    min_x = np.min(coords[:,0])
    max_x = np.max(coords[:,0])
    min_y = np.min(coords[:,1])
    max_y = np.max(coords[:,1])
    x_len = max_x - min_x
    y_len = max_y - min_y
    if x_len>y_len:
        small_len, large_len = y_len, x_len
    else:
        small_len, large_len = x_len, y_len
    
    N_small = int(N * (small_len / large_len))

    if x_len>y_len:
        xs = np.linspace(min_x, max_x, N)
        ys = np.linspace(min_y, max_y, N_small)
    else:
        xs = np.linspace(min_x, max_x, N_small)
        ys = np.linspace(min_y, max_y, N)

    Xs,Ys = np.meshgrid(xs, ys, indexing='ij')
    field_rect=np.zeros((field.shape[0], Xs.shape[0], Xs.shape[1])) # create an array to store the interpolated field
    for t,f in enumerate(times): # loop over all times
        f=field[t,:,].squeeze()
        f_rect_now=griddata(coords[:,0:2], f, (Xs,Ys), method="linear") # interpolate the field to the meshgrid
        field_rect[t,:,:]=f_rect_now
        if do_print:
            print(f"Completed {100*t/len(field):.2f}% of interpolations",end="\r")
    return field_rect, Xs, Ys



def triangle_mask(points, triangles, X, Y):
        import matplotlib.tri as mtri
        """
        points:    (N, 2) float array of (x, y) vertices
        triangles: (M, 3) int array of indices into `points`
        X, Y:      meshgrid arrays (same shape)
        Returns:
         mask: boolean array, True where (X, Y) lies inside at least one triangle
        """
        tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        finder = tri.get_trifinder()
        tri_ids = finder(X, Y)       # -1 for outside, otherwise triangle index
        return tri_ids != -1

class pulse:

    def __init__(self, times, Xs, Ys, allow_uneven_grids=False):
        self.times = times
        self.masks = np.zeros((len(times), Xs.shape[0], Xs.shape[1]), dtype=bool)
        self.Xs = Xs
        self.Ys = Ys
        self.dx = np.unique(Xs)[1] - np.unique(Xs)[0]
        self.dy = np.unique(Ys)[1] - np.unique(Ys)[0]
        if self.dx != self.dy and not allow_uneven_grids:
            raise ValueError("dx and dy must be the same")
        elif self.dx != self.dy and allow_uneven_grids:
            print("Warning: dx and dy are not the same - some calculations may be inaccurate")
        self.time_valid = [False for t in times]

    def conver_time_to_ind(self, time):
        return self.times.tolist().index(time)
    
    def check_valid_time(self, time):
        t=self.conver_time_to_ind(time)
        return self.time_valid[t]
    
    def count_number_valid_times(self):
        return np.sum(self.time_valid)
    
    def add_mask(self, time, mask):
        t=self.conver_time_to_ind(time)
        self.masks[t,:,:] = mask
        self.time_valid[t] = True

    def get_coords(self, time):
        t=self.conver_time_to_ind(time)
        if not self.time_valid[t]:
            raise ValueError("No mask for this time")
        inds = np.where(self.masks[t,:,:])
        return self.Xs[inds], self.Ys[inds]
    
    def get_centroid(self, time):
        Xs, Ys = self.get_coords(time)
        return np.mean(Xs), np.mean(Ys)
    
    def check_overlap_area(self, time1, mask2):
        t1=self.conver_time_to_ind(time1)
        if not self.time_valid[t1]:
            raise ValueError("No mask for this time")

        overlap = np.logical_and(self.masks[t1,:,:], mask2)
        area = np.sum(overlap) * self.dx * self.dy
        return area
    
    def check_min_dist(self, time1, mask2):
        t1=self.conver_time_to_ind(time1)
        if not self.time_valid[t1]:
            raise ValueError("No mask for this time")
        
        X1, Y1 = self.get_coords(time1)
        i2, j2 = np.where(mask2)
        X2 = self.Xs[i2, j2]
        Y2 = self.Ys[i2, j2]

        if len(X1) == 0 or len(X2) == 0:
            raise ValueError("One of the masks has no points")
        dists = np.sqrt((X1[:, None] - X2[None, :])**2 + (Y1[:, None] - Y2[None, :])**2)
        min_dist = np.min(dists)
        return min_dist
    
    def get_area(self, time):
        t=self.conver_time_to_ind(time)
        if not self.time_valid[t]:
            raise ValueError("No mask for this time")
        area = np.sum(self.masks[t,:,:]) * self.dx * self.dy
        return area
    
    def get_displacement(self, time1, time2):
        time_check = [self.check_valid_time(t) for t in [time1, time2]]
        if not all(time_check):
            print("One of the times has no mask")
            return np.nan,np.nan
        x1, y1 = self.get_centroid(time1)
        x2, y2 = self.get_centroid(time2)
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy

    def get_feret_min_max(self, time):
        import feret

        t=self.conver_time_to_ind(time)
        if not self.time_valid[t]:
            raise ValueError("No mask for this time")
        pix_count = np.sum(self.masks[t,:,:])
        if pix_count < 4:
            Warning("Feret diameters not well defined for very small objects "
            "- forced to discard a few pulses")
            return np.nan, np.nan
        mask_now = self.masks[t,:,:].squeeze().astype(np.uint8) * 255
        f_max = feret.max(mask_now)
        f_min = feret.min(mask_now)

        return f_min * self.dx, f_max * self.dx
    
    def get_lifetime(self):
        valid_times = [t for t in self.times if self.check_valid_time(t)]
        if len(valid_times) == 0:
            return 0.0
        return valid_times[-1] - valid_times[0]
    
    def time_avg_area(self):
        return np.nanmean([self.get_area(t) for t in self.times if self.check_valid_time(t)])
    
    def time_avg_feret(self):
        f_mins = []
        f_maxs = []
        for t in self.times:
            if self.check_valid_time(t):
                f_min, f_max = self.get_feret_min_max(t)
                f_mins.append(f_min)
                f_maxs.append(f_max)
        return np.nanmean(f_mins), np.nanmean(f_maxs)
    
    def get_boundary_coords(self, time):
        from skimage import measure

        t=self.conver_time_to_ind(time)
        if not self.time_valid[t]:
            raise ValueError("No mask for this time")
        mask_now = self.masks[t,:,:].astype(np.uint8) * 255
        contours = measure.find_contours(mask_now, 0.5)
        if len(contours) == 0:
            return np.array([]), np.array([])
        # Find the longest contour
        longest_contour = max(contours, key=len)
        # Convert pixel coordinates to physical coordinates
        # x_coords = self.Xs[0,0] + longest_contour[:, 0] * self.dx
        # y_coords = self.Ys[0,0] + longest_contour[:, 1] * self.dy
        x_inds = longest_contour[:, 0].astype(int)
        y_inds = longest_contour[:, 1].astype(int)
        x_coords = self.Xs[x_inds, y_inds]
        y_coords = self.Ys[x_inds, y_inds]
        return x_coords, y_coords
    
    # Function to associate clusters in time (determine when two clusters at different times are the same pulse which has moved)
def track_pulses(labels,Xs,Ys,times,dist_tol, allow_uneven_grids=False):
    pulses=[] # store the pulses
    
    labels0 = labels[0,:,:]
    label = 0
    for l in np.unique(labels0):
        if l == 0:
            continue
        mask = labels0 == l
        this_pulse = pulse(times, Xs, Ys, allow_uneven_grids=allow_uneven_grids)
        this_pulse.add_mask(times[0], mask)
        pulses.append(this_pulse)
        label += 1
    
    for t in range(1, labels.shape[0]):
        if t==0:
            continue

        prev_time = times[t-1]
        this_time = times[t]
        unclaimed_prev_labels = [p for p in pulses if p.check_valid_time(prev_time)]
        labels_now = np.unique(labels[t,:,:]) # still include background label 0
        labels_now = labels_now[labels_now != 0] # remove background label 0
        counter = 0

        for l in labels_now:
            mask_l = labels[t,:,:] == l
            for p in unclaimed_prev_labels:
                # area_overlap = p.check_overlap_area(prev_time, mask_l)
                # if area_overlap > area_tol:
                min_dist = p.check_min_dist(prev_time, mask_l)
                if min_dist < dist_tol:
                    p.add_mask(this_time, mask_l)
                    unclaimed_prev_labels.remove(p)
                    labels_now = labels_now[labels_now != l] # remove claimed label
                    counter += 1
                    break
        for l in labels_now:
            mask = labels[t,:,:] == l
            this_pulse = pulse(times, Xs, Ys, allow_uneven_grids=allow_uneven_grids)
            this_pulse.add_mask(this_time, mask)
            pulses.append(this_pulse)
            label += 1
    
    return pulses

class pulse_set:

    def __init__(self, times, Xs, Ys, thresh_field, all_pulses):
        self.times = times
        self.Xs = Xs
        self.Ys = Ys
        self.thresh_field = thresh_field
        self.pulse_list = all_pulses 
        self.num_pulses = len(all_pulses)
                             

    def get_pulse(self, label):
        return self.pulse_list[label]
    
    def get_num_pulses_time_dep(self):
        num_pulses_time = []
        for t in self.times:
            num_pulses = np.sum([p.check_valid_time(t) for p in self.pulse_list])
            num_pulses_time.append(num_pulses)
        return np.array(num_pulses_time)
    
    def get_init_site(self, label, growth_check=True, dist_check=None):
        p = self.get_pulse(label)
        for nt, t in enumerate(self.times):
            if p.check_valid_time(t):
                site = p.get_centroid(t)
                valid = True

                # now do an additional checks that this is not due to a splitting event

                # check if the site is within the minimum distance of any other pulse
                if dist_check is not None:
                    if t > self.times[0]:
                        prev_t = self.times[nt-1]
                        for other_p in self.pulse_list:
                            if other_p.check_valid_time(prev_t):
                                min_dist = other_p.check_min_dist(prev_t, p.masks[nt,:,:])
                                if min_dist < dist_check:
                                    valid = False
                                    break

                # check whether pulse grows from this time to the next
                if growth_check and valid:
                    if t < self.times[-1]:
                        next_t = self.times[nt+1]
                        if not p.check_valid_time(next_t):
                            valid = False
                        else:
                            area_now = p.get_area(t)
                            area_next = p.get_area(next_t)
                            if area_next <= area_now:
                                valid = False

                if valid:
                    return site, t
                else:
                    return (np.nan, np.nan), np.nan
        
        raise ValueError("No valid times for this pulse")
                    
    def get_init_sites(self, growth_check=True, dist_check=None):
        sites = []
        times = []
        for l in range(self.num_pulses):
            site, time = self.get_init_site(l, growth_check=growth_check, dist_check=dist_check)
            # print(site, time)
            sites.append(site)
            times.append(time)
        sites = np.array(sites)
        times = np.array(times)
        sites = sites[~np.isnan(times)]
        times = times[~np.isnan(times)]
        return sites, times

    def animate_initiation_sites(self, outfile, fps=20, init_sites=None, init_times=None):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        import numpy as np

        def construct_init_by_time(sites, times):
            init_by_time = {t: [] for t in self.times}
            for site, t in zip(sites, times):
                if not np.any(np.isnan(site)):  # skip NaN results
                    init_by_time[t].append(site)
            # convert lists to numpy arrays for easier plotting
            for t in init_by_time:
                if len(init_by_time[t]) > 0:
                    init_by_time[t] = np.array(init_by_time[t])
                else:
                    init_by_time[t] = np.empty((0, 2))
            return init_by_time

        if init_sites is None or init_times is None:
            print("Computing initiation sites with default parameters")
            init_sites, init_times = self.get_init_sites(growth_check=True, dist_check=1.0)

        init_by_time = construct_init_by_time(init_sites, init_times)

        Xs, Ys, times, thresh_field = self.Xs, self.Ys, self.times, self.thresh_field
        vmin, vmax = np.nanmin(thresh_field), np.nanmax(thresh_field)

        fig, ax = plt.subplots(figsize=(6, 6))
        # use pcolormesh for irregular grids
        field_plot = ax.pcolormesh(Xs, Ys, thresh_field[0, :, :],
                                shading='auto', cmap='gray',
                                vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(field_plot, ax=ax)

        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # create scatter once
        scatter_plot = ax.scatter([], [], s=50, edgecolor='red', facecolor='none', lw=2)

        def init():
            # set the first frame
            field_plot.set_array(thresh_field[0, :, :].ravel())
            scatter_plot.set_offsets(np.empty((0, 2)))
            ax.set_title(f"Time = {times[0]:.2f}, Num Inits = {len(init_by_time[times[0]])}")
            return field_plot, scatter_plot

        def animate(i):
            # update the field
            field_plot.set_array(thresh_field[i, :, :].ravel())
            # update scatter
            current_inits = init_by_time[times[i]]
            if current_inits.size > 0:
                scatter_plot.set_offsets(current_inits)
            else:
                scatter_plot.set_offsets(np.empty((0, 2)))
            # update title
            ax.set_title(f"Time = {times[i]:.2f}, Num Inits = {len(current_inits)}")
            return field_plot, scatter_plot

        ani = animation.FuncAnimation(fig, animate, frames=len(times),
                                    init_func=init, blit=True, interval=1000/fps)
        ani.save(outfile, fps=fps, writer="ffmpeg")
        plt.close(fig)
        return

    def get_all_areas(self):
        areas = []
        for p in self.pulse_list:
            for t in self.times:
                if p.check_valid_time(t):
                    areas.append(p.get_area(t))
        return np.array(areas)
    
    def get_avg_area(self):
        areas = self.get_all_areas()
        return np.nanmean(areas)

    def get_all_displacements(self):
        displacements_x = []
        displacements_y = []
        for p in self.pulse_list:
            valid_times = [t for t in self.times if p.check_valid_time(t)]
            for i in range(len(valid_times)-1):
                t1 = valid_times[i]
                t2 = valid_times[i+1]
                dx, dy = p.get_displacement(t1, t2)
                displacements_x.append(dx)
                displacements_y.append(dy)
        return np.array(displacements_x), np.array(displacements_y)

    def get_avg_feret(self):
        f_mins = []
        f_maxs = []
        for t in self.times:
            for p in self.pulse_list:
                if p.check_valid_time(t):
                    f_min, f_max = p.get_feret_min_max(t)
                    f_mins.append(f_min)
                    f_maxs.append(f_max)
        return np.nanmean(f_mins), np.nanmean(f_maxs)
    
    def get_avg_num_pulses(self):
        num_pulses = []
        for t in self.times:
            num_pulses.append(np.sum([p.check_valid_time(t) for p in self.pulse_list]))
        return np.nanmean(num_pulses)
    
    def get_max_simultaneous_pulses(self):
        num_pulses = []
        for t in self.times:
            num_pulses.append(np.sum([p.check_valid_time(t) for p in self.pulse_list]))
        return np.max(num_pulses), np.argmax(num_pulses), self.times[np.argmax(num_pulses)]
    
    def get_all_lifetimes(self):
        lifetimes = []
        for p in self.pulse_list:
            lifetimes.append(p.get_lifetime())
        return np.array(lifetimes)
    
    def get_pixel_distribution(self):
        num_pix = []
        for p in self.pulse_list:
            for t in self.times:
                if p.check_valid_time(t):
                    Xs, Ys = p.get_coords(t)
                    num_pix.append(len(Xs))
        return np.array(num_pix)
    
    def prune_false_positives(self, min_pix=4):
        valid_labels = []
        for l in range(self.num_pulses):
            p = self.get_pulse(l)
            pix_counts = []
            for t in self.times:
                if p.check_valid_time(t):
                    Xs, Ys = p.get_coords(t)
                    pix_counts.append(len(Xs))
            if np.max(np.array(pix_counts)) < min_pix:
                print(f"Pruning pulse {l} with max pixel count {np.max(pix_counts)}")
            else:
                valid_labels.append(l)
        self.pulse_list = [self.get_pulse(l) for l in valid_labels]
        self.num_pulses = len(self.pulse_list)
    
    def animate_pulse_set(self, outfile, fps=20):
        import matplotlib.pyplot as plt
        from matplotlib import animation

        num_simultaneous = self.get_num_pulses_time_dep()
        Xs = self.Xs
        Ys = self.Ys
        times = self.times
        thresh_field = self.thresh_field
        vmin = np.nanmin(thresh_field)
        vmax = np.nanmax(thresh_field)

        unique_labels = [l for l in range(self.num_pulses)]
        colors = [plt.cm.jet(i) for i in np.linspace(0, 1, len(unique_labels))]
        np.random.shuffle(colors)  # Shuffle colors to avoid similar colors being adjacent
        color_dict = {label: color for label, color in zip(unique_labels, colors)}

        fig, ax = plt.subplots(figsize=(6, 6))
        field_plot = ax.pcolormesh(Xs, Ys, thresh_field[0,:,:], shading='auto', cmap='gray', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(field_plot, ax=ax)

        plot_dict = {l: ax.plot([], [], color=color_dict[l], lw=2)[0] for l in unique_labels}
        text_dict = {l: ax.text(0, 0, '', color=color_dict[l], fontsize=8) for l in unique_labels}

        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        def animate(i):
            field_plot.set_array(thresh_field[i,:,:].ravel())
            ax.set_title(f'Time = {times[i]:.2f}, Num Pulses = {num_simultaneous[i]}')
            for l in unique_labels:
                p = self.get_pulse(l)
                if p.check_valid_time(times[i]):
                    x_coords, y_coords = p.get_boundary_coords(times[i])
                    plot_dict[l].set_data(x_coords, y_coords)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        text_dict[l].set_position((np.mean(x_coords), np.mean(y_coords)))
                        text_dict[l].set_text(f'ID: {l}')
                    else:
                        text_dict[l].set_text('')
                else:
                    plot_dict[l].set_data([], [])
                    text_dict[l].set_text('')
            return [field_plot] + list(plot_dict.values()) + list(text_dict.values())
        
        ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=200, blit=False)
        ani.save(outfile, dpi=200, fps=fps)
        plt.close(fig)

        return
    


def initiation_site_analysis(sites, region_mask, Xs, Ys, rotation_angle):
    import matplotlib.pyplot as plt 

    COM = np.array([np.mean(Xs[region_mask]), np.mean(Ys[region_mask])])
    shifted_Xs = Xs - COM[0]
    shifted_Ys = Ys - COM[1]
    shifted_sites = sites - COM
    bound = mask_bound_2_coords(region_mask, shifted_Xs, shifted_Ys)
    pts = np.vstack((bound[0], bound[1])).T

    rot_mat = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                        [np.sin(rotation_angle),  np.cos(rotation_angle)]])
    sites_rot = np.matmul(shifted_sites, rot_mat.T)
    pts_rot = np.matmul(pts, rot_mat.T)

    max_x = np.max(pts_rot[:,0])
    min_x = np.min(pts_rot[:,0])
    max_y = np.max(pts_rot[:,1])
    min_y = np.min(pts_rot[:,1])
    if max_x - min_x < 1e-5 or max_y - min_y < 1e-5:
        raise ValueError("Region mask is too small or degenerate after rotation.")
    sites_rot[:,0] = np.clip(sites_rot[:,0], min_x, max_x)
    sites_rot[:,1] = np.clip(sites_rot[:,1], min_y, max_y)

    post_counts_mat = np.zeros((3,3))
    mean_coords_mat = np.zeros((3,3,2))

    height_third = (np.max(pts_rot[:,1]) - np.min(pts_rot[:,1])) / 3.0 # height of one third of the cell
    top_third = np.max(pts_rot[:,1]) - height_third
    bottom_third = np.min(pts_rot[:,1]) + height_third

    for s_ind in range(sites_rot.shape[0]):
        s = sites_rot[s_ind,:]

        points_to_left = pts_rot[pts_rot[:,0] <= s[0], :]
        points_to_right = pts_rot[pts_rot[:,0] >= s[0], :]
        
        if points_to_left.shape[0] == 0 or points_to_right.shape[0] == 0:
            print(s[0], np.min(pts_rot[:,0]), np.max(pts_rot[:,0]))
            print(s[1], np.min(pts_rot[:,1]), np.max(pts_rot[:,1]))
            print(s[0]==np.min(pts_rot[:,0]), s[0]==np.max(pts_rot[:,0]))
            print(s[1]==np.min(pts_rot[:,1]), s[1]==np.max(pts_rot[:,1]))

        y_dists_left = np.abs(points_to_left[:,1] - s[1])
        y_dists_right = np.abs(points_to_right[:,1] - s[1])
        closest_lp = points_to_left[np.argmin(y_dists_left), :] 
        closest_rp = points_to_right[np.argmin(y_dists_right), :]

        width = np.linalg.norm(closest_lp - closest_rp)
        left_third = closest_lp[0] + width / 3.0
        right_third = closest_rp[0] - width / 3.0

        if s[1] >= top_third:
            row = 0
        elif s[1] <= bottom_third:
            row = 2
        else:
            row = 1

        if s[0] <= left_third:
            col = 0
        elif s[0] >= right_third:
            col = 2
        else:
            col = 1

        post_counts_mat[row, col] += 1
        mean_coords_mat[row, col, :] += s

    for r in range(3):
        for c in range(3):
            if post_counts_mat[r,c] > 0:
                mean_coords_mat[r,c,:] /= post_counts_mat[r,c]
            else:
                mean_coords_mat[r,c,:] = np.array([np.nan, np.nan])
    
    return post_counts_mat, mean_coords_mat


    
def plot_interpolated_data(Xs, Ys, fields, rect_mask, save_path, time_index=-1, offset=100):
    field = fields[time_index,:,:]
    centre_y = field.shape[1]//2
    lower_xs = Xs[:, centre_y - offset]
    upper_xs = Xs[:, centre_y + offset]
    middle_xs = Xs[:, centre_y]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    c=ax[0].pcolor(Xs, Ys, field, cmap='gray', shading='auto')
    ax[0].plot(lower_xs, Ys[:, centre_y - offset], 'r--', linewidth=1)
    ax[0].plot(upper_xs, Ys[:, centre_y + offset], 'g--', linewidth=1)
    ax[0].plot(middle_xs, Ys[:, centre_y], 'b-', linewidth=1)
    ax[0].set_aspect('equal')
    ax[0].set_title("Interpolated Field")
    ax[0].set_xlabel("X Position")
    ax[0].set_ylabel("Y Position")
    cbar = fig.colorbar(c, ax=ax[0])
    cbar.set_label("Field Intensity")

    ax[1].plot(middle_xs, field[:, centre_y], 'b-', label='Middle Line')
    ax[1].plot(lower_xs, field[:, centre_y - offset], 'r--', label='Lower Line')
    ax[1].plot(upper_xs, field[:, centre_y + offset], 'g--', label='Upper Line')
    ax[1].set_title("Field Intensity Profiles")
    ax[1].set_xlabel("X Position")
    ax[1].set_ylabel("Field Intensity")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, transparent=True)
    plt.close(fig)


def bootstrap_morans_I(field, mask, Xs, Ys, radius=10.0, n_bootstrap=20):
    time_inds = np.arange(field.shape[0])
    # choose n times randomly with replacement
    chosen_times = np.random.choice(time_inds, size=n_bootstrap, replace=True)
    morans_I_values = []
    for n,t in enumerate(chosen_times):
        print(f"Bootstrap Moran's I, {100*n/n_bootstrap:.1f}% complete", end="\r")
        field_t = field[t, :, :]
        I = morans_I(field_t, mask, Xs, Ys, radius=radius)
        morans_I_values.append(I)
    return np.mean(morans_I_values), np.std(morans_I_values)

def av_morans_I_stack(field_stack, mask, Xs, Ys):
    """
    Calculate the average Moran's I over a stack of 2D fields within a given mask.
    """
    morans_I_values = []
    for i in range(field_stack.shape[0]):
        field = field_stack[i, :, :]
        I = morans_I(field, mask, Xs, Ys)
        morans_I_values.append(I)
    return np.mean(morans_I_values), np.std(morans_I_values)


def morans_I(field, mask, Xs, Ys, radius=10.0):
    from scipy.spatial import cKDTree
    valid_points = np.where(mask)
    coords = np.column_stack((Xs[valid_points], Ys[valid_points]))
    values = field[valid_points]
    n = len(values)

    mean_value = np.nanmean(values)
    dev = values - mean_value

    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius)

    numerator = 0.0
    W = 0.0
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist == 0:
            continue
        w = 1.0 / dist
        numerator += w * dev[i] * dev[j] * 2  # symmetry
        W += w * 2

    denominator = np.sum(dev ** 2)
    return (n / W) * (numerator / denominator)


