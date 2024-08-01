import fields_methods as fm # all the helper functions for analysing the simulation output\
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import json

input_dir="./cell_12_sim_out" # directory containing the output of the cell_12 simulation
output_dir="./cell_12_sim_analysis" # directory to save the output of the analysis
max_sim_time=300 # maximum simulation time
equil_time=200 # equilibration time (we usually only want to take data after the system has reached a steady state)
os.makedirs(output_dir,exist_ok=True) # create the output directory if it does not exist

times,coords,myo_field=fm.load_fenics_field_xdmf(input_dir,"rho_b","rho_b",max_time=max_sim_time) # load the myosin field
late_times=times[times>equil_time] # get the times after the equilibration time
late_myo=myo_field[times>equil_time,:] # get the myosin field after the equilibration time

times,coords,vel_field=fm.load_fenics_field_xdmf(input_dir,"velocity","velocity",max_time=max_sim_time) # load the velocity field
late_vel=vel_field[times>equil_time,:,:] # get the velocity field after the equilibration time
v_mag=np.sqrt(late_vel[:,0,:]**2+late_vel[:,1,:]**2) # calculate the magnitude of the velocity field as its easier to work with scalar fields
v_phase=np.arctan2(late_vel[:,1,:],late_vel[:,0,:]) 

times,coords,div_v_field=fm.load_fenics_field_xdmf(input_dir,"v_div","div_v",max_time=max_sim_time) # load the divergence of the velocity field
late_div_v=div_v_field[times>equil_time,:] # get the divergence of the velocity field after the equilibration time


# Plot spatial average of the fields as a function of time
def _spatial_av(sig):
    return np.mean(sig,axis=1),np.std(sig,axis=1)/np.sqrt(sig.shape[1])

av_myo,er_myo=_spatial_av(late_myo) # calculate the spatially averaged myosin field
av_v_mag,er_v_mag=_spatial_av(v_mag) # calculate the spatially averaged velocity magnitude
av_v_phase,er_v_phase=_spatial_av(v_phase) # calculate the spatially averaged velocity phase
av_div_v,er_div_v=_spatial_av(late_div_v) # calculate the spatially averaged divergence of the velocity field


# Basic plotting in time
def plot_time_series(ax,times,vals,errors,title,xlabel,ylabel,xlim=None,ylim=None):
    ax.errorbar(times,vals,yerr=errors,label=title,
                fmt="o",ms=4,markerfacecolor='none',markeredgecolor='red',
                capsize=4,elinewidth=2,ecolor='red',
                color='grey',ls='--',lw=2,alpha=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
fig,ax=plt.subplots(2,2,figsize=(10,10))
plot_time_series(ax[0,0],late_times,av_myo,er_myo,"Myosin field","Time","Myosin field",xlim=(280,300))
plot_time_series(ax[0,1],late_times,av_v_mag,er_v_mag,"Velocity magnitude","Time","Velocity magnitude",xlim=(280,300))
plot_time_series(ax[1,0],late_times,av_v_phase,er_v_phase,"Velocity phase","Time","Velocity phase",xlim=(280,300))
plot_time_series(ax[1,1],late_times,av_div_v,er_div_v,"Divergence of velocity field","Time","Divergence of velocity field",xlim=(280,300))
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"time_series.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)
plt.close(fig)  


# Same with power spectra  

dt=late_times[1]-late_times[0] # calculate the time step
fs=1/dt # calculate the sampling frequency

def av_spectra(sig):
    freqs,pxx=signal.periodogram(sig,fs=fs,axis=0,scaling="spectrum",detrend="constant") # calculate the power spectrum
    return freqs,np.mean(pxx,axis=1),np.std(pxx,axis=1)/np.sqrt(pxx.shape[1])

freqs,av_myo_p,er_myo_p=av_spectra(late_myo) # calculate the power spectrum of the myosin field
_,av_v_mag_p,er_v_mag_p=av_spectra(v_mag) # calculate the power spectrum of the velocity magnitude
_,av_v_phase_p,er_v_phase_p=av_spectra(v_phase) # calculate the power spectrum of the velocity phase
_,av_div_v_p,er_div_v_p=av_spectra(late_div_v) # calculate the power spectrum of the divergence of the velocity field

fig,ax=plt.subplots(2,2,figsize=(10,10))
plot_time_series(ax[0,0],freqs,av_myo_p,er_myo_p,"Myosin field","Frequency","Power spectrum",xlim=(0,2))
plot_time_series(ax[0,1],freqs,av_v_mag_p,er_v_mag_p,"Velocity magnitude","Frequency","Power spectrum",xlim=(0,2))
plot_time_series(ax[1,0],freqs,av_v_phase_p,er_v_phase_p,"Velocity phase","Frequency","Power spectrum",xlim=(0,2))
plot_time_series(ax[1,1],freqs,av_div_v_p,er_div_v_p,"Divergence of velocity field","Frequency","Power spectrum",xlim=(0,2))
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"spectra.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)
plt.close(fig)

# Compute time correlation functions using bootstrap to average over multiple spatial points
corls_mm,corls_std_er_mm,dts=fm.bootstrap_time_correl_FT(late_myo,late_myo,np.arange(0,late_div_v.shape[0])*times[1],N_boot=100) # calculate the time autocorrelation function of the myosin field
corls_m_div,corls_std_er_m_div,dts=fm.bootstrap_time_correl_FT(late_myo,late_div_v,np.arange(0,late_div_v.shape[0])*times[1],N_boot=100) # calculate the time cross-correlation function of the myosin field and the divergence of the velocity field

fig,ax=plt.subplots(1,2,figsize=(10,5))
plot_time_series(ax[0],dts,corls_mm,corls_std_er_mm,"Myosin field","Time","Correlation",xlim=(-10,10))
plot_time_series(ax[1],dts,corls_m_div,corls_std_er_m_div,"Myosin field and divergence of velocity field","Time","Correlation",xlim=(-10,10))
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"correlations_time.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)
plt.close(fig)


# spatial correlation functions (note this can give weird values are very large distances)
corls,dXs,dYs=fm.compute_correlation_unstruc(late_myo,coords)

fig,ax=plt.subplots(1,1,figsize=(5,5))
ax.pcolormesh(dXs,dYs,corls,cmap='viridis',vmin=-1,vmax=1)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Correlation of myosin")
fig.savefig(os.path.join(output_dir,"correlation_spatial_2d.pdf"),dpi=400,transparent=True,pad_inches=0.01,bbox_inches='tight')
plt.close(fig)

# find where dx and dy =0
x_ind=dXs.shape[0]//2
y_ind=dYs.shape[1]//2

xs_flat=dXs[:,y_ind].flatten()
corl_flat_x_var=corls[:,y_ind].flatten()
hwhmx=fm.compute_HWHM(xs_flat,corl_flat_x_var)

ys_flat=dYs[x_ind,:].flatten()
corl_flat_y_var=corls[x_ind,:].flatten()
hwhmy=fm.compute_HWHM(ys_flat,corl_flat_y_var)

fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].plot(xs_flat,corl_flat_x_var)
ax[0].set_xlabel("dx")
ax[0].set_ylabel("Correlation")
ax[0].set_title("Correlation vs dx")
ax[0].axvline(hwhmx, color='red',linestyle='--')
ax[1].plot(ys_flat,corl_flat_y_var)
ax[1].set_xlabel("dy")
ax[1].set_ylabel("Correlation")
ax[1].set_title("Correlation vs dy")
ax[1].axvline(hwhmy, color='red',linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"correlation_spatial_1d.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)
plt.close(fig)
    

# Interesting feature not really discussed in the paper is the the following plot
# We compared the direction of the velocity field with the magnitude of the velocity field
# This is done accross all spatial points and all (late) times.

# We see a strong correlation, due to geometry of the cell
fig,ax=plt.subplots(1,1,figsize=(5,5))
ax.plot(v_phase.flatten(),v_mag.flatten(),"o",ms=1)
ax.set_xlabel("Velocity phase")
ax.set_ylabel("Velocity magnitude")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"velocity_phase_vs_magnitude.png"),format="png",transparent=True,bbox_inches="tight",pad_inches=0.01, dpi=300)
plt.close(fig)


# Now let's locate the peaks in spatially averaged myosin field
mean_myo=np.mean(late_myo,axis=1) 
top_myos=np.percentile(mean_myo,80) # look for when the myosin field is in the top 20% of values
bottom_myos=np.percentile(mean_myo,20) # look for when the myosin field is in the bottom 20% of values
myo_peaks,_=signal.find_peaks(mean_myo) # find the peaks in the spatially averaged myosin field
above_mean_peaks=myo_peaks[mean_myo[myo_peaks]>top_myos] # only keep the peaks that are in the top 20% of values
myo_troughs,_=signal.find_peaks(-mean_myo) # find the troughs in the spatially averaged myosin field
above_mean_troughs=myo_troughs[mean_myo[myo_troughs]<bottom_myos] # only keep the troughs that are in the bottom 20% of values

mean_peak_val=np.mean(mean_myo[above_mean_peaks]) # calculate the mean value of the myosin field at the peaks
mean_trough_val=np.mean(mean_myo[above_mean_troughs]) # calculate the mean value of the myosin field at the troughs
std_er_peak_val=np.std(mean_myo[above_mean_peaks])/np.sqrt(len(above_mean_peaks)) # calculate the standard error of the mean of the myosin field at the peaks
std_er_trough_val=np.std(mean_myo[above_mean_troughs])/np.sqrt(len(above_mean_troughs)) # calculate the standard error of the mean of the myosin field at the troughs

max_myos=np.max(late_myo,axis=1) # get the maximum myosin field at each time
min_myos=np.min(late_myo,axis=1) # get the minimum myosin field at each time

mean_max_myos=np.mean(max_myos) # calculate the mean maximum myosin field 
mean_min_myos=np.mean(min_myos) # calculate the mean minimum myosin field

std_er_max_myos=np.std(max_myos)/np.sqrt(len(max_myos)) # calculate the standard error of the mean of the maximum myosin field
std_er_min_myos=np.std(min_myos)/np.sqrt(len(min_myos)) # calculate the standard error of the mean of the minimum myosin field

# plot myo signal and peaks
fig,ax=plt.subplots(1,1,figsize=(5,5))
ax.plot(late_times,mean_myo,label="Myosin field",marker='None',linestyle='-',color='black',linewidth=1)
ax.plot(late_times[above_mean_peaks],mean_myo[above_mean_peaks],"o",label="Peaks",color='red')
ax.plot(late_times[above_mean_troughs],mean_myo[above_mean_troughs],"o",label="Troughs",color='blue')
ax.set_xlabel("Time")
ax.set_ylabel("Myosin field")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"myosin_field_peaks.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)


# Now let's try and locate the contracted regions
# We will do this by looking for "bright" regions in the myosin field

connecs=fm.get_msh_conec(os.path.join(input_dir,"rho_b.xdmf")) # get the connectivity of the mesh
bound_points=fm.get_msh_boundary_points(os.path.join(input_dir,"rho_b.xdmf")) # get the boundary points of the mesh
# look at late myosin field, look for at least 2 points when the myosin field is 1.5 standard deviations above the mean
clusters=fm.cluster_time_fenics_dat(late_myo,connecs,num_std=1.5,min_cluster_size=2,pulses="high") # cluster the myosin field

# Connect up clusters we found in time 
all_pulses=fm.track_pulses(clusters,coords,late_times,dist_tol=2)
my_PS=fm.pulse_set(late_times,coords,late_myo,all_pulses)

# Now we can get some information about the pulses
num_pulses_in_time=np.array(my_PS.get_num_pulses_time_dep()) # get the number of pulses at each time
vels_in_time=my_PS.get_pulse_averaged_velocities() # get the velcoity averaged over all pulses at each time
vel_mat=my_PS.get_vel_matrix() # compute the velocity matrix, ie velcoity of each pulse at each time
vel_mags=np.sqrt(vel_mat[:,:,0]**2+vel_mat[:,:,1]**2).flatten() # compute the magnitude of the velocity of each pulse at each time
vel_ang=np.arctan2(vel_mat[:,:,1],vel_mat[:,:,0]).flatten() # compute the angle of the velocity of each pulse at each time
AR=my_PS.get_pulse_averaged_aspect_ratios()  # get the feret aspect ratio averaged over all pulses at each time
num_pulses_at_peaks=num_pulses_in_time[above_mean_peaks] # get the number of CRs present at the peaks

# Average quantities over time
av_num_pulses=np.mean(num_pulses_in_time)
max_num_pulses=float(np.max(num_pulses_in_time))
mean_peak_num_pulses=np.mean(num_pulses_at_peaks)
av_vel=np.nanmean(vel_mags)
AR=float(np.nanmean(AR))


# Get basic information about the initiation sites of the pulses
init_sites=my_PS.get_all_init_sites() # get the initiation sites of the pulses
init_site_dict=fm.initation_site_analysis(init_sites[:,0:2],bound_points[:,0:2]) # analyse the locations of the initiation sites
init_site_mat=init_site_dict["pos_count_mat"] # get the matrix of the number of pulses initiated at each segment (this is shown in fig 4 of the paper)
nn_ar=init_site_dict["pulse_neighbours"] # get the list of nums of nearby sites for each site
bound_points_rot=init_site_dict["bounds_rot"] # get the rotated boundary points such that the long axis of the cell is vertical
init_sites_rot=init_site_dict["sites_rot"] # get the rotated initiation sites

# Two different ways to plot the initiation sites
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].scatter(bound_points_rot[:,0],bound_points_rot[:,1],c='black',s=10,marker='o')
c=ax[0].scatter(init_sites_rot[:,0],init_sites_rot[:,1],c=nn_ar,cmap='inferno',s=10,marker='o')
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_aspect('equal')
cb1=fig.colorbar(c,ax=ax[0])
cb1.set_label("Number of nearby initiation sites")
ax[0].set_title(" All initiation sites")
c2=ax[1].imshow(init_site_mat,cmap='inferno',origin='lower')
ax[1].set_title("Segmented initiation sites")
cb2=fig.colorbar(c2,ax=ax[1])
cb2.set_label("Number of pulses in segment")
# don't show x and y axis
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"initiation_sites.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)
plt.close(fig)


# Animate some of the data to give an idea of the dynmaics
# Note that since the shape is non-convex python isn't great at plotting the mesh so it is better to use paraview

fm.animate_tripfield(late_times,coords,late_myo,os.path.join(output_dir,"myosin_field.mp4"),title="Myosin Field")
fm.animate_field_and_pulses(late_times,coords,late_myo,all_pulses,os.path.join(output_dir,"myosin_field_pulses"))


# Save some of these quatities to an output file
pulse_info_dict={"av_num_pulses":av_num_pulses,
                 "std_err_num_pulses":np.std(num_pulses_in_time)/np.sqrt(len(num_pulses_in_time)),
                 "max_num_pulses":max_num_pulses,
                 "peak_num_pulses":mean_peak_num_pulses,
                "av_vel":av_vel,
                "AR":AR,
                "mean_peak_val":float(mean_peak_val),
                "std_er_peak_val":float(std_er_peak_val),
                "mean_trough_val":float(mean_trough_val),
                "std_er_trough_val":float(std_er_trough_val),
                "mean_max_myos":float(mean_max_myos),
                "std_er_max_myos":float(std_er_max_myos),
                "mean_min_myos":float(mean_min_myos),
                "std_er_min_myos":float(std_er_min_myos),
                }

pulse_file=os.path.join(output_dir,"summary_info.json")
with open(pulse_file,'w') as f:
    json.dump(pulse_info_dict,f)