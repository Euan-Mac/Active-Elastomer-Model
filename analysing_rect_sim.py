import fields_methods as fm # all the helper functions for analysing the simulation output
import fields_methods_im as fmi # helper functions for analysing field in a more image based way
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import json

input_dir="./rect_sim_out_not_PBC" # directory containing the output of the rectangular simulation without periodic boundary conditions
output_dir="./rect_sim_analysis" # directory to save the output of the analysis
max_sim_time=300 # maximum simulation time
equil_time=200 # equilibration time (we usually only want to take data after the system has reached a steady state)
os.makedirs(output_dir,exist_ok=True) # create the output directory if it does not exist

check_fields_dir="./check_fields" # directory to save the check files for the fields - checks fields have been loaded correctly
os.makedirs(check_fields_dir,exist_ok=True)


# Load in the fields
    
late_times, late_myo, mesh_ob  = fm.load_point_data_timeseries(os.path.join(input_dir,"rho_b.xdmf"), equil_time, "rho_b")
late_myo_series = fm.FenicsTimeSeries(os.path.join(input_dir,"rho_b.xdmf"), "rho_b", late_times, vector=False, checkfile=os.path.join(check_fields_dir,"check_rho_b.xdmf"))


late_times, late_vel, _  = fm.load_point_data_timeseries(os.path.join(input_dir,"velocity.xdmf"), equil_time, "velocity")
late_vel_series = fm.FenicsTimeSeries(os.path.join(input_dir,"velocity.xdmf"), "velocity", late_times, vector=True)
mag_late_vel=late_vel_series.magnitude() # calculate the magnitude of the velocity field
phase_late_vel=late_vel_series.phase() # calculate the phase of the velocity field

late_times, late_div_v, _ = fm.load_point_data_timeseries(os.path.join(input_dir,"v_div.xdmf"), equil_time, "div_v")
late_div_v_series = fm.FenicsTimeSeries(os.path.join(input_dir,"v_div.xdmf"), "div_v", late_times, vector=False, checkfile=os.path.join(check_fields_dir,"check_div_v.xdmf"))

av_myo = late_myo_series.spatial_mean() # calculate the spatially averaged myosin field
er_myo = late_myo_series.spatial_std() # calculate the standard deviation of the myosin field
av_div_v = late_div_v_series.spatial_mean() # calculate the spatially averaged divergence of the velocity field
er_div_v = late_div_v_series.spatial_std() # calculate the standard deviation of the divergence of the velocity field
av_v_mag = mag_late_vel.spatial_mean() # calculate the spatially averaged velocity magnitude
er_v_mag = mag_late_vel.spatial_std() # calculate the standard deviation of the velocity magnitude
av_v_phase = phase_late_vel.spatial_mean() # calculate the spatially averaged velocity phase
er_v_phase = phase_late_vel.spatial_std() # calculate the standard deviation of the velocity phase
    
    
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
freqs, av_myo_p, er_myo_p, _, _ = fm.fencis_periodogram_time(late_myo_series) # calculate the power spectrum of the myosin field averaged over space
_, av_v_mag_p, er_v_mag_p, _, _ = fm.fencis_periodogram_time(mag_late_vel) # calculate the power spectrum of the velocity magnitude averaged over space
_, av_v_phase_p, er_v_phase_p, _, _ = fm.fencis_periodogram_time(phase_late_vel) # calculate the power spectrum of the velocity phase averaged over space
_, av_div_v_p, er_div_v_p, _, _ = fm.fencis_periodogram_time(late_div_v_series) # calculate the power spectrum of the divergence of the velocity field averaged

fig,ax=plt.subplots(2,2,figsize=(10,10))
plot_time_series(ax[0,0],freqs,av_myo_p,er_myo_p,"Myosin field","Frequency","Power spectrum",xlim=(0,2))
plot_time_series(ax[0,1],freqs,av_v_mag_p,er_v_mag_p,"Velocity magnitude","Frequency","Power spectrum",xlim=(0,2))
plot_time_series(ax[1,0],freqs,av_v_phase_p,er_v_phase_p,"Velocity phase","Frequency","Power spectrum",xlim=(0,2))
plot_time_series(ax[1,1],freqs,av_div_v_p,er_div_v_p,"Divergence of velocity field","Frequency","Power spectrum",xlim=(0,2))
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"spectra.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)
plt.close(fig)

dts, corls_mm, corl_std_mm, _ = fm.fenics_correlate_time_FT(late_myo_series, late_myo_series) # calculate the time autocorrelation function of the myosin field
dts, corls_m_div, corls_std_er_m_div, _ = fm.fenics_correlate_time_FT(late_myo_series, late_div_v_series) # calculate the time cross-correlation function of the myosin field and the divergence of the velocity field
    
fig,ax=plt.subplots(1,2,figsize=(10,5))
plot_time_series(ax[0],dts,corls_mm,corl_std_mm,"Myosin field","Time","Correlation",xlim=(-10,10))
plot_time_series(ax[1],dts,corls_m_div,corls_std_er_m_div,"Myosin field and divergence of velocity field","Time","Correlation",xlim=(-10,10))
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"correlations_time.pdf"),format="pdf",transparent=True,bbox_inches="tight",pad_inches=0.01)
plt.close(fig)



# # pulse tracking
pulse_info_dir = os.path.join(output_dir, "pulse_info")
os.makedirs(pulse_info_dir, exist_ok=True)

field_masked, Xs, Ys, rect_mask = fmi.interpolate_to_rect(late_myo, late_times, mesh_ob, do_print=True, N=300) # interpolate the myosin field to a rectangular grid
fmi.plot_interpolated_data(Xs, Ys, field_masked, rect_mask, os.path.join(output_dir, "interpolated_field.png")) # plot the interpolated field to check it looks ok

# threshold data
thresh=1.0 # threshold for CR brightness criterion in terms of standard deviations above the mean
mean_myo = late_myo_series.space_time_mean() # calculate the mean myosin field over space and time
std_myo = late_myo_series.space_time_std() # calculate the standard deviation of the myosin field over space and time
thresh = mean_myo + thresh * std_myo # set the threshold value
print(f"Using threshold value of {thresh:.2f} for pulse detection")
CR_masks  = field_masked > thresh # create binary masks of where the myosin field is above the threshold
labels, labels_filtered = fmi.run_CC_analysis(CR_masks, smallest_alowed_pix=1) # find connected components

# track ccs to find pulses
pulses_list = fmi.track_pulses(labels_filtered, Xs, Ys, late_times, 1.0) # track the connected components to find pulses
my_PS = fmi.pulse_set(late_times, Xs, Ys, field_masked, pulses_list) # create a pulse set object - this has lots of useful methods for analysing the pulses

pix_counts = my_PS.get_pixel_distribution() # get the distribution of number of pixels in each pulse
# warn if any pulses have very few pixels and thus may be false positives
if np.any(pix_counts < 4):
    print(f"Warning: Some ({100*np.sum(pix_counts<4)/len(pix_counts):.2f}% of all) pulses have very few pixels (<4)")

# save pulse tracks animation
my_PS.prune_false_positives(min_pix = 5) # remove pulses with fewer than 5 pixels to reduce false positives
my_PS.animate_pulse_set(os.path.join(pulse_info_dir, "pulse_tracks.mp4")) # save an animation of the pulse tracks

# find key pulse statistics and save
N_cr = my_PS.get_avg_num_pulses() # get the average number of pulses present at any one time
feret_mean = my_PS.get_avg_feret() # get the mean feret diameters of the pulses
all_areas = my_PS.get_all_areas()
mean_area = np.nanmean(all_areas) # calculate the mean area of pulses

all_lifetimes = my_PS.get_all_lifetimes() # get the lifetimes of all pulses
mean_lifetime = np.nanmean(all_lifetimes) # calculate the mean lifetime of pulses
std_lifetime = np.nanstd(all_lifetimes) # calculate the standard deviation of the lifetimes
max_N, max_N_frame, max_N_time = my_PS.get_max_simultaneous_pulses() # get the maximum number of simultaneous pulses and when this occurs
x_displacemnts, y_displacements = my_PS.get_all_displacements() # get the displacements directions of all pulses

# save out pulse statistics
areas_file = os.path.join(pulse_info_dir, "pulse_areas.txt") 
np.savetxt(areas_file, all_areas)

displacements_file = os.path.join(pulse_info_dir, "pulse_displacements.txt")
np.savetxt(displacements_file, np.vstack((x_displacemnts, y_displacements)).T)

lifetimes_file = os.path.join(pulse_info_dir, "pulse_lifetimes.txt")
np.savetxt(lifetimes_file, all_lifetimes)

init_sites, init_times = my_PS.get_init_sites(growth_check=True, dist_check=1.0) # get the initiation sites of all pulses
init_site_counts, mean_coords = fmi.initiation_site_analysis(init_sites, rect_mask, Xs, Ys,  rotation_angle=rot_angle) # analyse the initiation sites to find their spatial distribution
# we rotate the coodinates to align with the rectangle axes by rot angle

# save out initiation site data
np.savetxt(os.path.join(pulse_info_dir, "initiation_site_counts.txt"), init_site_counts)
my_PS.animate_initiation_sites(os.path.join(pulse_info_dir, "initiation_sites.mp4"), fps=5, init_sites=init_sites, init_times=init_times)
np.savetxt(os.path.join(pulse_info_dir, "initiation_site_counts.txt"), init_site_counts)

# save initiation site plots
fig,axs=plt.subplots(1,2,figsize=(5,2.5))
ax=axs[0]
ax.pcolor(Xs, Ys, rect_mask, cmap='Greys', shading='auto', alpha=0.3)
c=ax.scatter(init_sites[:,0], init_sites[:,1], s=5, alpha=0.5, c=init_times, cmap='plasma')
cbar=fig.colorbar(c,ax=ax)
cbar.set_label("Initiation Time",fontsize=8)
ax.set_title("Initiation Sites Overlayed on Cell Mask",fontsize=8)
ax.set_xlabel("X Position",fontsize=8)
ax.set_ylabel("Y Position",fontsize=8)
ax.set_aspect('equal')

ax=axs[1]
c=ax.matshow(init_site_counts,cmap='inferno')
ax.set_title("Initiation Sites",fontsize=8)
cbar=fig.colorbar(c,ax=ax)
cbar.set_label("Number of Counts",fontsize=8)
#turn off axis
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(pulse_info_dir,"init_sites.pdf"),transparent=True,pad_inches=0.01,bbox_inches='tight')
plt.close(fig)


# peak and trough analysis
mean_myo = late_myo_series.spatial_mean() # calculate the spatially averaged myosin field -i.e. a time series of mean myosin
top_myos = np.percentile(mean_myo, 80) # define thresholds for peaks and troughs
bottom_myos = np.percentile(mean_myo, 20) # define thresholds for peaks and troughs
myo_peaks,_=signal.find_peaks(mean_myo) # find peaks in the mean myosin time series
above_mean_peaks=myo_peaks[mean_myo[myo_peaks]>top_myos] # only keep peaks above the threshold
myo_troughs,_=signal.find_peaks(-mean_myo) # find troughs in the mean myosin time series
above_mean_troughs=myo_troughs[mean_myo[myo_troughs]<bottom_myos] # only keep troughs below the threshold

mean_peak_val=np.mean(mean_myo[above_mean_peaks]) # calculate mean value of peaks
mean_trough_val=np.mean(mean_myo[above_mean_troughs]) # calculate mean value of troughs
std_peak_val=np.std(mean_myo[above_mean_peaks]) # calculate standard deviation of peaks
std_trough_val=np.std(mean_myo[above_mean_troughs]) # calculate standard deviation of troughs
std_er_peak_val=std_peak_val/np.sqrt(len(above_mean_peaks)) # calculate standard error of peaks
std_er_trough_val=std_trough_val/np.sqrt(len(above_mean_troughs)) # calculate standard error of troughs

# save out myosin peaks plot
fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=400)
ax.plot(late_times,mean_myo)
ax.scatter(late_times[above_mean_peaks],mean_myo[above_mean_peaks],color='r',s=10)
ax.scatter(late_times[above_mean_troughs],mean_myo[above_mean_troughs],color='b',s=10)
ax.legend([r"$\rho_b$","Peaks","Troughs"])
ax.set_xlabel("Time")
ax.set_ylabel(r"$\rho_b$")
ax.set_title("Myosin Peaks")
fig.savefig(os.path.join(pulse_info_dir,"myo_peaks.pdf"),transparent=True,pad_inches=0.01,bbox_inches='tight')

num_pulses_in_time = my_PS.get_num_pulses_time_dep() # get the number of pulses present at each time point
num_pulses_at_peaks=num_pulses_in_time[above_mean_peaks] # get the number of pulses present at each peak time
mean_peak_num_pulses=np.mean(num_pulses_at_peaks) # calculate the mean number of pulses present at peaks

# save pulse statistics
pulse_info_dir = os.path.join(output_dir, "pulse_info")
pulse_info_dict={"av_num_pulses":float(N_cr),
                    "std_num_pulses":float(np.std(num_pulses_in_time)),
                    "std_err_num_pulses":float(np.std(num_pulses_in_time)/np.sqrt(len(num_pulses_in_time))),
                    "long_ax": float(feret_mean[1]),
                    "short_ax": float(feret_mean[0]),
                    "area": float(mean_area),
                    "max_num_pulses": int(max_N),
                    "max_num_pulses_frame": int(max_N_frame),
                    "max_num_pulses_time": float(max_N_time),
                    "peak_num_pulses": float(mean_peak_num_pulses),
                    "mean_lifetime": float(mean_lifetime),
                    "std_lifetime": float(std_lifetime)
                }

os.makedirs(pulse_info_dir, exist_ok=True)
pulse_file=os.path.join(pulse_info_dir,"pulse_info.json")
with open(pulse_file,'w') as f:
    json.dump(pulse_info_dict,f)

# save num pulses at peaks
np.savetxt(os.path.join(pulse_info_dir,"num_pulses_at_peaks.txt"), num_pulses_at_peaks)


# create animation of the myosin field
print("Animating myosin field")
fm.animate_tripfield_mesh(late_times,mesh_ob,late_myo,output_dir=os.path.join(output_dir,"myosin_field.mp4"),title="Myosin Field")
