print('Importing libraries...')
import time
import psd
import plot

# Parameters
channel = 1
cols = ['freq', 'a_x', 'a_y', 'a_z', 'theta_x', 'theta_y', 'theta_z']
run = 'run_k'

# Directory and time array stuff
summary_dir = os.path.join('summaries', run)
summary_file = os.path.join(summary_dir, 'summary.' + cols[channel] + '.pkl')
times = time.get_gps_times(run)
delta_t_days = time.get_days_elapsed(times)

# If a summary file doesn't exist, generate it
if not summary_file in glob.glob(os.path.join(summary_dir, '*')):
    print('No PSD summaries file found. Importing data files...')
    psd.save_summary(run, cols[channel])
print('PSD summaries file found. Importing...')
summaries = psd.load_summary(run, cols[channel])
    
# Get differences from reference PSD
median = psd.get_median_psd(summaries)
unstacked = psd.unstack_median(summaries)

# Plot colormaps
print('Plotting...')
#sns.heatmap(unstacked.sub(median, axis=0))
fig, axs = plt.subplots(1, 2)
fig.suptitle('Channel ' + cols[channel] + ' - median comparison')
# Subplots
axs[0].title.set_text('PSD(t) - PSD_median')
plot_colormap(fig, axs[0], 
    unstacked.sub(median, axis=0), 
    times,
    cmap=cm.get_cmap('coolwarm'),
    vlims=(-2e-15,2e-15),
    logfreq=True,
    neutral=0.0,
    cbar_label='Absolute difference from reference PSD'
)
axs[1].title.set_text('|PSD(t) - PSD_median| / PSD_median')
plot_colormap(fig, axs[1], 
    abs(unstacked.sub(median, axis=0)).div(median, axis=0),
    times,
    cmap='PuRd',
    vlims=(0,0.5),
    logfreq=True
)
plt.show()

# Frequency slice
fig, axs = plt.subplots(2,2)
fig.suptitle('Channel ' + cols[channel]
    + ' - PSDs at selected frequencies')
plot_freq_slice(fig, axs[0,0], 0.01, times, summaries, 'b', ylim=(0e-15,3.5e-15))
plot_freq_slice(fig, axs[0,1], 0.10, times, summaries, 'b', ylim=(1e-15, 4.5e-15))
plot_freq_slice(fig, axs[1,0], 0.32, times, summaries, 'b', ylim=(0,2e-14))
plot_freq_slice(fig, axs[1,1], 0.50, times, summaries, 'b', ylim=(0,2e-14))
#plt.show()

# Time slice
fig, axs = plt.subplots(1,1)
fig.suptitle('Channel ' + cols[channel] + ' - PSDs at selected times since '
    + str(time.get_iso_date(times[0])) + ' UTC')
plot_time_slice(fig, axs, 0.32, times, summaries, 'b', logpsd=True)
plot_time_slice(fig, axs, 0.50, times, summaries, 'g')
plot_time_slice(fig, axs, 1.50, times, summaries, 'orange')
plot_time_slice(fig, axs, 2.50, times, summaries, 'r')
axs.title.set_text('')
#plt.show()
