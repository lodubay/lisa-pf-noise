print('Importing libraries...')
import time_functions as tf
import psd
import plot
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# Parameters
channel = 'a_x'
run = 'run_k'

# Import / generate summary PSD DataFrame
try:
    print('PSD summaries file found. Importing...')
    df = psd.load_summary(run, channel)
except FileNotFoundError:
    print('No PSD summaries file found. Importing ' + run + ' data files...')
    df = psd.save_summary(run, channel)

# Directory and time array stuff
times = tf.get_gps_times(run)
delta_t_days = tf.get_days_elapsed(times)

# Get differences from reference PSD
median = psd.get_median_psd(df)
# Unstack psd, removing all columns except the median
unstacked = psd.unstack_median(df)

# Plot colormaps
print('Plotting...')
fig, axs = plt.subplots(1, 2)
fig.suptitle('Channel ' + channel + ' - median comparison')
# Subplots
axs[0].title.set_text('PSD(t) - PSD_median')
plot.plot_colormap(fig, axs[0], 
    unstacked.sub(median, axis=0), 
    cmap=cm.get_cmap('coolwarm'),
    vlims=(-4e-16,4e-16),
    center=0.0,
    cbar_label='Absolute difference from reference PSD'
)
axs[1].title.set_text('|PSD(t) - PSD_median| / PSD_median')
plot.plot_colormap(fig, axs[1], 
    abs(unstacked.sub(median, axis=0)).div(median, axis=0),
    cmap='PuRd',
    vlims=(0,1)
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
