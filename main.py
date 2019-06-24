print('Importing libraries...')
import time_functions as tf
import psd
import plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

# Parameters
run = 'drs/run_q'
channel = 'theta_z'

# Summary file locations
summary_dir = os.path.join('summaries', run)
summary_file = os.path.join(summary_dir, 'summary.pkl')

# Import / generate summary PSD DataFrame
try:
    print('PSD summaries file found. Importing...')
    df = pd.read_pickle(summary_file)
except FileNotFoundError:
    print('No PSD summaries file found. Importing ' + run + ' data files...')
    df = pd.save_summary(run, summary_file)

# Channel-specific things
df = df.loc[channel]
# Get differences from reference PSD
median = psd.get_median_psd(df)
# Unstack psd, removing all columns except the median
unstacked = psd.unstack_median(df)

# Plot colormaps
print('Plotting...')
fig, axs = plt.subplots(1, 2)
fig.suptitle(run + ' channel ' + channel + ' colormap')
# Subplots
axs[0].title.set_text('PSD(t) - PSD_median')
plot.colormap(fig, axs[0], 
    unstacked.sub(median, axis=0), 
    cmap=cm.get_cmap('coolwarm'),
    vlims=(-4e-16,4e-16),
    center=0.0,
    cbar_label='Absolute difference from reference PSD'
)
axs[1].title.set_text('|PSD(t) - PSD_median| / PSD_median')
plot.colormap(fig, axs[1], 
    abs(unstacked.sub(median, axis=0)).div(median, axis=0),
    cmap='PuRd',
    vlims=(0,1)
)
#plt.show()

# Frequency slices
fig, axs = plt.subplots(2,3)
fig.suptitle(run + ' channel ' + channel + ' PSDs at selected frequencies')
# Subplots
plot.freq_slice(fig, axs[0,0], 1e-3, df, ylim=(0, 1.2e-15))
plot.freq_slice(fig, axs[0,1], 3e-3, df)
plot.freq_slice(fig, axs[0,2], 5e-3, df)
plot.freq_slice(fig, axs[1,0], 1e-2, df)
plot.freq_slice(fig, axs[1,1], 3e-2, df)
plot.freq_slice(fig, axs[1,2], 5e-2, df)
# Legend
handles, labels = axs[1,2].get_legend_handles_labels()
fig.legend(handles, labels)
#plt.show()

# Time slice
fig, axs = plt.subplots(1,1)
#fig.suptitle('Channel ' + cols[channel] + ' - PSDs at selected times since '
#    + str(time.get_iso_date(times[0])) + ' UTC')
plot.time_slice(fig, axs, 0.32, df, 'b', logpsd=True)
plot.time_slice(fig, axs, 0.50, df, 'g')
#plot.time_slice(fig, axs, 1.50, df, 'orange')
#plot.time_slice(fig, axs, 2.50, df, 'r')
axs.title.set_text('')
plt.show()
