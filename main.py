print('Importing libraries...')
import time_functions as tf
import psd
import plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

# TODO move fslice and tslice plotting to plot.py

# Parameters
run = 'ltp/run_b2'
channel = 'theta_y'

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

print('Plotting...')
#plot.save_colormaps(run, channel, df, os.path.join('plots', run, channel + '_colormap.png'))

# Frequency slices
#plot.generate_plots(6)
plot.save_freq_slices(run, channel, df, os.path.join('plots', run, channel + '_colormap.png'))

# Time slice
fig, axs = plt.subplots(1,1)
fig.suptitle(run + ' channel ' + channel + ' PSDs at selected times')
plot.time_slice(fig, axs, tf.gps2day(run, 1143963964), df, 'b', logpsd=True)
#plot.time_slice(fig, axs, 0.50, df, 'g')
#plot.time_slice(fig, axs, 0.85, df, 'orange')
#plot.time_slice(fig, axs, 2.50, df, 'r')
axs.title.set_text('')
#plt.show()
