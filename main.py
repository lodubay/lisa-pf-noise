print('Importing libraries...')
import time_functions as tf
import psd
import plot
import linechain as lc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

# TODO move fslice and tslice plotting to plot.py

# Parameters
run = 'ltp/run_b2'
channel = 'theta_y'
chan_idx = 4

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
plot.save_colormaps(run, channel, df, os.path.join('plots', run, channel + '_colormap.png'), show=False)

# Frequency slices
plot.save_freq_slices(run, channel, df, os.path.join('plots', run, channel + '_fslices.png'), show=True)

# Time slice
times = lc.get_lines(run, chan_idx)
print('Plotting times...')
plot.save_time_slices(run, channel, df, 
    times[:7], os.path.join('plots', run, channel + '_tslices.png'),
    time_format='gps', exact=True, show=True
)
