print('Importing libraries...')
import time_functions as tf
import psd
import plot
import linechain as lc
import pandas as pd
import numpy as np
import os

# Parameters
run = 'run_k'
lc_threshold = 0.5

# File locations
output_dir = os.path.join('out', run)
if not os.path.exists(output_dir): os.makedirs(output_dir)
summary_file = os.path.join(output_dir, run + '_summary.pkl')
line_evidence_file = os.path.join(output_dir, 
        run + '_line_evidence_' + str(int(100 * lc_threshold)) + '.dat')
plot_dir = os.path.join('out', run, 'plots')
if not os.path.exists(plot_dir): os.makedirs(plot_dir)

# Import / generate summary PSD DataFrame
try:
    df = pd.read_pickle(summary_file)
    print('Imported PSD summaries file.')
except FileNotFoundError:
    print('No PSD summaries file found. Generating...')
    df = psd.save_summary(run, summary_file)

for channel in range(6):
    print('Plotting channel ' + str(channel) + '...')
    # Colormap
    cmap_file = os.path.join(plot_dir, 'colormap' + str(channel) + '.png')
    plot.save_colormaps(run, channel, df, cmap_file, show=False)

    # Frequency slices
    fslice_file = os.path.join(plot_dir, 'fslice' + str(channel) + '.png')
    plot.save_freq_slices(run, channel, df, fslice_file, show=False)

    # Time slice - first look for times with lines
    times = lc.get_lines(run, channel, line_evidence_file, lc_threshold)
    tslice_file = os.path.join(plot_dir, 'tslice' + str(channel) + '.png')
    # Still plot something if no lines are found
    if len(times) == 0:
        gps_times = tf.get_gps_times(run)
        times = gps_times[slice(0, len(gps_times), 3)]
    plot.save_time_slices(run, channel, df, 
        times[:min(8, len(times)+1)], tslice_file,
        time_format='gps', exact=True, show=False, logpsd=True
    )
print('Done!')
