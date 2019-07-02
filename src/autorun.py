'''
Inputs, summarizes, and automatically generates plots for the given run.
'''

import plot
import time_functions as tf
import psd
import linechain as lc
import pandas as pd
import os
import sys

runs = sys.argv[1:]

for run in runs:
    print('Run: ' + run)
    # File locations
    output_dir = os.path.join('out', run)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    summary_file = os.path.join(output_dir, run + '_summary.pkl')
    model_file = os.path.join(output_dir, run + '_line_evidence.dat')
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
        
        # Time slices - representative sample
        # Time plot file name
        tslice_file = os.path.join(plot_dir, 'tslice' + str(channel) + '.png')
        gps_times = tf.get_gps_times(run)
        line_times = lc.get_lines(run, channel, model_file)
        # Choose 6 times: beginning, end, and 4 evenly drawn from list
        l = len(gps_times)
        indices = [int(i / 5 * l) for i in range(1,5)]
        slice_times = sorted([gps_times[0], gps_times[-1]] +
            [gps_times[i] for i in indices]
        )
        # Plot
        plot.save_time_slices(run, channel, df, slice_times, tslice_file,
            time_format='gps', exact=True, show=False, logpsd=True
        )
        
        # Time slices - all spectral lines
        if len(line_times) > 0:
            tslice_file = os.path.join(plot_dir, 'tslice_lines'+str(channel)+'.png')
            plot.save_time_slices(run, channel, df, line_times, tslice_file,
                time_format='gps', exact=True, show=False, logpsd=True
            )
print('Done!')

