print('Importing libraries...')
import time_functions as tf
import psd
import plot
import linechain as lc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

# Parameters
run = 'ltp_run_b2'
channel = 'theta_z'
chan_idx = 5
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
    print('PSD summaries file found. Importing...')
    df = pd.read_pickle(summary_file)
except FileNotFoundError:
    print('No PSD summaries file found. Generating...')
    df = psd.save_summary(run, summary_file)

print('Plotting...')
# Colormap
cmap_file = os.path.join(plot_dir, 'colormap' + chan_idx + '.png')
plot.save_colormaps(run, channel, df, cmap_file, show=True)

# Frequency slices
fslice_file = os.path.join(plot_dir, 'fslice' + chan_idx + '.png')
plot.save_freq_slices(run, channel, df, fslice_file, show=True)

# Time slice
times = lc.get_lines(run, chan_idx, line_evidence_file, lc_threshold)
print('Plotting times...')
tslice_file = os.path.join(plot_dir, 'tslice' + chan_idx + '.png')
plot.save_time_slices(run, channel, df, 
    times[:min(8, len(times)+1)], tslice_file,
    time_format='gps', exact=True, show=True, logpsd=True
)
