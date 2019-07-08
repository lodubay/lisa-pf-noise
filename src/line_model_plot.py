'''
Plots a colormap of all possible line models for each run and channel
'''

import matplotlib.pyplot as plt
import linechain as lc
import time_functions as tf
import os
import sys
import pandas as pd
import numpy as np

run = 'ltp_run_b'
channel = 5

output_dir = os.path.join('out', run)
if not os.path.exists(output_dir): os.makedirs(output_dir)
counts_file = os.path.join(output_dir, 'linecounts' + str(channel) + '.dat')

# Time stuff
time_dirs = tf.get_time_dirs(run)
gps_times = tf.get_gps_times(run)

if os.path.exists(counts_file):
    counts = pd.read_csv(counts_file, sep=' ', index_col=0)
else:
    counts = []
    for i, t in enumerate(time_dirs):
        # Update progress
        sys.stdout.write('\r' + str(i) + '/' + str(len(time_dirs)))
        sys.stdout.flush()
        # Linechain file
        lc_file = os.path.join(t, 'linechain_channel'+str(channel)+'.dat')
        models = lc.get_counts(lc_file)
        counts.append([int(models[models == m].count()) for m in range(max(models)+1)])

    # Finish progress indicator
    sys.stdout.write('\n')
    sys.stdout.flush()
    counts = pd.DataFrame(counts, index=gps_times)
    
    # Median time step
    dt = int(np.median(
        [gps_times[i+1] - gps_times[i] for i in range(len(gps_times) - 1)]
    ))
    # Check for time gaps and fill with NaN DataFrames
    for i in range(len(gps_times) - 1):
        diff = gps_times[i+1] - gps_times[i]
        if diff > dt + 1:
            # Number of new times to insert
            n = int(np.floor(diff / dt))
            print('Filling ' + str(n) + ' missing times...')
            # List of missing times, with same time interval
            missing_times = [gps_times[i] + dt * k for k in range(1, n + 1)]
            # Create empty DataFrame, append to summaries, and sort
            filler = pd.DataFrame(columns=counts.columns, index=missing_times)
            counts = counts.append(filler).sort_index()
    print('Saving line counts...')
    counts.to_csv(counts_file, sep=' ')

# Plot
# Get start date in UTC
start_date = tf.gps2iso(gps_times[0])
# Change columns from GPS time to days elapsed from start of run
counts.index = pd.Series(tf.gps2day_list(counts.index), name='TIME')
# Convert column headers to ints
counts.columns = [int(c) for c in counts.columns]
# Convert to fraction of total
counts = counts / sum(counts.iloc[0].dropna())
dt = int(np.median(
    [counts.index[i+1] - counts.index[i] for i in range(len(counts.index) - 1)]
))
fig, ax = plt.subplots(1, 1)
ax.title.set_text(run + ' channel ' + str(channel) + ' line models over time')
im = ax.pcolormesh(
    list(counts.index) + [counts.index[-1] + dt],
    list(counts.columns) + [int(counts.columns[-1]) + 1],
    counts.T,
    cmap='PuRd',
    vmax=0.5
)
# Axis labels
ax.set_xlabel('Days elapsed since ' + str(start_date) + ' UTC')
ax.set_ylabel('Line model')
# Put the major ticks at the middle of each cell
ax.set_yticks(counts.columns + 0.5, minor=False)
ax.set_yticklabels(counts.columns)
# Add and label colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Number of hits for line model', labelpad=15, rotation=270)
plt.savefig(os.path.join(output_dir, 'plots', 'linecounts' + str(channel) + '.png'))
plt.show()


