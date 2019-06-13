import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D # Registers the 3D projection
import corner
import numpy as np
import scipy.stats
import os
import glob

def psd_summarize(run_data, channel, alpha=0.9):
    # Parameters:
    #  run: a 3D array of all PSDs for a single run
    #  channel: int from 1-6, the channel index we're interested in
    #  alpha: percent credible interval
    # Returns:
    #  summary_psd: a 2D array with the mean PSD function and credible intervals
    #  | frequency | mean | mean - CI | mean + CI |
    summary_psd = []
    # For each row, find the mean and credible interval across the chain
    for j in range(run_data.shape[1]):
        mean, var, std = stats.mvsdist(run_data[:,j,channel])
        summary_psd.append(np.array([
            run_data[:,j,0], 
            mean.mean(), 
            mean.interval(alpha)[0], 
            mean.interval(alpha)[1]
        ]))
    return summary_psd

# The index of the channel we're interested in
channel = 1
# Run directories
run_dir = 'data/run_k/run_k_1159724317/'
os.chdir(run_dir)
# Grab the files with a single-digit index first to sort them correctly
# Assumes file name format 'psd.dat.#' and 'psd.dat.##'
psd_files = sorted(glob.glob('psd.dat.[0-9]'))
psd_files += sorted(glob.glob('psd.dat.[0-9][0-9]'))
# Import PSD files
run_data = []
for i in range(len(psd_files)):
    psd_data = np.loadtxt(psd_files[i])
    # Remove row of 2s
    if psd_data[-1,-1] == 2.0:
        psd_data = np.delete(psd_data, -1, 0)
    rows = psd_data.shape[0] # Number of rows in the psd.dat file
    time_col = np.full((rows, 1), 1159724317) # GPS time column
    chain_index = np.full((rows, 1), i) # Chain index column
    psd_data = np.hstack((time_col, chain_index, psd_data))
    run_data.append(psd_data)
# 2D array: GPS time | chain index | frequency | channel 1 | channel 2 | ...
run_data = np.array(run_data)
combined = np.concatenate(run_data)
median = np.median(run_data[:,:,channel+2], axis=0)

# Set up figure
print('Initializing figure...')
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.scatter(combined[:-1,2], combined[:-1,channel+2], marker='.')
plt.xscale('log')
ax.set_xlim(1e-3, np.max(combined[:-1,2]))
ax.set_ylim(np.min(combined[:-1,channel+2]), np.max(combined[:-1,channel+2]))
#plt.show()
