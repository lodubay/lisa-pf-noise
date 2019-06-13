import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D # Registers the 3D projection
import corner
import numpy as np
import scipy.stats
import os
import glob

# Input: a 3D array of PSDs from different points in the MCMC for a single run
def psd_summarize(run, channel, alpha=(0.5, 0.9)):
    summary_psd = []
    # For each row, find the median and credible interval across the chain
    for j in range(run.shape[1]):
        median = np.median(run[:,j,channel])
        #for interval in invervals:
            
        

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

# Individual histogram
plt.hist(run_data[:,2,channel+2], bins=20, density=True)
plt.show()

# Set up figure
print('Initializing figure...')
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.scatter(combined[:-1,2], combined[:-1,channel+2], marker='.')
plt.xscale('log')
ax.set_xlim(1e-3, np.max(combined[:-1,2]))
ax.set_ylim(np.min(combined[:-1,channel+2]), np.max(combined[:-1,channel+2]))
#plt.show()
