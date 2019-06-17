print('Importing libraries...')
import matplotlib.pyplot as plt
import numpy as np
from pymc3.stats import hpd
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
    frequencies = run_data[0,:,0:1]
    # Create 2D array with columns frequency and rows chain
    chan_data = run_data[:,:,channel]
    # Medians column
    medians = np.array([np.median(chan_data, axis=0)]).T
    # Credible intervals columns
    # Uses the highest posterior density (HPD), or minimum width Bayesian CI
    # pymc3 uses alpha to mean Type I error probability, so adjust
    credible_intervals = hpd(chan_data, alpha=1-alpha)
    summary_psd = np.hstack((frequencies, medians, credible_intervals))
    return summary_psd

print('Importing data files...')
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
    #time_col = np.full((rows, 1), 1159724317) # GPS time column
    #chain_index = np.full((rows, 1), i) # Chain index column
    #psd_data = np.hstack((time_col, chain_index, psd_data))
    run_data.append(psd_data)
# 3D array
run_data = np.array(run_data)
print('Creating summary PSD...')
summary = psd_summarize(run_data, channel, alpha=0.9)

# Set up figure
print('Initializing figure...')
fig = plt.figure(1)
ax = fig.add_subplot(111)
#print(summary)
plt.fill_between(summary[:,0], summary[:,2], summary[:,3], color='orange')
plt.plot(summary[:,0], summary[:,1])
#plt.xscale('log')
#ax.set_xlim(1e-3, np.max(combined[:-1,2]))
#ax.set_ylim(np.min(combined[:-1,channel+2]), np.max(combined[:-1,channel+2]))
plt.show()
