print('Importing libraries...')
import matplotlib.pyplot as plt
import numpy as np
from pymc3.stats import hpd
import os
import glob

def import_run(run_dir):
    # Grab the files with a single-digit index first to sort them correctly
    # Assumes file name format 'psd.dat.#' and 'psd.dat.##'
    psd_files = sorted(glob.glob(run_dir + 'psd.dat.[0-9]'))
    psd_files += sorted(glob.glob(run_dir + 'psd.dat.[0-9][0-9]'))
    # Import PSD files into 3D array
    run_data = np.array([np.loadtxt(psd_file) for psd_file in psd_files])
    # Strip rows of 2s
    run_data = run_data[:,np.min(run_data!=2., axis=(0,2))]
    return run_data # A 3D array
    

def summarize_psd(run_data, channel, alpha=0.9):
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

def get_reference_psd(summary_psds, channel):
    return summary_psds[0][:,channel]

print('Importing data files:')
# The index of the channel we're interested in
channel = 1
# Current directory
top_dir = os.getcwd()
# List of the run directories. Only using a few for testing purposes
run_dirs = sorted(glob.glob('data/run_k/run_k_*/'))[0:4] 

# Pull PSD files from target run
summaries = [] # List of summary PSDs, one for each run
times = [] # List of GPS times corresponding to each run
for run_dir in run_dirs:
    time = int(run_dir[-11:-1]) # 10-digit GPS time
    times.append(time)
    print('\tImporting ' + str(time) + '...')
    run_data = import_run(run_dir)
    # Create 2D summary array and append to running list
    summaries.append(summarize_psd(run_data, channel, alpha=0.9))

# Turn into 3D array
summaries = np.array(summaries)
times = np.array(times)

ref_psd = get_reference_psd(summaries, channel)
print(ref_psd)
