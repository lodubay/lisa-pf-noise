print('Importing libraries...')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pymc3.stats import hpd
import os
import glob

def import_time(time_dir):
    # Grab the files with a single-digit index first to sort them correctly
    # Assumes file name format 'psd.dat.#' and 'psd.dat.##'
    # Returns a 3D array, formatted (PSD index, frequency, channel)
    #  first column is the frequency
    psd_files = sorted(glob.glob(time_dir + 'psd.dat.[0-9]'))
    psd_files += sorted(glob.glob(time_dir + 'psd.dat.[0-9][0-9]'))
    # Import PSD files into 3D array
    time_data = np.array([np.loadtxt(psd_file) for psd_file in psd_files])
    # Strip rows of 2s
    time_data = time_data[:,np.min(time_data!=2., axis=(0,2))]
    return time_data
    
def summarize_psd(time_data, channel, alpha=0.9):
    # Parameters:
    #  run: a 3D array of all PSDs for a single time
    #  channel: int from 1-6, the channel index we're interested in
    #  alpha: percent credible interval
    # Returns:
    #  summary_psd: a 2D array with the mean PSD function and credible intervals
    #  | frequency | median | mean - CI | mean + CI |
    # Frequencies column
    frequencies = time_data[0,:,0:1]
    # Create 2D array with format (frequency, chain index)
    chan_data = time_data[:,:,channel]
    # Medians column
    medians = np.array([np.median(chan_data, axis=0)]).T
    # Credible intervals columns
    # Uses the highest posterior density (HPD), or minimum width Bayesian CI
    # pymc3 uses alpha to mean Type I error probability, so adjust
    credible_intervals = hpd(chan_data, alpha=1-alpha)
    return np.hstack((frequencies, medians, credible_intervals))

def get_reference_psd(summary_psds):
    # Parameters:
    #  summary_psds: 3D array, format (time, frequency, stats), for one channel
    #  channel: the channel index we're interested in
    # Returns a single column, median only
    return summary_psds[0,:,1:2]

def plot_time_colormap(fig, ax, psd_differences, cmap=None, vlims=None):
    # Parameters:
    #  psd_differences: 2D array of differences to reference PSD
    #  cmap: color map
    #  vlims: tuple, color scale limits
    cbar_label = 'Fractional difference from reference PSD'
    if vlims:
        cbar_label += ' (manual scale)'
    else:
        # Automatically set color scale so 0 is neutral
        colorscale = np.max((
            np.abs(np.min(psd_differences)), 
            np.max(psd_differences)
        ))
        vlims = (-colorscale, colorscale)
        cbar_label += ' (auto scale)'
    im = ax.imshow(
        psd_differences,
        cmap=cmap,
        aspect='auto',
        origin='lower',
        vmin=vlims[0],
        vmax=vlims[1],
        extent=[min(times), max(times), 0., 1.]
    )
    ax.set_xlabel('GPS time since 1,159,000,000 s')
    ax.set_ylabel('Frequency (Hz)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270)

print('Importing data files:')
# The index of the channel we're interested in
channel = 1
# Current directory
top_dir = os.getcwd()
# List of the run directories. Only using a few for testing purposes
time_dirs = sorted(glob.glob('data/run_k/run_k_*/'))

# Pull PSD files from target run
summaries = [] # List of summary PSDs, one for each run
times = [] # List of GPS times corresponding to each run
for time_dir in time_dirs:
    time = int(time_dir[-11:-1]) # 10-digit GPS time
    times.append(time)
    print('\tImporting ' + str(time) + '...')
    time_data = import_time(time_dir)
    # Create 2D summary array for the desired channel and append to running list
    summary_psd = summarize_psd(time_data, channel, alpha=0.9)
    summaries.append(summary_psd)

print('Adjusting arrays...')
# Make all arrays the same length
rows = min([summary.shape[0] for summary in summaries])
summaries = [summary[:rows] for summary in summaries]
# Turn into 3D array
summaries = np.array(summaries)
times = np.array(times)
# From here on, we just care about the medians. Will figure out CIs later.
# Get differences from reference PSD
ref_psd = get_reference_psd(summaries)
channel_intensity = summaries[:,:,1].T

print('Plotting...')
fig, axs = plt.subplots(2, 2)
# Color map
cmap = cm.get_cmap('coolwarm')
cmap.set_under(color='k')
cmap.set_over(color='w')
# Subplots
axs[0, 0].title.set_text('(PSD(t) - ref) / PSD')
plot_time_colormap(fig, axs[0, 0], 
    (channel_intensity - ref_psd) / channel_intensity,
    cmap=cmap
)
axs[0, 1].title.set_text('(PSD(t) - ref) / ref')
plot_time_colormap(fig, axs[0, 1], (channel_intensity - ref_psd) / ref_psd,
    cmap=cmap,
    vlims=(-3,3)
)
axs[1, 0].title.set_text('PSD(t) / ref')
plot_time_colormap(fig, axs[1, 0], channel_intensity / ref_psd,
    cmap=cmap,
    vlims=(0,2)
)
axs[1, 1].title.set_text('|PSD(t) - ref| / ref')
plot_time_colormap(fig, axs[1, 1], 
    np.abs(channel_intensity - ref_psd) / ref_psd,
    cmap=cmap,
    vlims=(0,2)
)
plt.show()
