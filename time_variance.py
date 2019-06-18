print('Importing libraries...')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import glob

# TODO set make auto-scaling color bar that excludes extreme values
# TODO look into lambdas for slice function

def import_time(time_dir):
    # Grab the files with a single-digit index first to sort them correctly
    # Assumes file name format 'psd.dat.#' and 'psd.dat.##'
    # Returns a 3D array, formatted (PSD index, frequency, channel)
    #  first column is the frequency
    psd_files = sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9]')))
    psd_files += sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9][0-9]')))
    # Import PSD files into 3D array
    time_data = np.array([np.loadtxt(psd_file) for psd_file in psd_files])
    # Strip rows of 2s
    time_data = time_data[:,np.min(time_data!=2., axis=(0,2))]
    return time_data
    
def summarize_psd(time_data, channel, alpha=0.9,):
    # Parameters:
    #  run: a 3D array of all PSDs for a single time
    #  channel: int from 1-6, the channel index we're interested in
    #  alpha: tuple of percent credible intervals
    # Returns:
    #  summary_psd: a 2D array with the mean PSD function and credible intervals
    #  | frequency | median | mean - CI1 | mean + CI1 | mean - CI2 | ...
    # Frequencies column
    frequencies = time_data[0,:,0:1]
    # Create 2D array with format (frequency, chain index)
    chan_data = time_data[:,:,channel]
    # Medians column
    medians = np.array([np.median(chan_data, axis=0)]).T
    # Credible intervals columns
    # Uses the highest posterior density (HPD), or minimum width Bayesian CI
    # pymc3 uses alpha to mean Type I error probability, so adjust
    credible_intervals = [hpd(chan_data, alpha=1-a) for a in alpha]
    return np.hstack((frequencies, medians) + tuple(credible_intervals))

def plot_colormap(fig, ax, psd, 
        cmap=None, vlims=None, logfreq=True):
    # Parameters:
    #  fig, ax: the figure and axes of the plot
    #  psd: 2D array with shape (frequency, time)
    #  cmap: color map
    #  vlims: tuple, color scale limits
    #  logfreq: if true, scales the y axis logarithmically
    cbar_label = 'Fractional difference from reference PSD'
    if vlims:
        cbar_label += ' (manual scale)'
    elif np.min(psd) > 0:
        vlims = (0, np.max(psd))
    else:
        # Automatically set color scale so 0 is neutral
        colorscale = np.max((np.abs(np.min(psd)), np.max(psd)))
        vlims = (-colorscale, colorscale)
        cbar_label += ' (auto scale)'
    im = ax.imshow(
        psd,
        cmap=cmap,
        aspect='auto',
        origin='lower',
        vmin=vlims[0],
        vmax=vlims[1],
        extent=[min(delta_t_days), max(delta_t_days), 0., 1.]
    )
    if logfreq:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3, top=1.)
    ax.set_xlabel('Time elapsed (days)')
    ax.set_ylabel('Frequency (Hz)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, labelpad=15, rotation=270)
    
def plot_freq_slice(fig, ax, freq, times, summaries, cred, ylim=None):
    # Plots time vs PSD at a specific frequency
    # Parameters:
    #  fig, ax: the figure and axes of the plot
    #  freq: the frequency along which to slice
    #  summaries: 3D array, shape (time, frequency, stats)
    #   with stats arranged | frequency | median | median - CI | median + CI |
    freqs = summaries[0,:,0]
    # Get the index of the nearest frequency to the one requested
    freq_index = int(freq / (np.max(freqs) - np.min(freqs)) * freqs.shape[0])
    ci_index = (2*alpha.index(cred)+2, 2*alpha.index(cred)+3)
    ax.fill_between(times,
        summaries[:,freq_index,ci_index[0]],
        summaries[:,freq_index,ci_index[1]], 
        alpha=0.5,
        label=str(int(100 * cred))+'% credible interval')
    ax.plot(times, summaries[:,freq_index,1], label='Median PSD')
    ax.legend()
    ax.set_xlabel('Time elapsed (days)')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel('PSD')
    ax.title.set_text('PSD at ' + str(freq) + ' Hz')
    
def plot_time_slice(fig, ax, time, times, summaries, cred, logfreq=True, ylim=None):
    # Plots frequency vs PSD at a specific time
    # Parameters:
    #  fig, ax: the figure and axes of the plot
    #  time: the time along which to slice
    #  times: array of times for which data exists
    #  summaries: 3D array, shape (time, frequency, stats)
    #   with stats arranged | frequency | median | median - CI | median + CI |
    #  logfreq: whether to plot frequency on a log scale
    #  ylim: optional y axis limits, tuple
    # Get the index of the nearest time to the one requested
    time_index = int(time / np.max(times) * times.shape[0])
    ci_index = (2*alpha.index(cred)+2, 2*alpha.index(cred)+3)
    ax.fill_between(summaries[time_index,:,0], 
        summaries[time_index,:,ci_index[0]], 
        summaries[time_index,:,ci_index[1]], 
        alpha=0.5,
        label=str(int(100 * cred))+'% credible interval')
    ax.plot(summaries[time_index,:,0], summaries[time_index,:,1], 
        label='Median PSD')
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    if logfreq:
        ax.set_xscale('log')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel('PSD')
    ax.title.set_text('PSD at ' + str(time) + ' days')

# The index of the channel we're interested in
channel = 2
channels = ['freq', 'x', 'y', 'z', 'vx', 'vy', 'vz']
# Credible intervals
alpha = (0.5, 0.9)
# Directories
top_dir = os.getcwd()
run = 'run_k'
run_dir = os.path.join(top_dir, 'data', run)
summary_file = os.path.join(run_dir, 'summary.' + channels[channel] + '.npy')
# Get a list of the time directories
time_dirs = sorted(glob.glob(os.path.join(run_dir, run + '*')))
# Array of run times
times = np.array([int(time_dir[-10:]) for time_dir in time_dirs])
delta_t_days = (times - times[0]) / (60 * 60 * 24)

print('Looking for PSD summaries file...')
# If a summary file already exists, load it
if summary_file in glob.glob(os.path.join(run_dir, '*')):
    print('PSD summaries file found. Importing...')
    summaries = np.load(summary_file)
else:
    print('No PSD summaries file found. Importing data files:')
    from pymc3.stats import hpd
    # Pull PSD files from target run
    summaries = [] # List of summary PSDs, one for each run
    for time_dir in time_dirs:
        time = int(time_dir[-10:]) # 10-digit GPS time
        print('\tImporting ' + str(time) + '...')
        time_data = import_time(time_dir)
        # Create 2D summary array for the desired channel and append to running list
        summary_psd = summarize_psd(time_data, channel, alpha=alpha)
        summaries.append(summary_psd)
    print('Adjusting arrays...')
    # Make all arrays the same length
    rows = min([summary.shape[0] for summary in summaries])
    summaries = [summary[:rows] for summary in summaries]
    # Turn into 3D array
    summaries = np.array(summaries)
    print('Writing to PSD summaries file...')
    np.save(summary_file, summaries)
    
# Get differences from reference PSD
median_psd = np.median(summaries[:,:,1:2], axis=0)
t0_psd = summaries[0,:,1:2]
ref_psd = median_psd
channel_intensity = summaries[:,:,1].T

print('Plotting...')
fig, axs = plt.subplots(1, 3)
fig.suptitle('Channel ' + channels[channel] + ' - median comparison')
# Color map
cmap = cm.get_cmap('coolwarm')
cmap.set_under(color='b')
cmap.set_over(color='r')
# Subplots
axs[0].title.set_text('PSD(t) - PSD_median')
plot_colormap(fig, axs[0], channel_intensity - median_psd,
    cmap=cmap,
    vlims=(-1e-14,1e-14),
    logfreq=True
)
axs[1].title.set_text('PSD(t) / PSD_median')
plot_colormap(fig, axs[1], channel_intensity / median_psd,
    cmap=cmap,
    vlims=(-3,5),
    logfreq=True
)
axs[2].title.set_text('|PSD(t) - PSD_median| / PSD_median')
plot_colormap(fig, axs[2], 
    np.abs(channel_intensity - median_psd) / median_psd,
    cmap='PuRd',
    vlims=(0,2.5),
    logfreq=True
)
plt.show()

cred = 0.9

# Frequency slice
fig, axs = plt.subplots(1,2)
fig.suptitle('PSDs at different frequencies over time with ' + 
    str(int(100 * cred)) + '% credible intervals')
plot_freq_slice(fig, axs[0], 0.01, delta_t_days, summaries, cred, ylim=(0, 1e-14))
plot_freq_slice(fig, axs[1], 0.75, delta_t_days, summaries, cred)
plt.show()

# Time slice
fig, axs = plt.subplots(1, 2)
fig.suptitle('PSD at multiple times with ' + 
    str(int(100 * cred)) + '% credible intervals')
plot_time_slice(fig, axs[0], 0.14, delta_t_days, summaries, cred, ylim=(0, 1e-15))
plot_time_slice(fig, axs[1], 1.8, delta_t_days, summaries, cred, ylim=(0, 1e-14))
plt.show()
