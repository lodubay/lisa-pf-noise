print('Importing libraries...')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
import os
import glob
from astropy.time import Time

# TODO set make auto-scaling color bar that excludes extreme values
# TODO look into lambdas for slice function

def get_times(run):
    # Get a list of the time directories
    time_dirs = sorted(glob.glob(os.path.join('data', run, run + '*')))
    # Array of run times
    return np.array([int(time_dir[-10:]) for time_dir in time_dirs])
    
def get_days_elapsed(gps_times):
    return (gps_times - gps_times[0]) / (60 * 60 * 24)
    
def get_iso_date(gps_int):
    gps_time = Time(gps_int, format='gps')
    return Time(gps_time, format='iso')

def import_time(time_dir):
    # Grab the files with a single-digit index first to sort them correctly
    # Assumes file name format 'psd.dat.#' and 'psd.dat.##'
    # Returns a 3D array, formatted (PSD index, frequency, channel)
    #  first column is the frequency
    print('\tImporting ' + time_dir[-11:-1] + '...')
    psd_files = sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9]'))) + \
        sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9][0-9]')))
    # Import PSD files into 3D array
    time_data = np.array([np.loadtxt(psd_file) for psd_file in psd_files])
    # Strip rows of 2s
    return time_data[:,np.min(time_data!=2., axis=(0,2))]
    
def summarize_psd(time_data, channel):
    # Parameters:
    #  run: a 3D array of all PSDs for a single time
    #  channel: int from 1-6, the channel index we're interested in
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
    alpha = (0.5, 0.9)
    credible_intervals = [hpd(chan_data, alpha=1-a) for a in alpha]
    return np.hstack((frequencies, medians) + tuple(credible_intervals))
    
def summarize_run(run, channel):
    # Get a list of the time directories
    time_dirs = sorted(glob.glob(os.path.join('data', run, run + '*/')))
    # Pull PSD files from target run
    print('Importing ' + run + '...')
    # List of summary PSDs (each a 2D array), one for each time
    # Takes a long time
    summaries = [summarize_psd(import_time(d),channel) for d in time_dirs]
    print('Adjusting arrays...')
    # Make all arrays the same length and turn into 3D array
    rows = min([summary.shape[0] for summary in summaries])
    return np.array([summary[:rows] for summary in summaries])
    
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap

def plot_colormap(fig, ax, psd, gps_times, cmap, vlims, 
        logfreq=True, cbar_label=None, neutral=None):
    '''
    Function to plot the colormap of a PSD with frequency on the y-axis and
    time on the x-axis.
    
    Input
    -----
      fig, ax : The figure and axes of the plot
      psd : The PSD, a 2D array with shape (frequency, time)
      cmap : The unaltered color map to use
      vlims : A tuple of the color scale limits
      logfreq : If true, scales the y-axis logarithmically
      cbar_label : Color bar label
      neutral : Value that should be represented by a "neutral" color
    '''
    if not cbar_label: cbar_label = 'Fractional difference from reference PSD'
    if neutral:
        cmap = shiftedColorMap(
            cmap, 
            midpoint=(neutral-vlims[0])/(vlims[1]-vlims[0]), 
            name='shifted colormap'
        )
    im = ax.imshow(
        psd,
        cmap=cmap,
        aspect='auto',
        origin='lower',
        vmin=vlims[0],
        vmax=vlims[1],
        extent=[
            get_days_elapsed(gps_times)[0], 
            get_days_elapsed(gps_times)[-1], 
            0., 1.
        ]
    )
    if logfreq:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3, top=1.)
    ax.set_xlabel('Days elapsed since ' + str(get_iso_date(gps_times[0])) + ' UTC')
    ax.set_ylabel('Frequency (Hz)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, labelpad=15, rotation=270)
    
def plot_freq_slice(fig, ax, freq, gps_times, summaries, color, ylim=None):
    # Plots time vs PSD at a specific frequency
    # Parameters:
    #  fig, ax: the figure and axes of the plot
    #  freq: the frequency along which to slice
    #  summaries: 3D array, shape (time, frequency, stats)
    #   with stats arranged | frequency | median | median - CI | median + CI |
    freqs = summaries[0,:,0]
    # Get the index of the nearest frequency to the one requested
    freq_index = int(freq / (np.max(freqs) - np.min(freqs)) * freqs.shape[0])
    days_elapsed = get_days_elapsed(gps_times)
    ax.fill_between(days_elapsed,
        summaries[:,freq_index,2],
        summaries[:,freq_index,3], 
        color=color,
        alpha=0.5,
        label='50% credible interval at ' + str(freq) + ' Hz')
    ax.fill_between(days_elapsed,
        summaries[:,freq_index,4],
        summaries[:,freq_index,5],
        color=color,
        alpha=0.2,
        label='90% credible interval at ' + str(freq) + ' Hz')
    ax.plot(days_elapsed, summaries[:,freq_index,1], 
        label='Median PSD at ' + str(freq) + ' Hz', color=color)
    ax.legend()
    ax.set_xlabel('Days elapsed since ' + str(get_iso_date(gps_times[0])) + ' UTC')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel('PSD')
    ax.title.set_text('PSD at ' + str(freq) + ' Hz')
    
def plot_time_slice(fig, ax, day, gps_times, summaries, color, 
        logfreq=True, ylim=None, logpsd=False):
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
    days_elapsed = get_days_elapsed(gps_times)
    time_index = int(day / np.max(days_elapsed) * days_elapsed.shape[0])
    ax.fill_between(summaries[time_index,:,0], 
        summaries[time_index,:,2], 
        summaries[time_index,:,3],
        color=color, 
        alpha=0.5,
        label='50% credible interval at ' + str(day) + ' days')
    ax.fill_between(summaries[time_index,:,0], 
        summaries[time_index,:,4], 
        summaries[time_index,:,5],
        color=color, 
        alpha=0.1,
        label='90% credible interval at ' + str(day) + ' days')
    ax.plot(summaries[time_index,:,0], summaries[time_index,:,1], 
        label='Median PSD at T+' + str(day) + ' days', 
        color=color)
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    if logfreq:
        ax.set_xscale('log')
    if ylim:
        ax.set_ylim(ylim)
    if logpsd:
        ax.set_yscale('log')
    ax.set_ylabel('PSD')
    ax.title.set_text('PSD at ' + str(get_iso_date(gps_times[time_index])) + ' UTC')

# The index of the channel we're interested in
channel = 2
channels = ['freq', 'x', 'y', 'z', 'vx', 'vy', 'vz']
# Directories
top_dir = os.getcwd()
run = 'run_k'
run_dir = os.path.join(top_dir, 'data', run)
summary_dir = os.path.join(top_dir, 'summaries', run)
summary_file = os.path.join(summary_dir, 'summary.' + channels[channel] + '.npy')
# Get a list of the time directories
time_dirs = sorted(glob.glob(os.path.join(run_dir, run + '*')))
# Array of run times
times = np.array([int(time_dir[-10:]) for time_dir in time_dirs])
delta_t_days = (times - times[0]) / (60 * 60 * 24)

print('Looking for PSD summaries file...')
# If a summary file already exists, load it
if summary_file in glob.glob(os.path.join(summary_dir, '*')):
    print('PSD summaries file found. Importing...')
    summaries = np.load(summary_file)
else:
    print('No PSD summaries file found. Importing data files:')
    from pymc3.stats import hpd
    summaries = summarize_run(run, channel)
    print('Writing to PSD summaries file...')
    np.save(summary_file, summaries)
    
# Get differences from reference PSD
median_psd = np.median(summaries[:,:,1:2], axis=0)
channel_intensity = summaries[:,:,1].T

print('Plotting...')
fig, axs = plt.subplots(1, 2)
fig.suptitle('Channel ' + channels[channel] + ' - median comparison')
# Subplots
axs[0].title.set_text('PSD(t) - PSD_median')
plot_colormap(fig, axs[0], channel_intensity - median_psd, times,
    cmap=cm.get_cmap('coolwarm'),
    vlims=(-1e-13,1e-13),
    logfreq=True,
    neutral=0.0,
    cbar_label='Absolute difference from reference PSD'
)
axs[1].title.set_text('|PSD(t) - PSD_median| / PSD_median')
plot_colormap(fig, axs[1], 
    np.abs(channel_intensity - median_psd) / median_psd,
    times,
    cmap='PuRd',
    vlims=(0,3),
    logfreq=True
)
plt.show()

cred = 0.9

# Frequency slice
fig, axs = plt.subplots(2,2)
fig.suptitle('Channel ' + channels[channel]
    + ' - PSDs at selected frequencies')
plot_freq_slice(fig, axs[0,0], 0.01, times, summaries, 'b', ylim=(0,1e-14))
plot_freq_slice(fig, axs[0,1], 0.10, times, summaries, 'b', ylim=(1e-15, 1e-14))
plot_freq_slice(fig, axs[1,0], 0.50, times, summaries, 'b', ylim=(2e-13,5e-13))
plot_freq_slice(fig, axs[1,1], 0.99, times, summaries, 'b')
#plt.show()

# Time slice
fig, axs = plt.subplots(1,1)
fig.suptitle('Channel ' + channels[channel] + ' - PSDs at selected times since '
    + str(get_iso_date(times[0])) + ' UTC')
plot_time_slice(fig, axs, 0.30, times, summaries, 'b', logpsd=True)
plot_time_slice(fig, axs, 0.62, times, summaries, 'g')
plot_time_slice(fig, axs, 1.78, times, summaries, 'orange')
plot_time_slice(fig, axs, 1.90, times, summaries, 'r')
axs.title.set_text('')
#plt.show()
