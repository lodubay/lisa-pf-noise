print('Importing libraries...')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
import pandas as pd
import os
import glob
import psd
import time
    
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
            time.get_days_elapsed(gps_times)[0], 
            time.get_days_elapsed(gps_times)[-1], 
            0., 1.
        ]
    )
    if logfreq:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3, top=1.)
    ax.set_xlabel('Days elapsed since ' + str(time.get_iso_date(gps_times[0])) + ' UTC')
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
    freq = np.round(freqs[freq_index], decimals=4)
    days_elapsed = time.get_days_elapsed(gps_times)
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
    ax.set_xlabel('Days elapsed since ' + str(time.get_iso_date(gps_times[0])) + ' UTC')
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
    days_elapsed = time.get_days_elapsed(gps_times)
    time_index = int(day / np.max(days_elapsed) * days_elapsed.shape[0])
    day = np.round(days_elapsed[time_index], decimals=2)
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
    ax.title.set_text('PSD at ' + str(time.get_iso_date(gps_times[time_index])) + ' UTC')

# Parameters
channel = 1
cols = ['freq', 'a_x', 'a_y', 'a_z', 'theta_x', 'theta_y', 'theta_z']
run = 'run_k'

# Directory and time array stuff
summary_dir = os.path.join('summaries', run)
summary_file = os.path.join(summary_dir, 'summary.' + cols[channel] + '.pkl')
times = time.get_gps_times(run)
delta_t_days = time.get_days_elapsed(times)

# If a summary file doesn't exist, generate it
if not summary_file in glob.glob(os.path.join(summary_dir, '*')):
    print('No PSD summaries file found. Importing data files...')
    psd.save_summary(run, cols[channel])
print('PSD summaries file found. Importing...')
summaries = psd.load_summary(run, cols[channel])
    
# Get differences from reference PSD
median = psd.get_median_psd(summaries)
unstacked = psd.unstack_median(summaries)

# Plot colormaps
print('Plotting...')
#sns.heatmap(unstacked.sub(median, axis=0))
fig, axs = plt.subplots(1, 2)
fig.suptitle('Channel ' + cols[channel] + ' - median comparison')
# Subplots
axs[0].title.set_text('PSD(t) - PSD_median')
plot_colormap(fig, axs[0], 
    unstacked.sub(median, axis=0), 
    times,
    cmap=cm.get_cmap('coolwarm'),
    vlims=(-2e-15,2e-15),
    logfreq=True,
    neutral=0.0,
    cbar_label='Absolute difference from reference PSD'
)
axs[1].title.set_text('|PSD(t) - PSD_median| / PSD_median')
plot_colormap(fig, axs[1], 
    abs(unstacked.sub(median, axis=0)).div(median, axis=0),
    times,
    cmap='PuRd',
    vlims=(0,0.5),
    logfreq=True
)
plt.show()

# Frequency slice
fig, axs = plt.subplots(2,2)
fig.suptitle('Channel ' + cols[channel]
    + ' - PSDs at selected frequencies')
plot_freq_slice(fig, axs[0,0], 0.01, times, summaries, 'b', ylim=(0e-15,3.5e-15))
plot_freq_slice(fig, axs[0,1], 0.10, times, summaries, 'b', ylim=(1e-15, 4.5e-15))
plot_freq_slice(fig, axs[1,0], 0.32, times, summaries, 'b', ylim=(0,2e-14))
plot_freq_slice(fig, axs[1,1], 0.50, times, summaries, 'b', ylim=(0,2e-14))
#plt.show()

# Time slice
fig, axs = plt.subplots(1,1)
fig.suptitle('Channel ' + cols[channel] + ' - PSDs at selected times since '
    + str(time.get_iso_date(times[0])) + ' UTC')
plot_time_slice(fig, axs, 0.32, times, summaries, 'b', logpsd=True)
plot_time_slice(fig, axs, 0.50, times, summaries, 'g')
plot_time_slice(fig, axs, 1.50, times, summaries, 'orange')
plot_time_slice(fig, axs, 2.50, times, summaries, 'r')
axs.title.set_text('')
#plt.show()
