import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
import pandas as pd
import time_functions as tf
import psd
    
def shifted_cmap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
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

def plot_colormap(fig, ax, psd, cmap, vlims, cbar_label=None, center=None):
    '''
    Function to plot the colormap of a PSD with frequency on the y-axis and
    time on the x-axis.
    
    Input
    -----
      fig, ax : The figure and axes of the plot
      psd : The PSD, an unstacked DataFrame
      cmap : The unaltered color map to use
      vlims : A tuple of the color scale limits
      cbar_label : Color bar label
      center : The center value of a diverging colormap
    '''
    # Get start date in UTC
    start_date = tf.get_iso_date(int(psd.columns[0]))
    # Change columns from GPS time to days elapsed from start of run
    psd.columns = pd.Series(tf.get_days_elapsed(psd.columns), name='TIME')
    if not cbar_label: cbar_label = 'Fractional difference from reference PSD'
    if center:
        cmap = shifted_cmap(
            cmap, 
            midpoint=(center-vlims[0])/(vlims[1]-vlims[0]), 
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
            psd.columns[0],
            psd.columns[-1],
            0., 1.
        ]
    )
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-3, top=1.)
    ax.set_xlabel('Days elapsed since ' + str(start_date) + ' UTC')
    ax.set_ylabel('Frequency (Hz)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, labelpad=15, rotation=270)
    
def plot_freq_slice(fig, ax, freq, summary, color='b', ylim=None):
    # Plots time vs PSD at a specific frequency
    # Parameters:
    #  fig, ax: the figure and axes of the plot
    #  freq: the frequency along which to slice
    #  summaries: 3D array, shape (time, frequency, stats)
    #   with stats arranged | frequency | median | median - CI | median + CI |
    # Get the index of the nearest frequency to the one requested
    days_elapsed = tf.get_days_elapsed(summary.index.get_level_values(0))
    fslice, freq = psd.get_freq_slice(summary, freq)
    freq_str = str(np.round(freq, decimals=4))
    ax.fill_between(days_elapsed,
        fslice['CI_50_LO'],
        fslice['CI_50_HI'], 
        color=color,
        alpha=0.5,
        label='50% credible interval')
    ax.fill_between(days_elapsed,
        fslice['CI_90_LO'],
        fslice['CI_90_HI'],
        color=color,
        alpha=0.2,
        label='90% credible interval')
    ax.plot(days_elapsed, summary['MEDIAN'], label='Median PSD', color=color)
    ax.legend()
    ax.set_xlabel('Days elapsed since ' + str(tf.get_iso_date(gps_times[0])) + ' UTC')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel('PSD')
    ax.title.set_text('PSD at ' + freq_str + ' Hz')
    
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
    days_elapsed = tf.get_days_elapsed(gps_times)
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
    ax.title.set_text('PSD at ' + str(tf.get_iso_date(gps_times[time_index])) + ' UTC')

