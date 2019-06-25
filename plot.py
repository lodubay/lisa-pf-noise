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

def colormap(fig, ax, psd, cmap, vlims, cbar_label=None, center=None):
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

def all_psds(fig, ax, time_dir, channel, xlim=None, ylim=None):
    '''
    Plots all PSD samples in a single time directory for one channel
    '''
    df = psd.import_time(time_dir).loc[channel]
    summary = psd.summarize_psd(time_dir).loc[channel]
    freqs = df.index.get_level_values('FREQ')
    for i in range(100):
        ax.scatter(freqs, df[i], marker='.', color='b')
    if xlim: ax.set_xlim(xlim)
    else: ax.set_xlim(auto=True)
    if ylim: ax.set_ylim(ylim)
    else: ax.set_ylim(auto=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
def freq_slice(fig, ax, freq, summary, color='b', ylim=None):
    '''
    Plots time vs PSD at a specific frequency.

    Input
    -----
      fig, ax : the figure and axes of the plot
      freq : the approximate frequency along which to slice
      summary : the summary DataFrame
      color : the color of the plot, optional
      ylim : tuple of y axis bounds, optional
    '''
    # Get the index of the nearest frequency to the one requested
    freq = psd.get_exact_freq(summary, freq)
    # Slice along that frequency
    fslice = psd.get_freq_slice(summary, freq)
    # Date stuff
    days_elapsed = tf.get_days_elapsed(fslice.index)
    start_date = tf.get_iso_date(int(fslice.index[0]))
    # Plot 50% credible interval
    ax.fill_between(days_elapsed,
        fslice['CI_50_LO'],
        fslice['CI_50_HI'], 
        color=color,
        alpha=0.5,
        label='50% credible interval')
    # Plot 90% credible interval
    ax.fill_between(days_elapsed,
        fslice['CI_90_LO'],
        fslice['CI_90_HI'],
        color=color,
        alpha=0.2,
        label='90% credible interval')
    # Plot median
    ax.plot(days_elapsed, fslice['MEDIAN'], label='Median PSD', color=color)
    # Vertical scale
    if ylim: 
        ax.set_ylim(ylim)
    else:
        med = fslice['MEDIAN'].median()
        hi = fslice['CI_90_HI'].median() - med
        lo = med - fslice['CI_90_LO'].median()
        ax.set_ylim((max(med - 6 * lo, 0), med + 6 * hi))
    # Axis labels
    ax.set_xlabel('Days elapsed since ' + str(start_date) + ' UTC')
    ax.set_ylabel('PSD')
    ax.title.set_text(str(np.around(freq*1000, 3)) + ' mHz')

def time_slice(fig, ax, time, summary, color='b', ylim=None, logpsd=False):
    '''
    Plots frequency vs PSD at a specific time.

    Input
    -----
      fig, ax : the figure and axes of the plot
      time : the approximate time (run begins at 0) along which to slice
      summary : the summary DataFrame
      color : the color of the plot, optional
      ylim : tuple of y axis bounds, optional
      lobpsd : if true, plots psd on a log scale
    '''
    # Get the index of the nearest time to the one requested
    gps_time, day = tf.get_exact_time(summary, time)
    day = str(np.round(day, 4))
    # Get time slice
    tslice = psd.get_time_slice(summary, gps_time)
    # Plot 50% credible interval
    ax.fill_between(tslice.index, 
        tslice['CI_50_LO'], 
        tslice['CI_50_HI'],
        color=color, 
        alpha=0.5,
        label='50% credible interval at T+' + str(day) + ' days')
    # Plot 90% credible interval
    ax.fill_between(tslice.index, 
        tslice['CI_90_LO'], 
        tslice['CI_90_HI'],
        color=color, 
        alpha=0.1,
        label='90% credible interval at T+' + str(day) + ' days')
    # Plot median
    ax.plot(tslice.index, tslice['MEDIAN'], 
        label='Median PSD at T+' + str(day) + ' days', 
        color=color)
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xscale('log')
    if ylim: ax.set_ylim(ylim)
    if logpsd: ax.set_yscale('log')
    ax.set_ylabel('PSD')
    ax.title.set_text('PSD at ' + str(tf.get_iso_date(gps_time)) + ' UTC')

