import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.ticker as tkr

import psd
import utils
    
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

def colormap(fig, ax, run, psd, cmap, vlims=None, cbar_label=None, center=None,
        bar=True):
    '''
    Function to plot the colormap of a PSD with frequency on the y-axis and
    time on the x-axis.
    
    Input
    -----
      fig, ax : The figure and axes of the plot
      psd : The PSD, an unstacked DataFrame
      run : Run object
      cmap : The unaltered color map to use
      vlims : A tuple of the color scale limits
      cbar_label : Color bar label
      center : The center value of a diverging colormap
    '''
    # Change columns from GPS time to days elapsed from start of run
    psd.columns = pd.Series(run.gps2day(psd.columns), name='TIME')
    # Median frequency step
    df = np.median(np.diff(psd.index))
    # Auto colormap scale
    if not vlims:
        med = psd.median(axis=1).median()
        std = psd.std(axis=1).median()
        vlims = (med - 2 * std, med + 2 * std)
    # Shift colormap to place 0 in the center if needed
    if center:
        cmap = shifted_cmap(
            cmap, 
            midpoint=(center-vlims[0])/(vlims[1]-vlims[0]), 
            name='shifted colormap'
        )
    im = ax.pcolormesh(
        list(psd.columns) + [psd.columns[-1] + run.dt / (60*60*24)],
        list(psd.index) + [psd.index[-1] + df],
        psd,
        cmap=cmap,
        vmin=vlims[0],
        vmax=vlims[1]
    )
    # Vertical scale
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-3, top=1.)
    # Axis labels
    ax.set_xlabel(f'Days elapsed since {run.start_date} UTC', fontsize='large')
    # Tick label size
    ax.tick_params(axis='both', which='major', labelsize='large')
    # Add and label colorbar
    if bar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize='large')
        if cbar_label:
            cbar.set_label(cbar_label, labelpad=15, rotation=270)
    
    return im

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

def time_slice(fig, ax, time, summary, ylim=None, logpsd=False):
    '''
    Plots frequency vs PSD at a specific time.

    Input
    -----
      fig, ax : the figure and axes of the plot
      time : the exact gps time along which to slice
      summary : the summary DataFrame
      color : the color of the plot, optional
      ylim : tuple of y axis bounds, optional
      lobpsd : if true, plots psd on a log scale
    '''
    # Get time slice
    tslice = summary.xs(time)
    # Plot 90% credible interval
    ax.fill_between(tslice.index, 
        tslice['CI_90_LO'], 
        tslice['CI_90_HI'],
        color='#a1dab4', 
        #alpha=0.1,
        label='90% credible interval')
    # Plot 50% credible interval
    ax.fill_between(tslice.index, 
        tslice['CI_50_LO'], 
        tslice['CI_50_HI'],
        color='#41b6c4', 
        #alpha=0.5,
        label='50% credible interval')
    # Plot median
    ax.plot(tslice.index, tslice['MEDIAN'], label='Median PSD', color='#225ea8')
    ax.set_xscale('log')
    if ylim: ax.set_ylim(ylim)
    if logpsd: ax.set_yscale('log')
    ax.title.set_text(str(time))

def save_colormaps(run, channel, plot_file, show=False):
    df = run.psd_summary.loc[channel,'MEDIAN']
    # Unstack psd, removing all columns except the median
    unstacked = df.unstack(level='TIME')
    
    # Find median across all times
    median = unstacked.median(axis=1)
    # Set up figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Colormap of {run.mode.upper()} {run.name} channel {channel} PSD over time',
        fontsize='x-large'
    )
    # Subplots
    axs[0].title.set_text('Absolute difference from median PSD')
    axs[0].set_ylabel('Frequency (Hz)', fontsize='large')
    colormap(fig, axs[0], run,
        unstacked.sub(median, axis=0), 
        cmap=cm.get_cmap('coolwarm'),
        center=0.0
    )
    axs[1].title.set_text('Fractional difference from median PSD')
    colormap(fig, axs[1], run,
        abs(unstacked.sub(median, axis=0)).div(median, axis=0),
        cmap='PuRd',
        vlims=(0,1)
    )
    plt.savefig(plot_file, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

def compare_colormaps(runs, channel, plot_file=None, show=False):
    # Setup figure
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f'Channel {channel}\n' + \
            'Power compared to median at observed times, frequencies',
            y=0.99, fontsize='xx-large')
    
    for i, run in enumerate(runs):
        # Setup subplot
        ax = fig.add_subplot(1, len(runs), i+1)
        
        # Unstack psd, removing all columns except the median
        df = run.psd_summary.loc[channel,'MEDIAN']
        unstacked = df.unstack(level='TIME')
        # Find median across all times
        median = unstacked.median(axis=1)
        
        # Subplots
        ax.set_title(f'{run.mode.upper()} ({run.name})', size='x-large')
        im = colormap(fig, ax, run,
            unstacked.sub(median, axis=0).div(median, axis=0), 
            cmap=cm.get_cmap('coolwarm'), vlims=(-1,1),
            center=0.0, bar=False
        )
        # Only label y axis on left most plot
        if i==0:
            ax.set_ylabel('Frequency (Hz)', fontsize='large')
    
    # Set tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Make colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize='large')
    cbar.set_label('Relative difference from median PSD', labelpad=15, 
            rotation=270, fontsize='large')
    
    if plot_file: plt.savefig(plot_file, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

def save_freq_slices(runs, channel, frequencies, impacts=[], 
        plot_file=None, show=False):
    '''
    Plots frequency slices, with frequency increasing vertically. Also compares
    multiple runs side by side if more than one is provided.
    
    Input
    -----
      runs : list of utils.Run objects
      channel : string, channel name
      frequencies : 1D Numpy array, approximate frequencies to slice along
      impacts : DataFrame of micrometeoroid impacts, if any
      plot_file : string, path to plot output file, if any
      show : whether to display figure, defaults to no
    '''
    # Tweakables
    figsize = (20, 7.5) # Relative figure size
    plot_height = 4 # Relative height of each subplot
    hspace = 3 # Relative vertical spaceing between subplots
    wspace = 0.1
    impactplot_height = 1 # Relative height of the impacts subplot
    spine_pad = 10 # Spine offset from subplots
    scaled_offset = 0.0025 * spine_pad # Scaled spine_pad
    axlabelpad = 25 # Padding between axis label and spine
    # Font sizes
    ticklabelsize = 'large'
    offsetsize = 'medium'
    axlabelsize = 'x-large'
    titlesize = 'xx-large'
    legendsize = 'large'
    # Labels and titles
    plot_title = f'Channel {channel}\nPower at selected frequencies over time' 
    ylabel = 'Power at selected frequency'
    
    # Set up figure, grid
    fig = plt.figure(figsize=figsize)
    grid_height = plot_height * len(frequencies)
    grid = plt.GridSpec(grid_height, len(runs), hspace=hspace, wspace=wspace)
    fig.suptitle(plot_title, fontsize=titlesize)
    fig.tight_layout()
    
    for j, run in enumerate(runs):
        # Isolate given channel
        df = run.psd_summary.loc[channel]
        # Get nearest actual frequencies to those requested
        frequencies = psd.get_exact_freq(df, frequencies)
        # Plot highest frequency on top
        frequencies = np.flip(np.sort(frequencies))
        
        # Subplots
        for i, freq in enumerate(frequencies):
            # Add new subplot
            ax = fig.add_subplot(grid[plot_height*i:plot_height*i+plot_height, j])
            # Subplot title if top plot
            if i == 0:
                ax.set_title(f'{run.mode.upper()} ({run.name})')
                # Also grab top axis legend info
                ax1 = ax
            
            # Set up DataFrames
            fslice = df.xs(freq, level='FREQ') # Frequency slice
            #exp = int(np.floor(np.log10(fslice['CI_90_LO'].median()))) # Get exponent
            #fslice = fslice / (10 ** exp) # Scale
            days_elapsed = run.gps2day(fslice.index) # Convert to days elapsed
            
            # Plot 90% credible interval
            ax.fill_between(days_elapsed, fslice['CI_90_LO'], fslice['CI_90_HI'],
                    color='#b3cde3', label='90% credible interval')
            # Plot 50% credible interval
            ax.fill_between(days_elapsed, fslice['CI_50_LO'], fslice['CI_50_HI'], 
                    color='#8c96c6', label='50% credible interval')
            # Plot median
            ax.plot(days_elapsed, fslice['MEDIAN'], 
                    label='Median PSD', color='#88419d')
            
            # Smart-ish axis limits
            med = fslice['MEDIAN'].median()
            std = fslice['MEDIAN'].std()
            hi = min(2 * (fslice['CI_90_HI'].quantile(0.95) - med), 
                     max(fslice['CI_90_HI']) - med)
            lo = min(2 * abs(med - fslice['CI_90_LO'].quantile(0.05)), 
                     abs(med - min(fslice['CI_90_LO'])))
            ylim = (med - lo, med + hi)
            # Set vertical axis limits
            ax.set_ylim(ylim)
            ax.spines['left'].set_bounds(ylim[0], ylim[1])
            # Horizontal axis limits
            ax.set_xlim((min(days_elapsed), max(days_elapsed)))
            ax.spines['bottom'].set_bounds(min(days_elapsed), max(days_elapsed))
            
            # Frequency labels on right-most plots
            if j == len(runs) - 1:
                ax.text(1.01 * days_elapsed[-1], med,
                        f'%s mHz' % float('%.3g' % (freq * 1000.)), 
                        fontsize=legendsize, va='center')
            
            # Format left vertical axis
            #ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.1f'))
            #ax.yaxis.set_major_formatter(tkr.ScalarFormatter())
            y_formatter = tkr.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.yaxis.set_minor_locator(tkr.AutoMinorLocator())
            ax.spines['left'].set_position(('outward', spine_pad))
            ax.tick_params(axis='y', which='major', labelsize=ticklabelsize)
            # More mathy exponent label
            ax.ticklabel_format(axis='y', useMathText=True)
            exp_txt = ax.yaxis.get_offset_text()
            exp_txt.set_x(-0.005 * spine_pad)
            exp_txt.set_size(offsetsize)
            
            # Format bottom horizontal axis
            if i+1 < len(frequencies):# or len(impact_days) > 0:
                # Remove bottom axis if not the bottom plot
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(bottom=False)
                # Horizontal axis ticks
                ax.xaxis.set_major_locator(tkr.NullLocator())
                ax.xaxis.set_minor_locator(tkr.NullLocator())
            else:
                # Horizontal axis for bottom plot
                ax.set_xlabel(f'Days elapsed since {run.start_date} UTC', 
                        fontsize=axlabelsize)
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_position(('outward', spine_pad))
                ax.tick_params(bottom=True)
                ax.tick_params(axis='x', which='major', labelsize=ticklabelsize)
                # Minor ticks
                ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
    
                # Find micrometeoroid impacts, if any
                if len(impacts) > 0:
                    # Find impact times
                    gps_times = df.index.unique(level='TIME')
                    days = run.gps2day(gps_times)
                    impact_times = impacts[
                            (impacts['GPS'] >= gps_times[0]) \
                          & (impacts['GPS'] <= gps_times[-1])]
                    impact_days = run.gps2day(impact_times['GPS']).to_numpy()
                
                # Plot micrometeoroid impacts, if any
                impact_plt = ax.scatter(impact_days, 
                        [ylim[0] - (ylim[1]-ylim[0]) * 0.018 * spine_pad] * len(impact_days), 
                        c='red', marker='x', label='Impact event', clip_on=False)
            
            # Remove spines and ticks for other axes
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    
    # Add big subplot for common y axis label
    ax = fig.add_subplot(1, 1, 1, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, 
            right=False, which='both')
    ax.grid(False)
    # y axis label
    ax.set_ylabel(ylabel, fontsize=axlabelsize, ha='center', va='center', 
            labelpad=axlabelpad)
    
    # Make legend
    handles, labels = ax1.get_legend_handles_labels()
    handles += [impact_plt]
    order = [0, 2, 1, 3] # Reorder legend
    fig.legend(handles=[handles[i] for i in order], fontsize=legendsize, 
            loc='upper right', bbox_to_anchor=(0.95, 1), 
            bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(top=0.85)
    
    # Save / display figure
    if plot_file: plt.savefig(plot_file, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

def save_time_slices(run, channel, times, plot_file=None, show=False,
        time_format='gps', exact=True, logpsd=True):
    # Convert given times to gps if necessary
    if time_format == 'day': times = run.day2gps(times)
    # Find exact times if necessary
    if not exact: times = run.get_exact_gps(times)
    # Automatically create grid of axes
    nrows = int(np.floor(float(len(times)) ** 0.5))
    ncols = int(np.ceil(1. * len(times) / nrows))
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    fig.suptitle(f'Selected times for {run.mode.upper()} {run.name} channel {channel}')
    df = run.psd_summary.loc[channel]
    # Subplots
    for i, time in enumerate(times):
        ax = fig.add_subplot(nrows, ncols, i+1)
        time_slice(fig, ax, times[i], df, logpsd=logpsd)
        # Axis title
        ax.title.set_text(f'PSD at GPS time {time}')
        # Vertical axis label on first plot in each row
        if i % ncols == 0:
            ax.set_ylabel('PSD')
        # Horizontal axis label on bottom plot in each column
        if i >= len(times) - ncols:
            ax.set_xlabel('Frequency (Hz)')
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if plot_file: plt.savefig(plot_file)
    if show: plt.show()
    else: plt.close()

def linechain_scatter(run, channel, param, plot_file=None, show=False):
    df = run.lc_summary.loc[channel, :, :, param]
    # 90% error bars
    plt.errorbar(run.gps2day(df.index.get_level_values('TIME')), 
        df['MEDIAN'], 
        yerr=([df['MEDIAN'] - df['CI_90_LO'], df['CI_90_HI'] - df['MEDIAN']]), 
        ls='', marker='', capsize=3, alpha=0.2, ecolor='b'
    )
    # Median and 50% error bars
    plt.errorbar(run.gps2day(df.index.get_level_values('TIME')), 
        df['MEDIAN'], 
        yerr=([df['MEDIAN'] - df['CI_50_LO'], df['CI_50_HI'] - df['MEDIAN']]), 
        ls='', marker='.', capsize=5, ecolor='b'
    )
    plt.xlabel(f'Days elapsed since {run.start_date} UTC')
    plt.ylabel(param)
    plt.yscale('log')
    plt.title(f'{run.mode.upper()} {run.name} channel {channel} spectral line {param} over time')
    if plot_file:
        plt.savefig(plot_file)
    if show: plt.show()
    else: plt.close()

def linecounts_cmap(run, channel, plot_file=None, show=False):
    ''' Plots a colormap of the spectral line counts over time '''
    counts = run.linecounts.loc[channel]
    # Change GPS times to days elapsed
    counts.index = pd.Series(run.gps2day(counts.index), name='TIME')
    # Convert counts to fraction of total
    counts = counts / sum(counts.iloc[0].dropna())
    # Plot
    fig, ax = plt.subplots(1, 1)
    ax.title.set_text(
        f'Line model frequency over time for {run.mode.upper()} {run.name} channel {channel}'
    )
    im = ax.pcolormesh(
        list(counts.index) + [counts.index[-1] + run.dt / (60*60*24)],
        list(counts.columns) + [int(counts.columns[-1]) + 1],
        counts.to_numpy().T,
        cmap='PuRd',
        vmax=0.5
    )
    # Axis labels
    ax.set_xlabel(f'Days elapsed since {run.start_date} UTC')
    ax.set_ylabel('Modeled no. spectral lines')
    # Put the major ticks at the middle of each cell
    ax.set_yticks(counts.columns + 0.5, minor=False)
    ax.set_yticklabels(counts.columns)
    # Add and label colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Line model relative frequency', labelpad=15, rotation=270)
    # Save figure and close
    if plot_file: plt.savefig(plot_file)
    if show: plt.show()
    else: plt.close()

'''
DANGER - UNDER CONSTRUCTION
'''

def linecounts_combined(runs, channel, plot_file=None, show=False):
    ''' Plots a colormap of the spectral line counts over time for many runs '''
    # Sort runs by start date
    runs.sort(key=lambda run: run.start_date)
    # Fill missing times, if the gap between runs isn't too big
    for i in range(len(runs) - 1):
        diff = runs[i+1].gps_times[0] - runs[i].gps_times[-1]
        # If the gap is less than 3 days
        if diff < 3 * (24*60*60):
            all_times = runs[i].gps_times + runs[i+1].gps_times
            missing_times = runs[i].get_missing_times(all_times)
            missing_midx = pd.MultiIndex.from_product(
                    [run[i].channels, missing_times], names=['CHANNEL', 'TIME']
            )
            missing_df = pd.DataFrame(columns=run[i].linecounts.columns,
                    index=missing_midx
            )
            runs[i].linecounts = pd.concat(
                    [runs[i].linecounts, missing_df]
            ).sort_index(level=[0,1])
    counts = pd.concat([run.linecounts for run in runs]).loc[channel]
    print(counts[counts.isna()])
    # Change GPS times to days elapsed
    counts.index = pd.Series(runs[0].gps2day(counts.index), name='TIME')
    # Convert counts to fraction of total
    counts = counts / sum(counts.iloc[0].dropna())
    # Plot
    fig, ax = plt.subplots(1, 1)
    ax.title.set_text(
        f'Line model frequency over time for all runs, channel {channel}'
    )
    im = ax.pcolormesh(
        list(counts.index) + [counts.index[-1] + runs[0].dt / (60*60*24)],
        list(counts.columns) + [int(counts.columns[-1]) + 1],
        counts.to_numpy().T,
        cmap='PuRd',
        vmax=0.5
    )
    # Axis labels
    ax.set_xlabel(f'Days elapsed since {runs[0].start_date} UTC')
    ax.set_ylabel('Modeled no. spectral lines')
    # Put the major ticks at the middle of each cell
    ax.set_yticks(counts.columns + 0.5, minor=False)
    ax.set_yticklabels(counts.columns)
    # Add and label colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Line model relative frequency', labelpad=15, rotation=270)
    # Save figure and close
    if plot_file: plt.savefig(plot_file)
    if show: plt.show()
    else: plt.close()

