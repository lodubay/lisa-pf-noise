import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.ticker as tkr

import psd
import utils

# Font parameters
fig_title_size = 28
subplot_title_size = 24
legend_label_size = 20
ax_label_size = 20
tick_label_size = 18
offset_size = 16

# Tick parameters
major_tick_length = 10
minor_tick_length = 5

# Other parameters
subplot_title_pad = 15    

def save_time_slices(run, channel, times, plot_file=None, show=False,
        time_format='gps', exact=True, logpsd=True):

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

def compare_linecounts(runs, channel, plot_file=None, show=False):
    # Setup figure
    fig = plt.figure(figsize=(8 + len(runs) * 4, 9))
    fig.suptitle(f'Channel {channel}\n' + \
            'No. spectral lines over time',
            y=0.99, fontsize=fig_title_size)
    
    for i, run in enumerate(runs):
        # Setup subplot
        ax = fig.add_subplot(1, len(runs), i+1)
        
        # Get line counts
        counts = run.linecounts.loc[channel]
        # Change GPS times to days elapsed
        counts.index = pd.Series(run.gps2day(counts.index), name='TIME')
        # Convert counts to fraction of total
        counts = counts / sum(counts.iloc[0].dropna())
        # Cut off DataFrame after n=5
        counts = counts.iloc[:,:6]
        
        # Subplots
        ax.set_title(f'{run.mode.upper()}', size=subplot_title_size)
        im = ax.pcolormesh(
            list(counts.index) + [counts.index[-1] + run.dt / (60*60*24)],
            list(counts.columns) + [int(counts.columns[-1]) + 1],
            counts.to_numpy().T,
            cmap='PuRd',
            vmax=0.5
        )
        
        # Axis labels
        ax.set_xlabel(f'Days elapsed since {run.start_date} UTC', 
                fontsize=ax_label_size)
        ax.tick_params(labelsize=tick_label_size, length=major_tick_length)
        ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
        ax.tick_params(axis='x', which='minor', length=minor_tick_length)
        # Only label y axis on left most plot
        if i==0:
            ax.set_ylabel('Modeled no. spectral lines', fontsize=ax_label_size)
        # Put the major ticks at the middle of each cell
        ax.set_yticks(counts.columns + 0.5, minor=False)
        ax.set_yticklabels(counts.columns)
    
    # Set tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Make colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.66])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=tick_label_size)
    cbar.set_label('Relative frequency of line model', labelpad=25, 
            rotation=270, fontsize=ax_label_size)
    
    if plot_file: plt.savefig(plot_file, bbox_inches='tight')
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

