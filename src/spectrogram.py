#!/usr/bin/env python3

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.ticker as tkr

import utils

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='FFT analysis.')
    parser.add_argument('runs', type=str, nargs='*', 
        help='run directory name (default: all folders in "data/" directory)'
    )
    args = parser.parse_args()
    # Add all runs in data directory if none are specified
    if len(args.runs) == 0: 
        args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    
    for run in runs:
        # Import PSD summaries
        df = pd.read_pickle(run.psd_file)
        
        # Plot spectrograms for each channel
        p = utils.Progress(run.channels, 'Plotting spectrograms...')
        for i, channel in enumerate(run.channels):
            # Unstack psd, removing all columns except the median
            psd = df.loc[channel,'MEDIAN']
            unstacked = psd.unstack(level='TIME')
            # Find median across all times
            median = unstacked.median(axis=1)
            
            # Set up figure
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f'Spectrogram of {run.mode.upper()} channel {channel}',
                fontsize=28
            )
            
            # Subplots
            axs[0].set_title('Absolute difference from median PSD', 
                    fontsize=24, pad=15)
            axs[0].set_ylabel('Frequency (Hz)', fontsize=20)
            spectrogram(fig, axs[0], run,
                unstacked.sub(median, axis=0), 
                cmap=cm.get_cmap('coolwarm'),
            )
            axs[1].set_title('Fractional difference from median PSD',
                    fontsize=24, pad=15)
            spectrogram(fig, axs[1], run,
                abs(unstacked.sub(median, axis=0)).div(median, axis=0),
                cmap='PuRd',
                vlims=(0,1)
            )
            
            # Figure layout
            fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=-1)
            
            # Save plot
            plot_file = os.path.join(run.plot_dir, f'spectrogram{i}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            
            # Update progress
            p.update(i)

def spectrogram(fig, ax, run, psd, cmap, vlims=None, cbar_label=None, bar=True):
    '''
    Function to plot the colormap of a PSD with frequency on the y-axis and
    time on the x-axis.
    
    Input
    -----
      fig, ax : The figure and axes of the plot
      run : Run object
      psd : The PSD, an unstacked DataFrame
      cmap : The unaltered color map to use
      vlims : A tuple of the color scale limits
      cbar_label : Color bar label
      bar : bool, whether or not to include a colorbar in the plot
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
    # The spectrogram
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
    ax.set_xlabel(f'Days elapsed since\n{run.start_date} UTC', 
            fontsize=20)
    ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
    # Tick label size
    ax.tick_params(axis='both', which='major', labelsize=18,
            length=10)
    ax.tick_params(axis='both', which='minor', length=5)
    # Add and label colorbar
    if bar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=18)
        if cbar_label:
            cbar.set_label(cbar_label, labelpad=15, rotation=270)
        offset = ax.yaxis.get_offset_text()
        offset.set_size(16)
    
    return im

if __name__ == '__main__':
    main()

