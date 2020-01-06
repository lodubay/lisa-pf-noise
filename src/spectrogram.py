#!/usr/bin/env python3

import os
import argparse
from glob import glob
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.ticker as tkr

import utils

def main():
    # Argument parser
    args = utils.add_parser('Spectrogram analysis.', title=True)

    # Configuration parser
    config = configparser.ConfigParser()
    config.read('src/plotconfig.ini')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    
    # Individual spectrograms
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
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
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            if args.title:
                fig.suptitle(
                    f'Spectrogram of {run.mode.upper()} channel {channel}\n' + \
                            'compared to median PSD')
            
            # Subplots
            axs[0].set_title('Absolute difference')
            axs[0].set_ylabel('Frequency (Hz)')
            spectrogram(fig, axs[0], run,
                unstacked.sub(median, axis=0), 
                cmap=cm.get_cmap('coolwarm'), 
            )
            axs[1].set_title('Relative difference')
            spectrogram(fig, axs[1], run,
                abs(unstacked.sub(median, axis=0)).div(median, axis=0),
                cmap='PuRd',
                vlims=(0,1)
            )
            
            # Figure layout
            if args.title: hlim = 0.85
            else: hlim = 1
            fig.tight_layout(rect=[0, 0, 1, hlim], w_pad=2)
            
            # Save plot
            plot_file = os.path.join(run.plot_dir, f'spectrogram{i}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            
            # Update progress
            p.update(i)
    
    # Comparison spectrograms
    if args.compare:
        # Output directory
        multirun_dir = os.path.join('out', 'multirun')
        if not os.path.exists(multirun_dir): 
            os.makedirs(multirun_dir)
        plot_dir = os.path.join(multirun_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        run_str = [f'{run.mode} {run.name}' for run in runs]
        print(f'\n-- {", ".join(run_str)} --')
        
        # Plot spectrogram for each channel
        p = utils.Progress(runs[0].channels, 
                'Plotting comparison spectrograms...')
        for i, channel in enumerate(runs[0].channels):
            # Setup figure
            fig = plt.figure(figsize=(
                    len(runs) * config.getfloat('Placement', 'fig_x_scale'), 
                    config.getfloat('Placement', 'fig_y_scale')))
            fig.suptitle(f'Channel {channel}\n' + \
                    'Power compared to median at observed times, frequencies',
                    y=0.99)
            
            for r, run in enumerate(runs):
                # Import PSD summaries
                df = pd.read_pickle(run.psd_file)
                
                # Set up subplot
                ax = fig.add_subplot(1, len(runs), r+1)
                
                # Unstack psd, removing all columns except the median
                psd = df.loc[channel,'MEDIAN']
                unstacked = psd.unstack(level='TIME')
                # Find median across all times
                median = unstacked.median(axis=1)
                
                # Spectrogram
                im = spectrogram(fig, ax, run,
                    unstacked.sub(median, axis=0).div(median, axis=0), 
                    cmap=cm.get_cmap('coolwarm'), vlims=(-1,1), bar=False
                )
                
                # Subplot config
                ax.set_title(f'{run.mode.upper()}')
                if r==0: # only label y axis on left most plot
                    ax.set_ylabel('Frequency (Hz)')
            
            # Set tight layout
            fig.tight_layout(rect=[0, 0, 1, 0.9])
            
            # Make colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.66])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Relative difference from median PSD', labelpad=25, 
                    rotation=270)
            
            # Save plot
            plot_file = os.path.join(plot_dir, f'spectrogram{i}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            
            # Update progress
            p.update(i)

def spectrogram(fig, ax, run, psd, cmap, 
        vlims=None, cbar_label=None, bar=True):
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
    ax.set_xlabel(f'Days elapsed since\n{run.start_date} UTC')
    ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
    # Tick label size
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')
    # Add and label colorbar
    if bar:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, labelpad=15, rotation=270)
        offset = ax.yaxis.get_offset_text()
        offset.set_size(16)
    
    return im

if __name__ == '__main__':
    main()

