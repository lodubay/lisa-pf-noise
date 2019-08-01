#!/usr/bin/env python3

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

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

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Frequency slice analysis.')
    parser.add_argument('runs', type=str, nargs='*', 
            help='run directory name (default: all folders in "data/" directory)'
    )
    parser.add_argument('-c', '--compare', dest='compare', action='store_true',
            help='generate additional side-by-side run comparison plots')
    args = parser.parse_args()
    # Add all runs in data directory if none are specified
    if len(args.runs) == 0: 
        args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
        # Import PSD summaries
        df = pd.read_pickle(run.psd_file)
        
        # Even spread of observed times
        n = 6
        indices = [int(i / (n-1) * len(run.gps_times)) for i in range(1,n-1)]
        times = sorted([run.gps_times[0], run.gps_times[-1]] +
            [run.gps_times[i] for i in indices]
        )
        
        # Create plot for each channel
        p = utils.Progress(run.channels, 'Plotting frequency slices...')
        for c, channel in enumerate(run.channels):
            # Automatically create grid of axes
            nrows = int(np.floor(float(len(times)) ** 0.5))
            ncols = int(np.ceil(1. * len(times) / nrows))
            # Set up figure
            fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
            fig.suptitle(f'PSD at selected times for {run.mode.upper()} ' + \
                    f'{run.name} channel {channel}')
            
            # Subplots
            for i, time in enumerate(times):
                # Set up subplot
                ax = fig.add_subplot(nrows, ncols, i+1)
                
                # Time slice
                tslice(fig, ax, df, channel, time)
                
                # Subplot config
                ax.set_title(f'PSD at GPS time {time}')
                if i % ncols == 0:
                    ax.set_ylabel('PSD')
                if i >= len(times) - ncols:
                    ax.set_xlabel('Frequency (Hz)')
            
            # Legend
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 2, 1] # reorder legend
            fig.legend([handles[i] for i in order], [labels[i] for i in order])
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            
            # Save plot
            plot_file = os.path.join(run.plot_dir, f'tslice{c}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            
            # Update progress
            p.update(c)

def tslice(fig, ax, df, channel, time):
    '''
    Plots the median power spectral density with credible intervals at
    particular times.

    Input
    -----
      fig, ax : the figure and axes of the plot
      df : DataFrame, the full summary output from import_psd.py
      run : Run object
      channel : str, the channel to investigate
      time : float, the frequency along which to slice the PSD
    '''
    # Isolate given channel
    psd = df.loc[channel].xs(time, level='TIME')
    
    # Plot 90% credible interval
    ax.fill_between(psd.index, psd['CI_90_LO'], psd['CI_90_HI'],
        color='#a1dab4', label='90% credible interval')
    # Plot 50% credible interval
    ax.fill_between(psd.index, psd['CI_50_LO'], psd['CI_50_HI'],
        color='#41b6c4', label='50% credible interval')
    # Plot median
    ax.plot(psd.index, psd['MEDIAN'], label='Median', color='#225ea8')
    
    # Plot config
    ax.set_xscale('log')
    ax.set_yscale('log')

if __name__ == '__main__':
    main()

