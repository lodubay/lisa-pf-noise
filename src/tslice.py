#!/usr/bin/env python3

import os
import argparse
from glob import glob
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import utils

def main():
    # Argument parser
    args = utils.add_parser('Frequency slice analysis.')

    # Plot config parser
    config = configparser.ConfigParser()
    config.read('src/plotconfig.ini')
    
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
        p = utils.Progress(run.channels, 'Plotting time slices...')
        for c, channel in enumerate(run.channels):
            # Plot grid of PSDs
            fig = utils.gridplot(tslice, df, channel, times,
                    f'{run.mode.upper()} channel {channel}\n' + \
                            'PSDs at selected times',
                    'Frequency (Hz)', 'PSD', config)
            
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
    ax.set_title(f't={time}')
    ax.set_xscale('log')
    ax.set_yscale('log')

if __name__ == '__main__':
    main()

