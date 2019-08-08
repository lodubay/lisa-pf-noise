#!/usr/bin/env python3

import os
from glob import glob
import sys
import argparse
import configparser

import numpy as np
import pandas as pd
from pymc3.stats import hpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.ticker as tkr

import import_psd
import utils

def main():
    '''
    Initializes the argument parser and sets up the log file for the analysis
    scripts. Then calls functions to analyze individual runs and compare
    multiple runs.
    '''
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Generate PSD summaries and plots.'
    )
    parser.add_argument('runs', type=str, nargs='*', 
            help='run directory name (default: all folders in data directory)'
    )
    parser.add_argument('-c', '--compare', dest='compare', action='store_true',
            help='generate additional side-by-side run comparison plots')
    parser.add_argument('--overwrite-all', dest='overwrite', action='store_true',
            help='re-generate summary files even if they already exist '
                    '(default: ask for each run)'
    )
    parser.add_argument('--keep-all', dest='keep', action='store_true',
            help='do not generate summary file if it already exists ' + \
                    '(default: ask for each run)'
    )
    parser.add_argument('--no-title', dest='title', action='store_false',
            help='do not include a plot title')
    args = parser.parse_args()
    # Add all runs in data directory if none are specified
    if len(args.runs) == 0: 
        args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')
    
    # Plot config parser
    config = configparser.ConfigParser(
            interpolation=configparser.BasicInterpolation())
    config.read('config.ini')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    dfs = []
    
    # Import impacts file, if any
    impacts_file = 'impacts.dat'
    impacts = np.array([])
    if os.path.exists(impacts_file):
        print('Importing impacts file...')
        impacts = utils.get_impacts(impacts_file)
    
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
        
        # Confirm to overwrite if summary already exists
        if args.keep: 
            overwrite = False
        elif args.overwrite: 
            overwrite = True
        elif os.path.exists(run.psd_file):
            over = input('Found psd.pkl for this run. Overwrite? (y/N) ')
            overwrite = True if over == 'y' else False
        else: 
            overwrite = True
        
        # Import / generate summary PSD DataFrame
        if overwrite:
            df = import_psd.summarize(run)
        else:
            df = pd.read_pickle(run.psd_file)
        dfs.append(df)
        '''
        plot_fslices(run, df, np.array([1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2]),
                config)
                '''

    if args.compare:
        compare_fslices(runs, dfs, np.array([1e-3, 5e-3, 3e-2]), config)

def fslice(fig, ax, df, channel, frequency):
    '''
    Plots power at a specific frequency over time.
    
    Input
    -----
      fig, ax : the figure and axes of the plot
      df : DataFrame, the full summary output from import_psd.py
      channel : str, the channel to investigate
      frequency : float, the approximate frequency along which to slice the PSD
    '''
    # Get exact frequency along which to slice
    frequency = utils.get_exact_freq(df, frequency)
    freq_text = f'%s mHz' % float('%.3g' % (frequency * 1000.))
    
    # Isolate given channel
    time_series = df.loc[channel].xs(frequency, level='FREQ')
    days_elapsed = utils.gps2day(time_series.index) # Convert to days elapsed
    
    # Plot 90% credible interval
    ax.fill_between(days_elapsed, time_series['CI_90_LO'], 
            time_series['CI_90_HI'],
            color='#b3cde3', label='90% credible interval')
    # Plot 50% credible interval
    ax.fill_between(days_elapsed, time_series['CI_50_LO'], 
            time_series['CI_50_HI'], 
            color='#8c96c6', label='50% credible interval')
    # Plot median
    ax.plot(days_elapsed, time_series['MEDIAN'], 
            label='Median', color='#88419d')
    
    # Axis title
    ax.set_title(freq_text)
            
    # Vertical axis limits
    med = time_series['MEDIAN'].median()
    std = time_series['MEDIAN'].std()
    hi = min(2 * (time_series['CI_90_HI'].quantile(0.95) - med), 
             max(time_series['CI_90_HI']) - med)
    lo = min(2 * abs(med - time_series['CI_90_LO'].quantile(0.05)), 
             abs(med - min(time_series['CI_90_LO'])))
    ylim = (med - lo, med + hi)
    ax.set_ylim(ylim)
    
    # Horizontal axis limits
    ax.set_xlim((min(days_elapsed), max(days_elapsed)))

def plot_fslices(run, df, frequencies, config):
    # Create plot for each channel
    p = utils.Progress(run.channels, 'Plotting frequency slices...')
    for c, channel in enumerate(run.channels):
        # Plot grid of time series
        fig = utils.gridplot(fslice, df, channel, frequencies,
                f'{run.mode.upper()} channel {channel}\n' + \
                        'Power at selected frequencies',
                f'Days elapsed since\n{run.start_date} UTC', 'Power', 
                config)
        
        # Save plot
        plot_file = os.path.join(run.plot_dir, f'fslice{c}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        
        # Update progress
        p.update(c)

def compare_fslices(runs, dfs, frequencies, config):
    # Create plot for each channel
    p = utils.Progress(runs[0].channels, 'Plotting frequency slices...')
    for c, channel in enumerate(runs[0].channels):
        # Plot grid of time series
        fig = utils.compareplot(fslice, runs, dfs, channel, frequencies,
                f'Channel {channel}\nPower at selected frequencies over time',
                [f'Days elapsed since\n{run.start_date} UTC' for run in runs], 
                'Power', config)
        
        # Save plot
        plot_file = os.path.join(config.get('Directories', 'multirun_dir'), 
                'plots', f'fslice{c}.png')
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

