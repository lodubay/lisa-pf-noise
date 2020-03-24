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
    
    # Matplotlib RC Config 
    print('Using config file at ' + str(matplotlib.matplotlib_fname()))
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    dfs = []
    
    # Import impacts file, if any
    impacts_file = os.path.join('data', 'impacts.dat')
    impacts = np.array([])
    if os.path.exists(impacts_file):
        print('Importing impacts file...')
        impacts = utils.get_impacts(impacts_file)

    # Import array of frequencies, with default if no file specified
    frequencies = np.array([1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2])
    frequencies_file = os.path.join('data', 'frequencies.csv')
    if os.path.exists(frequencies_file):
        frequencies = np.loadtxt(frequencies_file, delimiter=',')
    
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

        # Generate plots
        '''
        plot_tslices(run, df)
        plot_fslices(run, df, frequencies)
        plot_spectrograms(run, df)
        plot_ffts(run, df, frequencies)
        '''

    if args.compare:
        run_str = [f'{run.mode} {run.name}' for run in runs]
        print(f'\n-- {", ".join(run_str)} --')
        #compare_fslices(runs, dfs, frequencies)
        #compare_spectrograms(runs, dfs)
        compare_ffts(runs, dfs, frequencies)

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

def plot_fslices(run, df, frequencies):
    # Create plot for each channel
    p = utils.Progress(run.channels, 'Plotting frequency slices...')
    for c, channel in enumerate(run.channels):
        # Plot grid of time series
        fig = utils.gridplot(fslice, df, channel, frequencies,
                f'{run.mode.upper()} channel {channel}\n' + \
                        'Power at selected frequencies',
                f'Days elapsed since\n{run.start_date} UTC', 'Power')
        
        # Save plot
        plot_file = os.path.join(run.plot_dir, f'fslice{c}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        
        # Update progress
        p.update(c)

def compare_fslices(runs, dfs, frequencies):
    # Create plot for each channel
    p = utils.Progress(runs[0].channels, 'Plotting comparison frequency slices...')
    for c, channel in enumerate(runs[0].channels):
        # Plot grid of time series
        fig = utils.compareplot(fslice, runs, dfs, channel, frequencies,
                f'Channel {channel}\nPower at selected frequencies over time',
                [f'Days elapsed since\n{run.start_date} UTC' for run in runs], 
                'Power')
        
        # Save plot
        plot_file = os.path.join('out', 'multirun', 'plots', f'fslice{c}.png')
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

def plot_tslices(run, df, times=[]):
    # If no times given, sample even selection of observed times
    if len(times) == 0:
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
                'Frequency (Hz)', 'PSD')
        
        # Save plot
        plot_file = os.path.join(run.plot_dir, f'tslice{c}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        
        # Update progress
        p.update(c)

def spectrogram(fig, ax, run, psd, cmap, 
        vlims=None, cbar_label=None, bar=True):
    '''
    Plot the spectrogram of a PSD with frequency on the y-axis and
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
    # Add and label colorbar
    if bar:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, labelpad=15, rotation=270)
        offset = ax.yaxis.get_offset_text()
        #offset.set_size(config.getfloat('Font', 'offset_size'))
    
    return im

def plot_spectrograms(run, df):
    '''
    Plot spectrograms for each channel. Produces separate plots for the absolute
    and relative differences from the mean PSD.
    '''
    p = utils.Progress(run.channels, 'Plotting spectrograms...')
    for i, channel in enumerate(run.channels):
        # Unstack psd, removing all columns except the median
        psd = df.loc[channel,'MEDIAN']
        unstacked = psd.unstack(level='TIME')
        # Find median across all times
        median = unstacked.median(axis=1)
        
        # Absolute difference spectrograms
        fig, ax = plt.subplots(1, 1)
        ax.set_title(
            f'Absolute difference of {run.mode.upper()} channel {channel}\n' + \
                    'compared to median PSD')
        ax.set_ylabel('Frequency (Hz)')
        spectrogram(fig, ax, run,
            unstacked.sub(median, axis=0), 
            cm.get_cmap('coolwarm')
        )
        plot_file = os.path.join(run.plot_dir, f'spectrogram_abs{i}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()

        # Relative difference spectrograms
        fig, ax = plt.subplots(1, 1)
        ax.set_title(
            f'Relative difference of {run.mode.upper()} channel {channel}\n' + \
                    'compared to median PSD')
        ax.set_ylabel('Frequency (Hz)')
        spectrogram(fig, ax, run,
            abs(unstacked.sub(median, axis=0)).div(median, axis=0),
            'PuRd', vlims=(0,1)
        )
        plot_file = os.path.join(run.plot_dir, f'spectrogram_rel{i}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        
        # Update progress
        p.update(i)

def compare_spectrograms(runs, dfs):
    # Plot spectrogram for each channel
    p = utils.Progress(runs[0].channels, 'Plotting comparison spectrograms...')
    for i, channel in enumerate(runs[0].channels):

        # Relative difference spectrograms

        # Setup figure
        fig_scale = 6
        fig = plt.figure(figsize=(len(runs) * fig_scale, fig_scale))
        fig.suptitle(f'Channel {channel}\n' + \
                'Power compared to median PSD at observed times, frequencies',
                y=0.99)
        
        for r, run in enumerate(runs):
            # Import PSD summaries
            df = dfs[r]
            
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
                cm.get_cmap('coolwarm'), vlims=(-1,1), bar=False
            )
            
            # Subplot config
            ax.set_title(f'{run.mode.upper()}')
            if r==0: # only label y axis on left most plot
                ax.set_ylabel('Frequency (Hz)')
        
        # Set tight layout
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        
        # Make colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.66])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Relative difference from median PSD', labelpad=25, 
                rotation=270)
        
        # Save plot
        plot_file = os.path.join('out', 'multirun', 'plots', 
                f'spectrogram_rel{i}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        
        # Update progress
        p.update(i)

def fft(fig, ax, df, channel, frequency):
    '''
    Plots a single Fourier transform of power at a specific frequency
    over time. First interpolates the data to get consistent dt.
    
    Input
    -----
      fig, ax : the figure and axes of the plot
      df : DataFrame, output from import_psd.py
      channel : string, channel of interest
      frequency : float, exact frequency along which to slice the DataFrame
    '''
    # Get exact frequency
    frequency = utils.get_exact_freq(df, frequency)
    freq_text = f'%s mHz' % float('%.3g' % (frequency * 1000.))

    # Select median column for specific run, channel
    median = df.loc[channel,'MEDIAN']
    median = median[median.notna()] # remove NaN values
    times = median.index.unique(level='TIME')
    
    # Find the mean time difference, excluding outliers
    diffs = np.diff(times)
    dt = np.mean(diffs[diffs < 1.5 * np.median(diffs)])
    
    # List of times at same time cadence
    n = int((times[-1] - times[0]) / dt)
    new_times = times[0] + np.arange(0, n * dt, dt)
    
    # Interpolate to remove data gaps
    median = median.xs(frequency, level='FREQ')
    new_values = np.interp(new_times, times, median)

    # Discrete Fourier transform, removing data point for f=0
    rfftfreq = np.fft.rfftfreq(n, dt)[1:]
    rfft = np.absolute(np.fft.rfft(new_values, n))[1:]

    # Plot PSD
    psd = np.absolute(rfft)**2
    ax.plot(rfftfreq, psd, color='#0077c8')
                
    # Axis configuration
    ax.set_title(freq_text)
    ax.set_yscale('log')

def plot_ffts(run, df, frequencies):
    # Remove large gap in LTP run
    if run.name == 'run_b' and run.mode == 'ltp':
        df = df[df.index.get_level_values('TIME') >= 1143962325]

    # Create plot for each channel
    p = utils.Progress(run.channels, 'Plotting FFTs...')
    for c, channel in enumerate(run.channels):
        # Plot grid of time series
        fig = utils.gridplot(fft, df, channel, frequencies,
                f'{run.mode.upper()} channel {channel}\n' + \
                        'FFT of power at selected frequencies',
                f'Days elapsed since\n{run.start_date} UTC', 'PSD')
        
        # Save plot
        plot_file = os.path.join(run.plot_dir, f'fft{c}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        
        # Update progress
        p.update(c)

def compare_ffts(runs, dfs, frequencies):
    # Remove large gap in LTP run
    for run in runs:
        if run.name == 'run_b' and run.mode == 'ltp':
            dfs[run] = dfs[run][df.index.get_level_values('TIME') >= 1143962325]

    # Create plot for each channel
    p = utils.Progress(runs[0].channels, 'Plotting comparison FFTs...')
    for c, channel in enumerate(runs[0].channels):
        # Plot grid of time series
        fig = utils.compareplot(fft, runs, dfs, channel, frequencies,
                f'Channel {channel}\nFFT of power at selected frequencies',
                [f'Days elapsed since\n{run.start_date} UTC' for run in runs], 
                'PSD')
        
        # Save plot
        plot_file = os.path.join('out', 'multirun', 
                'plots', f'fft{c}.png')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        
        # Update progress
        p.update(c)

if __name__ == '__main__':
    main()

