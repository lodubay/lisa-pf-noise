#!/usr/bin/env python3

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
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
        print(f'\n-- {run.mode} {run.name} --')
        # Set up log file
        log = utils.Log(os.path.join(run.log_dir, 'fft.log'), 
                f'fft.log file for {run.mode} {run.name}')
        
        # Import PSD summaries
        df = pd.read_pickle(run.psd_file)
        
        # Frequencies to examine
        frequencies = np.array([1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2])
        frequencies = utils.get_exact_freq(df, frequencies)
        
        # FFT plot for each channel
        print('Plotting...')
        for c, channel in enumerate(run.channels):
            log.log(f'\nChannel {channel}')
            # Automatically create grid of axes
            nrows = int(np.floor(float(len(frequencies)) ** 0.5))
            ncols = int(np.ceil(1. * len(frequencies) / nrows))
            # Set up figure
            fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
            fig.suptitle(f'{run.mode.upper()} channel {channel}', fontsize=28)
            
            # Plot FFT at each frequency
            for i, freq in enumerate(frequencies):
                freq_str = f'%s mHz' % float('%.3g' % (freq * 1000.))
                log.log(f'\n{freq_str}')
                ax = fig.add_subplot(nrows, ncols, i+1)
                
                rfftfreq, rfft = fft(df, channel, freq, log)
                psd = np.absolute(rfft[i])**2
                ax.plot(rfftfreq[i], psd, color='#0077c8')
                
                # Axis configuration
                ax.set_title(f'FFT of power at {freq_str}', fontsize=22)
                ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
                ax.set_yscale('log')
                if i % ncols == 0:
                    ax.set_ylabel('PSD', fontsize=20)
                if i >= len(frequencies) - ncols:
                    ax.set_xlabel('Frequency (Hz)', fontsize=20)
            
            plot_file = os.path.join(run.plot_dir, f'fft{c}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()

def fft(psd_summary, channel, frequency, log):
    '''
    Returns the discrete Fourier transform of power at a specific frequency
    over time. First interpolates the data to get consistent dt.
    '''
    # Select median column for specific run, channel
    median = psd_summary.loc[channel,'MEDIAN']
    median = median[median.notna()] # remove NaN values
    times = median.index.unique(level='TIME')
    
    # Find the mean time difference, excluding outliers
    diffs = np.diff(times)
    dt = np.mean(diffs[diffs < 1640])
    
    # List of times at same time cadence
    n = int((times[-1] - times[0]) / dt)
    new_times = times[0] + np.arange(0, n * dt, dt)
    
    # Discrete Fourier transform
    median = median.xs(frequency, level='FREQ')
    new_values = np.interp(new_times, times, median)
    rfftfreq = np.fft.rfftfreq(n, dt)
    rfft = np.absolute(np.fft.rfft(new_values))
    
    # Sanity checks
    log.log(f'dt = {dt}')
    log.log(f'1/df = {np.mean(np.diff(rfftfreq))}')
    log.log(f'f_max = {np.max(rfftfreq)}')
    log.log(f'1/(2*dt) = {1. / (2 * dt)}')
    
    return rfftfreq, rfft

if __name__ == '__main__':
    main()

