#!/usr/bin/env python3

import os
from glob import glob
import sys
import argparse

import numpy as np
import pandas as pd
from pymc3.stats import hpd

import linechain as lc
import import_psd
import plot
import utils




def fft(run, channel, frequencies, log=None):
    '''
    Returns the discrete Fourier transform of power at specific frequencies
    over time. First interpolates the data to get consistent dt.
    '''
    log.log('FFT analysis')
    # Select median column for specific run, channel
    df = run.psd_summary.loc[channel,'MEDIAN']
    # Remove NaN values
    df = df[df.notna()]
    times = df.index.unique(level='TIME')
    # Find time differences between each observation
    diffs = np.array([times[i] - times[i-1] for i in range(1, len(times))])
    # Find the mean time difference, excluding outliers
    dt = np.mean(diffs[diffs < 1640])
    log.log(f'dt = {dt}')
    log.log(f'1/(2*dt) = {1. / (2 * dt)}')
    
    # List of times at same time cadence
    n = int((times[-1] - times[0]) / dt)
    new_times = np.array([times[0] + i * dt for i in range(n)])
    
    rfftfreq = []
    rfft = []
    for f in frequencies:
        median = df.xs(f, level='FREQ')
        new_values = np.interp(new_times, times, median)
        rfftfreq.append(np.fft.rfftfreq(n, dt))
        rfft.append(np.absolute(np.fft.rfft(new_values)))
    
    if len(frequencies) == 1:
        rfftfreq = rfftfreq[0]
        rfft = rfft[0]
    
    return rfftfreq, rfft
    
def fft_peaks(rfftfreq, rfft):
    # Analysis parameters
    bin_width = 10 # number of points on either side of peak to bin
    f_step = 5e-6 # amount to step between each check
    min_sig = 3 # minimum significance for peak identification
    
    # Number of peak bins to check
    n_bins = int((rfftfreq[-bin_width] - rfftfreq[bin_width]) / f_step)
    # Cycle through frequency space looking for peaks
    f_peaks = []
    peaks = []
    sigs = []
    background = []
    for i in range(n_bins):
        # Define minimum and maximum peak frequencies
        f_min = rfftfreq[bin_width] + i * f_step
        f_max = f_min + f_step
        
        # Find background mean and variance
        b1 = rfft[rfftfreq < f_min]
        bin_width = min(bin_width, b1.shape[0]) # make sure start doesn't go below 0
        b1 = b1[-bin_width:]
        b2 = rfft[rfftfreq > f_max][:bin_width]
        b_mean = np.mean(np.concatenate([b1, b2]))
        b_var = np.mean([np.var(b1), np.var(b2)])
        b_std = np.sqrt(b_var)
        
        # Find peak value
        peak_psd = rfft[(rfftfreq > f_min) & (rfftfreq < f_max)]
        peak_range = rfftfreq[(rfftfreq > f_min) & (rfftfreq < f_max)]
        peak_val = np.max(peak_psd)
        f_peak = peak_range[np.argmax(peak_psd)]
        significance = (peak_val - b_mean) / b_std
        if significance >= min_sig:
            f_peaks.append(f_peak)
            peaks.append(peak_val)
            sigs.append(significance)
            background.append(b_mean)

    # Significance table
    f_peaks = np.array(f_peaks)
    peak_df = pd.DataFrame({'FREQ': f_peaks, 'PERIOD': 1/f_peaks, 
            'AMPLIUDE': peaks, 'SIG': sigs, 'BACKGROUND': background})
    
    return peak_df

def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Generate PSD summaries and plots.'
    )
    parser.add_argument('runs', type=str, nargs='*', 
        help='run directory name (default: all folders in "data/" directory)'
    )
    parser.add_argument('-c', '--compare', dest='compare', action='store_true',
            help='compare summary plots for different runs side by side')
    parser.add_argument('--overwrite-all', dest='overwrite', action='store_true',
        help='re-generate summary files even if they already exist (default: \
              ask for each run)'
    )
    parser.add_argument('--keep-all', dest='keep', action='store_true',
        help='do not generate summary file if it already exists (default: ask \
              for each run)'
    )
    args = parser.parse_args()
    # Add all runs in data directory if none are specified
    if len(args.runs) == 0: 
        args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    
    # Import impacts file, if any
    impacts_file = 'impacts.dat'
    impacts = np.array([])
    if os.path.exists(impacts_file):
        impacts = get_impacts(impacts_file)
    
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
        # Log output file
        log_file = os.path.join(run.summary_dir, 'psd.log')
        log = utils.Log(log_file, f'psd.py log file for {run.name}')
        # Confirm to overwrite if summary already exists
        if args.keep: overwrite = False
        elif args.overwrite: overwrite = True
        elif os.path.exists(run.psd_file):
            over = input('Found psd.pkl for this run. Overwrite? (y/N) ')
            overwrite = True if over == 'y' else False
        else: overwrite = True

        # Import / generate summary PSD DataFrame
        if overwrite:
            run.psd_summary = import_psd.summarize(run)
        else:
            run.psd_summary = pd.read_pickle(run.psd_file)
        
        # Make plots
        df = run.psd_summary
        # Frequency slices: roughly logarithmic, low-frequency
        plot_frequencies = np.array([1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2])
        plot_frequencies = get_exact_freq(run.psd_summary, plot_frequencies)
        
        if not args.compare:
            p = utils.Progress(run.channels, 'Plotting...')
            for i, channel in enumerate(run.channels):
                # FFT analysis
                fft_file = os.path.join(run.plot_dir, f'fft{i}.png')
                rfftfreq, rfft = fft(run, channel, plot_frequencies, log)
                plot.fft(rfftfreq, rfft, run, channel, plot_frequencies, 
                        logfreq=False, plot_file=fft_file)
                # Colormap
                cmap_file = os.path.join(run.plot_dir, f'colormap{i}.png')
                plot.save_colormaps(run, channel, cmap_file)
                # Frequency slices
                fslice_file = os.path.join(run.plot_dir, f'fslice{i}.png')
                plot.save_freq_slices([run], channel, plot_frequencies, 
                        impacts=impacts, plot_file=fslice_file)
                # Time slices
                tslice_file = os.path.join(run.plot_dir, f'tslice{i}.png')
                plot.save_time_slices(run, channel, slice_times, tslice_file)
                # Update progress
                p.update(i)
        
    # Plot run comparisons
    if args.compare:
        p = utils.Progress(runs[0].channels, '\nPlotting run comparisons...')
        multirun_dir = os.path.join('out', 'multirun')
        if not os.path.exists(multirun_dir): os.makedirs(multirun_dir)
        for i, channel in enumerate(runs[0].channels):
            plot.compare_colormaps(runs, channel, 
                    plot_file=os.path.join(multirun_dir, f'colormap{i}.png'))
            plot.save_freq_slices(runs, channel, plot_frequencies, 
                    impacts=impacts, 
                    plot_file=os.path.join(multirun_dir, f'fslice{i}.png'))
            p.update(i)
    
    print('Done!')

if __name__ == '__main__':
    main()

