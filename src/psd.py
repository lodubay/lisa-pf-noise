#! /bin/bash

import os
import glob
import sys

import numpy as np
import pandas as pd
from pymc3.stats import hpd

import time_functions as tf
import linechain as lc
import plot

def import_time(time_dir):
    '''
    Import and combine all psd.dat files in a single time directory 
    for many channels. Assumes file name format 'psd.dat.#' and 'psd.dat.##'.
    Returns a DataFrame with frequency increasing down the rows and 
    chain index increasing across the columns. The DataFrame is MultiIndexed,
    with indices (highest level to lowest) channel, time, frequency.

    Input
    -----
      time_dir : relative path to the time directory
    '''
    time = int(time_dir[-11:-1])
    # Channel names
    channels = list(range(6))
    # Column names
    cols = ['FREQ'] + channels
    # Sort so that (for example) psd.dat.2 is sorted before psd.dat.19
    psd_files = sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9]'))) + \
        sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9][0-9]')))
    # Import PSD files into DataFrame
    time_data = []
    for pf in psd_files:
        psd = pd.read_csv(
            pf, sep=' ', usecols=range(len(cols)), names=cols, index_col=0
        )
        # Concatenate columns vertically
        psd = pd.concat([psd[channel] for channel in channels])
        time_data.append(psd)
    # Concatenate psd series horizontally
    time_data = pd.concat(time_data, axis=1, ignore_index=True)
    # Define MultiIndex
    # Round frequency index to 5 decimals to deal with floating point issues
    time_data.index = pd.MultiIndex.from_product(
        [channels, [time],
            np.around(time_data.index.get_level_values('FREQ').unique(), 6)], 
        names=['CHANNEL', 'TIME', 'FREQ']
    )
    # Strip rows of 2s
    return time_data[time_data.iloc[:,0] < 2]

def summarize_psd(time_dir):
    '''
    Returns a DataFrame with the median and credible intervals for one time.
    Credible intervals are calculated using
    pymc3's highest posterior density (HPD) function, where alpha is the 
    desired probability of type I error (so, 1 - C.I.).
    Uses the same MultiIndex as import_time().
    
    Input
    -----
      time_dir : relative path to the time directory
    '''
    # Import time data
    time_data = import_time(time_dir)
    # Grab MultiIndex
    midx = time_data.index
    # Calculate HPDs
    time_data_np = time_data.to_numpy().T
    hpd_50 = hpd(time_data_np, alpha=0.5)
    hpd_90 = hpd(time_data_np, alpha=0.1)
    # Return summary DataFrame
    return pd.DataFrame({
        'MEDIAN'    : time_data.median(axis=1),
        'CI_50_LO'  : pd.Series(hpd_50[:,0], index=midx),
        'CI_50_HI'  : pd.Series(hpd_50[:,1], index=midx),
        'CI_90_LO'  : pd.Series(hpd_90[:,0], index=midx),
        'CI_90_HI'  : pd.Series(hpd_90[:,1], index=midx),
    }, index=midx)

def save_summary(run, summary_file):
    '''
    Returns a multi-index DataFrame of PSD summaries across multiple times 
    from one run folder. The first index represents channel, the second GPS time
    and the third frequency.
    
    Input
    -----
      run : string, name of the run directory
    '''
    # Get list of time directories within run directory
    time_dirs = tf.get_time_dirs(run)
    # Pull PSD files from target run
    sys.stdout.write('Importing ' + run + ' psd...   0%\b')
    sys.stdout.flush()
    steps = len(time_dirs)
    # Concatenate DataFrames of all times; takes a while
    summaries = []
    for i, d in enumerate(time_dirs):
        summaries.append(summarize_psd(d))
        # Update progress indicator
        progress = str(int((i+1) / steps * 100))
        sys.stdout.write('\b' * len(progress) + progress)
        sys.stdout.flush()
    # Finish progress indicator
    sys.stdout.write('\n')
    sys.stdout.flush()

    summaries = pd.concat(summaries)
    # List of GPS times from index
    gps_times = summaries.index.get_level_values('TIME').unique()
    # Median time step
    dt = int(np.median(
        [gps_times[i+1] - gps_times[i] for i in range(len(gps_times) - 1)]
    ))

    # Check for time gaps and fill with NaN DataFrames
    for i in range(len(gps_times) - 1):
        diff = gps_times[i+1] - gps_times[i]
        if diff > dt + 1:
            # Number of new times to insert
            n = int(np.floor(diff / dt))
            print('Filling ' + str(n) + ' missing times...')
            # List of missing times, with same time interval
            missing_times = [gps_times[i] + dt * k for k in range(1, n + 1)]
            # Create new MultiIndex for empty DataFrame
            channels = summaries.index.get_level_values('CHANNEL').unique()
            frequencies = summaries.index.get_level_values('FREQ').unique()
            midx = pd.MultiIndex.from_product(
                [channels, missing_times, frequencies],
                names=['CHANNEL', 'TIME', 'FREQ']
            )
            # Create empty DataFrame, append to summaries, and sort
            filler = pd.DataFrame(columns=summaries.columns, index=midx)
            summaries = summaries.append(filler).sort_index(level=[0, 1, 2])
    # Output to file
    print('Writing to ' + summary_file + '...')
    summaries.to_pickle(summary_file)
    return summaries

def get_exact_freq(summary, approx_freq):
    '''
    Takes an approximate input frequency and returns the closest measured
    frequency in the data.
    '''
    gps_times = list(summary.index.get_level_values('TIME').unique())
    freqs = list(summary.xs(gps_times[0]).index)
    freq_index = round(approx_freq / (max(freqs) - min(freqs)) * len(freqs))
    return freqs[freq_index]

def main():
    runs = sys.argv[1:]
    for run in runs:
        print('\n-- ' + run + ' --')
        # Directories
        output_dir = os.path.join('out', run)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_dir = os.path.join('out', run, 'plots')
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        # Output files
        summary_file = os.path.join(output_dir, 'psd.pkl')
        model_file = os.path.join(output_dir, 'line_evidence.dat')
        # Confirm to overwrite if summary already exists
        gen_new = True
        if os.path.exists(summary_file):
            over = input('Found summary.pkl for this run. Overwrite? (y/N) ')
            gen_new = True if over == 'y' else False

        # Import / generate summary PSD DataFrame
        if gen_new:
            print('Generating summary file...')
            df = save_summary(run, summary_file)
        else:
            print('Reading summary file...')
            df = pd.read_pickle(summary_file)

        # Make plots
        for channel in range(6):
            print('Plotting channel ' + str(channel) + '...')
            # Colormap
            cmap_file = os.path.join(plot_dir, 'colormap' + str(channel) + '.png')
            plot.save_colormaps(run, channel, df, cmap_file, show=False)

            # Frequency slices
            fslice_file = os.path.join(plot_dir, 'fslice' + str(channel) + '.png')
            plot.save_freq_slices(run, channel, df, fslice_file, show=False)
            
            # Time slices - representative sample
            # Time plot file name
            tslice_file = os.path.join(plot_dir, 'tslice' + str(channel) + '.png')
            gps_times = tf.get_gps_times(run)
            
            # Generate / import DataFrame of all times with spectral lines
            if os.path.exists(model_file):
                print('Line evidence file found. Reading...')
                line_df = pd.read_csv(model_file, sep=' ', index_col=0)
            else:
                print('No line evidence file found. Generating...')
                line_df = lc.gen_model_df(run, model_file)
            # Return list of times
            line_times = df[df.iloc[:,channel] > 0].index
            
            # Choose 6 times: beginning, end, and 4 evenly drawn from list
            l = len(gps_times)
            indices = [int(i / 5 * l) for i in range(1,5)]
            slice_times = sorted([gps_times[0], gps_times[-1]] +
                [gps_times[i] for i in indices]
            )
            # Plot
            plot.save_time_slices(run, channel, df, slice_times, tslice_file,
                time_format='gps', exact=True, show=False, logpsd=True
            )
            
            # Time slices - all spectral lines
            if len(line_times) > 0:
                tslice_file = os.path.join(plot_dir, 'tslice_lines'+str(channel)+'.png')
                plot.save_time_slices(run, channel, df, line_times, tslice_file,
                    time_format='gps', exact=True, show=False, logpsd=True
                )
    print('Done!')

if __name__ == '__main__':
    main()

