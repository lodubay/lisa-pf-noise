#!/bin/bash

import os
import glob
import sys

import numpy as np
import pandas as pd
from pymc3.stats import hpd

import linechain as lc
import plot
import utils

def import_time(run, time_dir):
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
    # Column names
    cols = ['FREQ'] + list(run.channels)
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
        psd = pd.concat([psd[channel] for channel in run.channels])
        time_data.append(psd)
    # Concatenate psd series horizontally
    time_data = pd.concat(time_data, axis=1, ignore_index=True)
    # Define MultiIndex
    # Round frequency index to 5 decimals to deal with floating point issues
    time_data.index = pd.MultiIndex.from_product(
        [run.channels, [time],
            np.around(time_data.index.unique(level='FREQ'), 6)], 
        names=['CHANNEL', 'TIME', 'FREQ']
    )
    # Strip rows of 2s
    return time_data[time_data.iloc[:,0] < 2]

def summarize_psd(run, time_dir):
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
    time_data = import_time(run, time_dir)
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
      run : Run object
    '''
    # Set up progress indicator
    p = utils.Progress(run.time_dirs, f'Importing {run.name} psd files...')
    # Concatenate DataFrames of all times; takes a while
    summaries = []
    for i, d in enumerate(run.time_dirs):
        summaries.append(summarize_psd(run, d))
        # Update progress indicator
        p.update(i)

    summaries = pd.concat(summaries)

    # Check for time gaps and fill with NaN DataFrames
    times = run.gps_times[:-1]
    p = utils.Progress(times, 'Checking for time gaps...')
    N = 0
    for i, gps_time in enumerate(times):
        diff = run.gps_times[i+1] - run.gps_times[i]
        if diff > run.dt + 1:
            # Number of new times to insert
            n = int(np.floor(diff / run.dt))
            N += n
            # List of missing times, with same time interval
            missing_times = [times[i] + run.dt * k for k in range(1, n + 1)]
            # Create new MultiIndex for empty DataFrame
            channels = summaries.index.unique(level='CHANNEL')
            frequencies = summaries.index.unique(level='FREQ')
            midx = pd.MultiIndex.from_product(
                [channels, missing_times, frequencies],
                names=['CHANNEL', 'TIME', 'FREQ']
            )
            # Create empty DataFrame, append to summaries, and sort
            filler = pd.DataFrame(columns=summaries.columns, index=midx)
            summaries = summaries.append(filler).sort_index(level=[0, 1, 2])
        # Update progress indicator
        p.update(i)
    print(f'Filled {N} missing times.')
    
    # Output to file
    print(f'Writing to {summary_file}...')
    summaries.to_pickle(summary_file)
    return summaries

def get_exact_freq(summary, approx_freq):
    '''
    Takes an approximate input frequency and returns the closest measured
    frequency in the data.
    '''
    gps_times = list(summary.index.unique(level='TIME'))
    freqs = list(summary.xs(gps_times[0]).index)
    freq_index = round(approx_freq / (max(freqs) - min(freqs)) * len(freqs))
    return freqs[freq_index]

def main():
    # Get all runs to use from the command line.
    runs = sys.argv[1:]
    # Defaults to all available runs
    if len(runs) == 0:
        runs = os.listdir('data')
    for run in runs:
        run = utils.Run(run)
        print(f'\n-- {run.name} --')
        # Directories
        output_dir = os.path.join('out', run.name, 'summaries')
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_dir = os.path.join('out', run.name, 'psd_plots')
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        # Output files
        summary_file = os.path.join(output_dir, 'psd.pkl')
        model_file = os.path.join(output_dir, 'linecounts.dat')
        # Confirm to overwrite if summary already exists
        gen_new = True
        if os.path.exists(summary_file):
            over = input('Found summary.pkl for this run. Overwrite? (y/N) ')
            gen_new = True if over == 'y' else False

        # Import / generate summary PSD DataFrame
        if gen_new:
            df = save_summary(run, summary_file)
        else:
            df = pd.read_pickle(summary_file)

        # Make plots
        for channel in run.channels:
            print('Plotting channel ' + str(channel) + '...')
            # Colormap
            cmap_file = os.path.join(plot_dir, f'colormap{channel}.png')
            plot.save_colormaps(run, channel, df, cmap_file, show=False)

            # Frequency slices
            fslice_file = os.path.join(plot_dir, f'fslice{channel}.png')
            plot.save_freq_slices(run, channel, df, fslice_file, show=False)
            
            # Time slices - representative sample
            # Time plot file name
            tslice_file = os.path.join(plot_dir, f'tslice{channel}.png')
            
            # Generate / import DataFrame of all times with spectral lines
            if os.path.exists(model_file):
                print('Line evidence file found. Reading...')
                line_df = pd.read_csv(model_file, sep=' ', index_col=0)
            else:
                print('No line evidence file found. Generating...')
                line_df = lc.gen_model_df(run, model_file)
            # Return list of times
            line_times = line_df[line_df.iloc[:,channel] > 0].index
            
            # Choose 6 times: beginning, end, and 4 evenly drawn from list
            l = len(run.gps_times)
            indices = [int(i / 5 * l) for i in range(1,5)]
            slice_times = sorted([run.gps_times[0], run.gps_times[-1]] +
                [run.gps_times[i] for i in indices]
            )
            # Plot
            plot.save_time_slices(run, channel, df, slice_times, tslice_file,
                time_format='gps', exact=True, show=False, logpsd=True
            )
            
            # Time slices - all spectral lines
            if len(line_times) > 0:
                tslice_file = os.path.join(plot_dir, f'tslice_lines{channel}.png')
                plot.save_time_slices(run, channel, df, line_times, tslice_file,
                    time_format='gps', exact=True, show=False, logpsd=True
                )
    print('Done!')

if __name__ == '__main__':
    main()

