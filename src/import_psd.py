#!/usr/bin/env python3

import os
from glob import glob
import argparse

import numpy as np
import pandas as pd
from pymc3.stats import hpd

import utils

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Generate PSD summaries.')
    parser.add_argument('runs', type=str, nargs='*', 
        help='run directory name (default: all folders in "data/" directory)'
    )
    args = parser.parse_args()
    # Add all runs in data directory if none are specified
    if len(args.runs) == 0: 
        args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    
    # Summarize all runs
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
        summarize(run)

def summarize(run):
    '''
    For all times in a run, import and combine all psd.dat files. Assumes 
    file name format 'psd.dat.#' and 'psd.dat.##'. Each time is summarized in
    a DataFrame with the median and credible intervals for that time.
    Credible intervals are calculated using pymc3's highest posterior density 
    (HPD) function.
    
    Returns a multi-index DataFrame of PSD summaries across multiple times 
    from one run. The DataFrame is MultiIndexed, with indices (in order) 
    channel, time, and frequency. Inserts blank rows in place of time gaps.
    
    Input
    -----
      run : Run object
    '''
    
    # Set up progress indicator
    p = utils.Progress(run.time_dirs, f'Importing {run.name} psd files...')
    
    # Concatenate DataFrames of all times; takes a while
    summaries = []
    for i, time_dir in enumerate(run.time_dirs):
        time = run.get_time(time_dir) # GPS time int
        # Sort so that (for example) psd.dat.2 is sorted before psd.dat.19
        psd_files = sorted(glob(os.path.join(time_dir, 'psd.dat.[0-9]'))) + \
                sorted(glob(os.path.join(time_dir, 'psd.dat.[0-9][0-9]')))
        
        # Import PSD files into DataFrame
        time_data = []
        for f in psd_files:
            # Import data file
            psd = pd.read_csv(f, sep=' ', usecols=range(7), 
                    header=None, index_col=0)
            psd.index.name = 'FREQ'
            psd.index = np.around(psd.index, 5) # deal with floating point errors
            psd.columns = pd.Series(run.channels, name='CHANNEL')
            psd['TIME'] = time
            psd.set_index('TIME', append=True, inplace=True)
            # Concatenate columns vertically
            psd = psd.stack()
            time_data.append(psd)
        
        # Concatenate psd series horizontally
        time_data = pd.concat(time_data, axis=1, ignore_index=True)
        time_data = time_data.reorder_levels(['CHANNEL', 'TIME', 'FREQ']
                ).sort_index()
        time_data = time_data[time_data.iloc[:,0] < 2] # strip rows of 2s
        
        # Summarize PSD files: 50% and 90% credible intervals and median
        midx = time_data.index
        time_data_np = time_data.to_numpy().T
        hpd_50 = hpd(time_data_np, alpha=0.5) # alpha = 1 - C.I.
        hpd_90 = hpd(time_data_np, alpha=0.1)
        # Return summary DataFrame
        time_summary = pd.DataFrame({
            'MEDIAN'   : time_data.median(axis=1),
            'CI_50_LO' : pd.Series(hpd_50[:,0], index=midx),
            'CI_50_HI' : pd.Series(hpd_50[:,1], index=midx),
            'CI_90_LO' : pd.Series(hpd_90[:,0], index=midx),
            'CI_90_HI' : pd.Series(hpd_90[:,1], index=midx),
        }, index=midx)
        summaries.append(time_summary)
        
        # Update progress indicator
        p.update(i)
    
    # Concatenate all summaries
    summaries = pd.concat(summaries)
    
    # Check for time gaps and fill with NaN DataFrames
    print('Checking for time gaps...')
    frequencies = summaries.index.unique(level='FREQ')
    midx = pd.MultiIndex.from_product(
        [run.channels, run.missing_times, frequencies],
        names=['CHANNEL', 'TIME', 'FREQ']
    )
    filler = pd.DataFrame(columns=summaries.columns, index=midx)
    summaries = summaries.append(filler).sort_index(level=[0, 1, 2])
    print(f'Filled {len(run.missing_times)} missing times with NaN.')
    
    # Output to file
    print(f'Writing to {run.psd_file}...')
    summaries.to_pickle(run.psd_file)
    return summaries

if __name__ == '__main__':
    main()

