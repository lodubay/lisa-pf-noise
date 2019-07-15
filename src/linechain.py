#!/bin/bash

import os
import sys
import itertools
import argparse

import pandas as pd
import numpy as np
from pymc3.stats import hpd

import plot
import utils

def get_counts(lc_file):
    '''
    Returns a histogram of the counts for each model in the given linechain
    file. Output is an array where the index equals the number of lines.
    
    Input
    -----
      lc_file : string, path to the linechain file
    '''
    with open(lc_file) as lc:
        # Only use the first column
        dim = np.array([int(line.split()[0]) for line in lc])
    
    # List of unique model numbers through max from lc file
    dim_u = np.arange(np.max(dim)+1)
    # Count how often each model is used
    counts = np.array([len(dim[dim == m]) for m in dim_u])
    return counts

def gen_model_df(run, model_file):
    '''
    Returns a DataFrame with times as rows and channels as columns. Cells
    are filled with the most likely model number
    '''
    all_lc = list(itertools.product(range(len(run.time_dirs)), run.channels))
    # Create empty DataFrame
    df = pd.DataFrame(index=gps_times, columns=run.channels)
    p = utils.Progress(all_lc, 'Generating best model DataFrame...')
    for i, tup in enumerate(lc):
        t, c = tup
        # linechain file name
        lc_file = os.path.join(run.time_dirs[t], f'linechain_channel{c}.dat')
        # Find the mode
        model = get_counts(lc_file).argmax()
        df.loc[run.gps_times[t], c] = model
        # Update progress
        p.update(i)
    # Write to CSV
    df.to_csv(model_file, sep=' ')
    return df

def import_linechain(lc_file, model):
    '''
    Imports a linechain file for the given time and channel.
    Returns a 3D array of all line parameters matching the preferred model.
    
    Input
    -----
      lc_file : string, path to linechain file
      model : int, preferred model number, must be greater than 0
    '''
    # Import all rows with dim == model
    with open(lc_file) as lc:
        lines = [l.split()[1:] for l in lc if int(l.split()[0]) == model]
    
    # Configure array
    line_array = np.array(lines, dtype='float64')
    # Create 3D array with index order [index, line, parameter]
    params = []
    for p in range(3):
        param = [line_array[:,3*c+p:3*c+p+1] for c in range(model)]
        params.append(np.hstack(param))
    
    params = np.dstack(params)
    return params

def sort_params(params, log):
    '''
    Sorts the frequencies in the linechain array so that each column corresponds
    to just one spectral line. Returns an array of the same shape as params.
    
    Input
    -----
      params : 3D numpy array, the output of import_linechain()
      log : utils.Log object
    '''
    # Calculate modes for each column
    # This should give a rough value for the location of each spectral line
    modes = []
    for c in range(params.shape[1]):
        f = params[:,c,0]
        hist, bin_edges = np.histogram(f, bins=2000)
        hist_max = hist.argmax()
        mode = np.mean(bin_edges[hist_max:hist_max+2])
        modes.append(mode)
    
    modes = np.sort(np.array(modes))
    # For debugging
    log.log('Spectral line modal frequencies:')
    log.log(np.array2string(modes, max_line_width=80))
    
    # Iterate through rows and sort values to correct columns
    for i, row in enumerate(params):
        # Compute permutations of all frequencies
        f = row[:,0]
        perm = np.array(list(itertools.permutations(f)))
        # Permutations of indices
        idx = np.array(list(itertools.permutations(range(len(f)))))
        # Calculate the distances between each permutation and the modes
        dist = np.abs(perm - modes) / modes
        # Compute the total distance magnitudes
        # Inverting to lessen penalty for one value that doesn't match
        sums = np.sum(dist ** -1, axis=1) ** -1
        # Use permutation that minimizes total distance
        min_idx = idx[sums.argmin()]
        params[i] = row[min_idx]

    return params

def summarize_linechain(time_dir, channel, log):
    '''
    Returns DataFrame of percentile values for each parameter.
    
    Input
    -----
      time_dir : string, time directory
      channel : int, channel index
      log : utils.Log object
    '''
    time = int(time_dir[-11:-1])
    log.log(f'\n-- {time} CHANNEL {channel} --')
    # Import linechain
    lc_file = time_dir + 'linechain_channel' + str(channel) + '.dat'
    # Get preferred model
    counts = get_counts(lc_file)
    model = counts.argmax()
    
    log.log('Line model histogram:')
    log.log(np.array2string(counts, max_line_width=80))
    log.log(f'{model} spectral lines found.')
    
    # Initialize summary DataFrame
    #cols = pd.Series([5, 25, 50, 75, 95], name='PERCENTILE')
    cols = pd.Series(['MEDIAN', 'CI_50_LO', 'CI_50_HI', 'CI_90_LO', 'CI_90_HI'])
    parameters = ['FREQ', 'AMP', 'QF']
    summary = pd.DataFrame([], columns=cols)
    
    if model > 0:
        params = import_linechain(lc_file, model)
        # Line model
        model = params.shape[1]
        # Sort
        if model > 1:
            params = sort_params(params, log)
    
        # Summary statistics
        #percentiles = np.percentile(params, [5, 25, 50, 75, 95], axis=0)
        # Transpose to index as [line, param, index]
        #percentiles = np.transpose(percentiles, axes=(1,2,0))
        
        # HPD
        median = np.median(params, axis=0).flatten()[:, np.newaxis]
        hpd_50 = np.vstack(hpd(params, alpha=0.5))
        hpd_90 = np.vstack(hpd(params, alpha=0.1))
        stats = np.hstack([median, hpd_50, hpd_90])
        midx = pd.MultiIndex.from_product(
            [[channel], [time], list(range(model)), parameters],
            names=['CHANNEL', 'TIME', 'LINE', 'PARAMETER']
        )
        #summary = pd.DataFrame(np.vstack(percentiles), columns=cols, index=midx)
        summary = pd.DataFrame(stats, columns=cols, index=midx)
        
        log.log('Line parameter summary:')
        log.log(summary.to_string(max_cols=80))
                
    return summary

def save_summary(run, summary_file, log_file=None):
    '''
    Returns a summary DataFrame for all linechain files in the given run.
    
    Input
    -----
      run : Run object
      summary_file : string, path to summary pickle file
      log_file : string, path to log file (if any)
    '''
    # Set up log file
    log = utils.Log(log_file, f'linechain.py log file for {run.name}')
    
    # Generate all summaries
    all_lc = list(itertools.product(run.channels, run.time_dirs))
    summaries = []
    # Set up progress indicator
    p = utils.Progress(all_lc, f'Importing {run.name} linechain...')
    for i, t in enumerate(all_lc):
        channel, time_dir = t
        summaries.append(summarize_linechain(time_dir, channel, log))
        p.update(i)
    
    # Combine summaries into one DataFrame
    summaries = pd.concat(summaries, axis=0)
    midx = pd.MultiIndex.from_tuples(
        summaries.index, names=['CHANNEL', 'TIME', 'LINE', 'PARAMETER']
    )
    summaries.index = midx
    
    # Log final output
    log.log('All summaries:')
    log.log(summaries.to_string(max_cols=80))
    
    # Output to file
    summaries.to_pickle(summary_file)
    print('Output written to ' + summary_file)
    return summaries
            
def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Generate linechain summaries and plots.'
    )
    parser.add_argument('runs', type=str, nargs='*', 
        help='run directory name (default: all folders in "data/" directory)'
    )
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
    if len(args.runs) == 0: args.runs = os.listdir('data')
    
    for run in args.runs:
        # Initialize Run object
        run = utils.Run(run)
        print(f'\n-- {run.name} --')
        # Directories
        output_dir = os.path.join('out', run.name, 'summaries')
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plot_dir = os.path.join('out', run.name, 'linechain_plots')
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        # Output files
        summary_file = os.path.join(output_dir, 'linechain.pkl')
        log_file = os.path.join(output_dir, 'linechain.log')
        
        # Confirm to overwrite if summary already exists
        if args.keep: overwrite = False
        elif args.overwrite: overwrite = True
        elif os.path.exists(summary_file):
            over = input('Found linechain.pkl for this run. Overwrite? (y/N) ')
            overwrite = True if over == 'y' else False
        else: overwrite = True
        
        if overwrite:
            df = save_summary(run, summary_file, log_file)
        else:
            df = pd.read_pickle(summary_file)
        
        # Plot
        for channel in df.index.unique(level='CHANNEL'):
            for param in df.index.unique(level='PARAMETER'):
                # Only plot if more than 2 spectral lines
                if df.loc[channel, :, :, param].shape[0] > 2:
                    plot_file = os.path.join(
                        plot_dir, f'linechain_{param.lower()}{channel}.png'
                    )
                    plot.linechain_scatter(
                        df, param, run, channel, 
                        plot_file=plot_file, show=False
                    )

if __name__ == '__main__':
    main()

