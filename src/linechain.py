#!/usr/bin/env python3

import os
import sys
import itertools
import argparse
from glob import glob

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
    df = pd.DataFrame(index=run.gps_times, columns=run.channels)
    p = utils.Progress(all_lc, 'Generating best model DataFrame...')
    for i, tup in enumerate(all_lc):
        t, channel = tup
        c = run.get_channel_index(channel)
        # linechain file name
        lc_file = os.path.join(run.time_dirs[t], f'linechain_channel{c}.dat')
        # Find the mode
        model = get_counts(lc_file).argmax()
        df.loc[run.gps_times[t], channel] = model
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

def summarize_linechain(run, time_dir, channel, time_counts, log):
    '''
    Returns DataFrame of percentile values for each parameter.
    
    Input
    -----
      time_dir : string, time directory
      channel : str, channel name
      time_counts : histogram of the number of times each model was chosen for 
                    this time and channel
      log : utils.Log object
    '''
    time = run.get_time(time_dir)
    ch_idx = run.get_channel_index(channel)
    log.log(f'\n-- {time} CHANNEL {channel} --')
    # Import linechain
    lc_file = f'{time_dir}linechain_channel{ch_idx}.dat'
    # Get preferred model
    model = time_counts.argmax()
    
    log.log('Line model histogram:')
    log.log(np.array2string(time_counts, max_line_width=80))
    log.log(f'{model} spectral lines found.')
    
    # Initialize summary DataFrame
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
        
        # HPD
        median = np.median(params, axis=0).flatten()[:, np.newaxis]
        hpd_50 = np.vstack(hpd(params, alpha=0.5))
        hpd_90 = np.vstack(hpd(params, alpha=0.1))
        stats = np.hstack([median, hpd_50, hpd_90])
        midx = pd.MultiIndex.from_product(
            [[channel], [time], list(range(model)), parameters],
            names=['CHANNEL', 'TIME', 'LINE', 'PARAMETER']
        )
        summary = pd.DataFrame(stats, columns=cols, index=midx)
        
        log.log('Line parameter summary:')
        log.log(summary.to_string(max_cols=80))
                
    return summary

def save_summary(run, log_file=None):
    '''
    Returns a summary DataFrame for all linechain files in the given run.
    
    Input
    -----
      run : Run object
      log_file : string, path to log file (if any)
    '''
    # Set up log file
    log = utils.Log(log_file, f'linechain.py log file for {run.name}')
    
    # Generate iterable of channels and times
    all_lc = list(itertools.product(run.channels, run.time_dirs))
    counts = []
    summaries = []
    # Set up progress indicator
    p = utils.Progress(all_lc, f'Importing {run.name} linechain...')
    for i, t in enumerate(all_lc):
        channel, time_dir = t
        ch_idx = run.get_channel_index(channel)
        # Counts for each viable model
        lc_file = os.path.join(time_dir, f'linechain_channel{ch_idx}.dat')
        time_counts = get_counts(lc_file)
        counts.append(time_counts)
        # Spectral line summary statistics
        summaries.append(
            summarize_linechain(run, time_dir, channel, time_counts, log)
        )
        # Update progress indicator
        p.update(i)
    
    # Combine counts into one DataFrame
    counts = pd.DataFrame(counts, index=pd.MultiIndex.from_product(
            [run.channels, run.gps_times], names=['CHANNEL', 'TIME']
    ))
    # Combine with DataFrame of missing times
    missing = pd.DataFrame(columns=counts.columns, 
        index=pd.MultiIndex.from_product(
            [run.channels, run.missing_times], names=['CHANNEL', 'TIME']
        )
    )
    counts = pd.concat([counts, missing]).sort_index(level=[0, 1])
    counts = counts.astype('float64')
    # Log final output
    log.log('All line counts:')
    log.log(counts.to_string(max_cols=80))
    # Output to file
    counts.to_pickle(run.linecounts_file)
    print('Model counts written to ' + run.linecounts_file)
    
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
    summaries.to_pickle(run.linechain_file)
    print('Summary written to ' + run.linechain_file)
    return counts, summaries
            
def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Generate linechain summaries and plots.'
    )
    parser.add_argument('runs', type=str, nargs='*', 
        help='run directory name (default: all folders in "data/" directory)'
    )
    parser.add_argument('-c', '--compare', dest='compare', action='store_true',
            help='compare summary plots for different runs side by side')
    parser.add_argument('--overwrite-all', dest='overwrite', 
        action='store_true',
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
    
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
        # Log output file
        log_file = os.path.join(run.summary_dir, 'linechain.log')
        # Confirm to overwrite if summary already exists
        if args.keep: overwrite = False
        elif args.overwrite: overwrite = True
        elif os.path.exists(run.linechain_file):
            over = input('Found linechain.pkl for this run. Overwrite? (y/N) ')
            overwrite = True if over == 'y' else False
        else: overwrite = True
        
        if overwrite:
            run.linecounts, run.lc_summary = save_summary(run, log_file)
        else:
            run.lc_summary = pd.read_pickle(run.linechain_file)
            run.linecounts = pd.read_pickle(run.linecounts_file)
        
        if not args.compare:
            # Plot line parameters
            print('Plotting...')
            # Plot linecount colormaps
            for i, channel in enumerate(run.channels):
                plot_file = os.path.join(run.plot_dir, f'linecounts{i}.png')
                plot.linecounts_cmap(run, channel, plot_file)
                if channel in run.lc_summary.index.unique(level='CHANNEL'):
                    for param in run.lc_summary.index.unique(level='PARAMETER'):
                        plot_file = os.path.join(
                            run.plot_dir, f'linechain_{param.lower()}{i}.png'
                        )
                        plot.linechain_scatter(
                            run, channel, param, plot_file=plot_file, show=False
                        )
    
    if args.compare:
        for i, channel in enumerate(runs[0].channels):
            plot.compare_linecounts(runs, channel, 
                    plot_file=f'out/multi_linecounts{i}.png')
    
    print('Done!')

if __name__ == '__main__':
    main()

