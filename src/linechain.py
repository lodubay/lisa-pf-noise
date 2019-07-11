import pandas as pd
import numpy as np
import os
import sys
import time_functions as tf
import itertools


def get_counts(lc_file):
    '''
    Returns a histogram of the counts for each model in the given linechain file
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
    time_dirs = tf.get_time_dirs(run)
    gps_times = tf.get_gps_times(run)
    # Create empty DataFrame
    df = pd.DataFrame(index=gps_times, columns=range(6))
    for t, c in list(itertools.product(range(len(time_dirs)), range(6))):
        # Update progress
        sys.stdout.write('\r' + str(t*6 + c+1) + '/' + str(len(time_dirs)*6))
        sys.stdout.flush()
        # linechain file name
        lc_file = os.path.join(
            time_dirs[t], 'linechain_channel' + str(c) + '.dat'
        )
        # Find the mode
        model = get_counts(lc_file).argmax()
        df.loc[gps_times[t], c] = model
    # Finish progress indicator
    sys.stdout.write('\n')
    sys.stdout.flush()
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
        

def sort_params(params, log_file=None):
    '''
    Sorts the frequencies in the linechain array so that each column corresponds
    to just one spectral line. Returns an array of the same shape as params.
    
    Input
    -----
      params : 3D numpy array, the output of import_linechain()
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
    if log_file:
        with open(log_file, 'a+') as log:
            log.write('Spectral line modal frequencies:\n')
            log.write(np.array2string(modes, max_line_width=80))
            log.write('\n')
    
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


def summarize_linechain(time_dir, channel, log_file=None):
    '''
    Returns DataFrame of percentile values for each parameter.
    
    Input
    -----
      time_dir : string, time directory
      channel : int, channel index
    '''
    time = int(time_dir[-11:-1])
    # Import linechain
    lc_file = time_dir + 'linechain_channel' + str(channel) + '.dat'
    # Get preferred model
    counts = get_counts(lc_file)
    model = counts.argmax()
    
    # Initialize summary DataFrame
    percentiles = np.array([5, 25, 50, 75, 95])
    cols = pd.Series(percentiles.astype('str'), name='PERCENTILE')
    parameters = ['FREQ', 'AMP', 'QF']
    summary = pd.DataFrame([], columns=cols)
    
    if model > 0:
        params = import_linechain(lc_file, model)
        # Line model
        model = params.shape[1]
        # Sort
        if model > 1:
            params = sort_params(params, log_file)
    
        # Summary statistics
        p_array = np.percentile(params, percentiles, axis=0)
        # Transpose to index as [line, param, index]
        p_array = np.transpose(p_array, axes=(1,2,0))
        midx = pd.MultiIndex.from_product(
            [[channel], [time], list(range(model)), parameters],
            names=['CHANNEL', 'TIME', 'LINE', 'PARAMETER']
        )
        summary = pd.DataFrame(np.vstack(p_array), columns=cols, index=midx)
    
    # Write to log file
    if log_file:
        with open(log_file, 'a+') as log:
            log.write('-- ' + str(time) + ' CHANNEL ' + str(channel) + ' --\n')
            log.write('Line model histogram:\n') 
            log.write(np.array2string(counts, max_line_width=80))
            log.write('\n' + str(model) + ' spectral lines found\n')
            if model > 0:
                log.write('Line parameter summary:\n')
                log.write(summary.to_string(max_cols=80))
                log.write('\n')
            log.write('\n')
                
    return summary


def save_summary(run, summary_file, log_file=None):
    '''
    Returns a summary DataFrame for all linechain files in the given run.
    
    Input
    -----
      run : string, name of run
    '''
    time_dirs = tf.get_time_dirs(run)
    # Set up log file
    if log_file:
        print('Logging output to ' + log_file)
        with open(log_file, 'w+') as log:
            log.write('linechain.py log file for ' + run + '\n\n')
    # Set up progress indicator
    sys.stdout.write('Importing ' + run + ' linechain...   0%\b')
    sys.stdout.flush()
    steps = 6 * len(time_dirs)
    # Generate all summaries
    summaries = []
    for i, t in enumerate(list(itertools.product(range(6), time_dirs))):
        channel, time_dir = t
        summaries.append(summarize_linechain(time_dir, channel, log_file))
        # Update progress indicator
        progress = str(int((i+1) / steps * 100))
        sys.stdout.write('\b' * len(progress) + progress)
        sys.stdout.flush()
    
    # Finish progress indicator
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    summaries = pd.concat(summaries, axis=0)
    midx = pd.MultiIndex.from_tuples(
        summaries.index, names=['CHANNEL', 'TIME', 'LINE', 'PARAMETER']
    )
    summaries.index = midx
    # Log final output
    if log_file:
        with open(log_file, 'a+') as log:
            log.write('All summaries:\n')
            log.write(summaries.to_string(max_cols=80))
            log.write('\n')
    # Output to file
    summaries.to_pickle(summary_file)
    print('Output written to ' + summary_file + '...')
    return summaries
            

run = 'ltp_run_c'
summary_file = 'out/' + run + '/linechain.pkl'
log_file = 'out/' + run + '/linechain.log'
save_summary(run, summary_file, log_file)

