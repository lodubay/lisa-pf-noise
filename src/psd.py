from pymc3.stats import hpd
import numpy as np
import pandas as pd
import os
import glob
import sys
import time_functions as tf

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
    # Sort so that (for example) psd.dat.2 is sorted after psd.dat.19
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
            np.around(time_data.index.get_level_values('FREQ').unique(), 5)], 
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
    print('Importing ' + run + '...')
    # Concatenate DataFrames of all times; takes a while
    summaries = []
    for i, d in enumerate(time_dirs):
        summaries.append(summarize_psd(d))
        # Progress indicator
        sys.stdout.write('\r' + str(i+1) + '/' + str(len(time_dirs)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    summaries = pd.concat(summaries)
    # Output to file
    print('Writing to ' + summary_file + '...')
    summaries.to_pickle(summary_file)
    return summaries

def get_exact_freq(summary, approx_freq):
    '''
    Takes an approximate input frequency and returns the closest measured
    frequency in the data.
    '''
    gps_times = list(summary.index.get_level_values(0).unique())
    freqs = list(summary.xs(gps_times[0]).index)
    freq_index = round(approx_freq / (max(freqs) - min(freqs)) * len(freqs))
    return freqs[freq_index]
