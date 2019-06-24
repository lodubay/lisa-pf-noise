from pymc3.stats import hpd
import numpy as np
import pandas as pd
import os
import glob
import time_functions as tf
    
def import_channel(time_dir, channel):
    '''
    Import and combine all psd.dat files in a single time directory 
    for one channel. Assumes file name format 'psd.dat.#' and 'psd.dat.##'.
    Returns a DataFrame with frequency increasing down the rows and 
    chain index increasing across the columns.

    Input
    -----
      time_dir : relative path to the directory
      channel : string, channel header
    '''
    print('\tImporting ' + time_dir[-11:-1] + '...')
    # Column names
    cols = ['FREQ', 'a_x', 'a_y', 'a_z', 'theta_x', 'theta_y', 'theta_z']
    # Sort so that (for example) psd.dat.2 is sorted after psd.dat.19
    psd_files = sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9]'))) + \
        sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9][0-9]')))
    # Import PSD files into DataFrame
    time_data = pd.concat(
        [ pd.read_csv(
            psd, sep=' ', usecols=range(len(cols)), names=cols, index_col=0
        )[channel] for psd in psd_files ],
        axis=1,
        ignore_index=True
    )
    # Round frequency index to deal with floating point accuracy
    time_data.index = np.around(time_data.index.get_level_values('FREQ'), 5)
    # Strip rows of 2s
    return time_data[time_data.iloc[:,0] < 2]
    
def summarize_psd(time_data, gps_time):
    '''
    Returns a DataFrame with the median and credible intervals for one time.
    Credible intervals are calculated using
    pymc3's highest posterior density (HPD) function, where alpha is the 
    desired probability of type I error (so, 1 - C.I.).
    
    Input
    -----
      time_data : a DataFrame of all PSDs for a single time
      channel : string, channel header
    '''
    freqs = time_data.index
    time_data_np = time_data.to_numpy().T
    hpd_50 = hpd(time_data_np, alpha=0.5)
    hpd_90 = hpd(time_data_np, alpha=0.1)
    return pd.DataFrame({
        'TIME'      : gps_time,
        'MEDIAN'    : time_data.median(axis=1),
        'CI_50_LO'  : pd.Series(hpd_50[:,0], index=freqs),
        'CI_50_HI'  : pd.Series(hpd_50[:,1], index=freqs),
        'CI_90_LO'  : pd.Series(hpd_90[:,0], index=freqs),
        'CI_90_HI'  : pd.Series(hpd_90[:,1], index=freqs),
    }, index=time_data.index)
    
def summarize_run(run, channel):
    '''
    Returns a multi-index DataFrame of PSD summaries across multiple times 
    from one run folder. The first index represents time and the 
    second frequency.
    
    Input
    -----
      run : string, name of the run directory
      channel : string, channel header
    '''
    # Get list of time directories within run directory
    time_dirs = tf.get_time_dirs(run)
    # For testing purposes
    
    # Pull PSD files from target run
    print('Importing ' + run + '...')
    # Concatenate DataFrames of all times; takes a while
    summaries = pd.concat([
        summarize_psd(import_channel(d, channel), int(d[-11:-1])) 
        for d in time_dirs
    ])
    # Set TIME and FREQ columns as indices
    return summaries.set_index(['TIME', summaries.index])
    
def save_summary(run, ch_name):
    '''
    Calls summarize_run() writes output to a pickle file, and returns output.
    
    Input
    -----
      run : string, name of the run directory
      ch_name : string, name of the channel of interest
    '''
    summaries = summarize_run(run, ch_name)
    print('Writing to PSD summaries file...')
    summaries.to_pickle(
        os.path.join('summaries', run, 'summary.' + ch_name + '.pkl')
    )
    return summaries
    
def save_all_channels(run, channels):
    for channel in channels:
        save_summary(run, channel)

def load_summary(run, ch_name):
    return pd.read_pickle(
        os.path.join('summaries', run, 'summary.' + ch_name + '.pkl')
    )

def get_time_slice(summary, gps_time):
    return summary.xs(gps_time)

def get_exact_freq(summary, approx_freq):
    '''
    Takes an approximate input frequency and returns the closest measured
    frequency in the data.
    '''
    gps_times = list(summary.index.get_level_values(0))
    freqs = list(summary.xs(gps_times[0]).index)
    freq_index = int(approx_freq / (max(freqs) - min(freqs)) * len(freqs))
    return freqs[freq_index]

def get_freq_slice(summary, freq):
    '''
    Returns a DataFrame sliced along the input frequency. 
    '''
    return summary.xs(freq, level='FREQ')

def unstack_median(summary):
    '''
    Returns an unstacked DataFrame of just median PSD values, with time
    along columns and frequency along rows.
    '''
    return summary['MEDIAN'].unstack(level=0)

def get_median_psd(summary):
    '''
    Returns PSD medianed across all times.
    '''
    return summary['MEDIAN'].unstack(level=0).median(axis=1)
