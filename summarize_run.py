print('Importing dependencies...')
from pymc3.stats import hpd
import numpy as np
import pandas as pd
import os
import glob
import time_functions as tf

# TODO make everything run with pandas?
    
def import_channel(time_dir, channel):
    '''
    Import and combine all psd.dat files in a single time directory 
    for one channel. Assumes file name format 'psd.dat.#' and 'psd.dat.##'.
    Returns a DataFrame with frequency increasing down the rows and 
    chain index increasing across the columns.
    '''
    print('\tImporting ' + time_dir[-11:-1] + '...')
    # Column names
    cols = ['Frequency', 'a_x', 'a_y', 'a_z', 'theta_x', 'theta_y', 'theta_z']
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
    # Strip rows of 2s
    return time_data[time_data.iloc[:,0] < 2]
    
def summarize_psd(time_data):
    '''
    Returns a 2D array with the median and credible intervals for one time.
    The columns are | frequency | median PSD | 50% C.I. low | 50% C.I. high | 
    90% C.I. low | 90% C.I. high |. Credible intervals are calculated using
    pymc3's highest posterior density (HPD) function, where alpha is the 
    desired probability of type I error (so, 1 - C.I.).
    
    Input
    -----
      time_data : a DataFrame of all PSDs for a single time
      channel : int, index of the channel of index
    '''
    freqs = time_data.index
    time_data_np = time_data.to_numpy()
    hpd_50 = hpd(time_data_np, alpha=0.5)
    hpd_90 = hpd(time_data_np, alpha=0.1)
    print(time_data.median(axis=1))
    df = pd.DataFrame({
        'Median'     : time_data.median(axis=1),
        'Low 50% CI' : pd.Series(hpd_50[:,0], index=freqs),
        'High 50% CI': hpd.Series(pd_50[:,1],
        'Low 90% CI' : hpd_90[:,0],
        'High 50% CI': hpd_90[:,1]
    }, index=time_data.index)
    print(df)
    return df
    
def summarize_run(run, channel):
    '''
    Returns a 3D array of PSD summaries across multiple times from one run
    folder. The first index represents time, the second frequency and the third
    columns (see summarize_psd).
    
    Note that actual times are not included in the final output. These need to
    be generated using time_functions.py.
    
    Input
    -----
      run : string, name of the run directory
      channel : int, index of the channel of interest
    '''
    # Get list of time directories within run directory
    time_dirs = tf.get_time_dirs(run)
    
    # Pull PSD files from target run
    print('Importing ' + run + '...')
    # List of summary PSDs (each a 2D array), one for each time
    # Takes a long time
    summaries = [summarize_psd(import_channel(d, channel)) for d in time_dirs]
    
    # Make all arrays the same length and turn into 3D array
    print('Adjusting arrays...')
    min_rows = min([summary.shape[0] for summary in summaries])
    return np.array([summary[:min_rows] for summary in summaries])
    
def save_summary(run, channel, ch_name):
    '''
    Calls summarize_run() and writes output to binary .npy file
    
    Input
    -----
      run : string, name of the run directory
      channel : int, index of the channel of interest
      ch_name : string, name of the channel of interest
    '''
    summaries = summarize_run(run, ch_name)
    print('Writing to PSD summaries file...')
    np.save(
        os.path.join('summaries', run, 'summary.' + ch_name + '.npy'),
        summaries
    )

