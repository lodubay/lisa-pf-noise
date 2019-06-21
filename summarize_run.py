print('Importing dependencies...')
from pymc3.stats import hpd
import numpy as np
import pandas as pd
import os
import glob
import time_functions

# TODO make everything run with pandas?
    
def import_time_pd(time_dir):
    '''
    Import and combine all psd.dat files in a single time directory. Assumes
    file name format 'psd.dat.#' and 'psd.dat.##'. Returns a 3D array with
    the first index representing the PSD index, the second representing
    frequency, and the third representing data channel. The first item in
    every row is the frequency corresponding to that row.
    
    Uses pandas to import since it's much faster, then converts to numpy.
    '''
    print('\tImporting ' + time_dir[-11:-1] + '...')
    # Sort so that (for example) psd.dat.2 is sorted after psd.dat.19
    psd_files = sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9]'))) + \
        sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9][0-9]')))
    # Import PSD files into 3D array
    time_data = np.array([pd.read_csv(psd, sep=' ').to_numpy() for psd in psd_files])
    # Strip rows of 2s
    return time_data[:,np.min(time_data!=2., axis=(0,2))]
    
def summarize_psd(time_data, channel):
    '''
    Returns a 2D array with the median and credible intervals for one time.
    The columns are | frequency | median PSD | 50% C.I. low | 50% C.I. high | 
    90% C.I. low | 90% C.I. high |. Credible intervals are calculated using
    pymc3's highest posterior density (HPD) function, where alpha is the 
    desired probability of type I error (so, 1 - C.I.).
    
    Input
    -----
      time_data : a 3D array of all PSDs for a single time
      channel : int, index of the channel of index
    '''
    chan_data = time_data[:,:,channel] # 2D array with just one channel
    return np.hstack((
        time_data[0,:,0:1], # frequencies
        np.array([np.median(chan_data, axis=0)]).T, # medians
        hpd(chan_data, alpha=0.5), # 50% credible interval
        hpd(chan_data, alpha=0.1)  # 90% credible interval
    ))
    
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
    time_dirs = time_functions.get_time_dirs(run)
    
    # Pull PSD files from target run
    print('Importing ' + run + '...')
    # List of summary PSDs (each a 2D array), one for each time
    # Takes a long time
    summaries = [summarize_psd(import_time_pd(d), channel) for d in time_dirs]
    
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
    summaries = summarize_run(run, channel)
    print('Writing to PSD summaries file...')
    np.save(
        os.path.join('summaries', run, 'summary.' + ch_name + '.npy'),
        summaries
    )
    
def save_all_summaries(run, ch_names):
    for i in range(1,7):
        summaries = summarize_run(run, i)
        print('Writing to PSD summaries file...')
        np.save(
            os.path.join('summaries', run, 'summary.' + ch_names[i] + '.npy'),
            summaries
        )

