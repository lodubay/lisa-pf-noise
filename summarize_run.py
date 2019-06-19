print('Importing dependencies...')
from pymc3.stats import hpd
import numpy as np
import os
import glob

def get_times(run):
    # Get a list of the time directories
    time_dirs = sorted(glob.glob(os.path.join('data', run, run + '*')))
    # Array of run times
    return np.array([int(time_dir[-10:]) for time_dir in time_dirs])
    
def get_days_elapsed(gps_times):
    return (gps_times - gps_times[0]) / (60 * 60 * 24)
    
def get_iso_date(gps_int):
    gps_time = Time(gps_int, format='gps')
    return Time(gps_time, format='iso')

def import_time(time_dir):
    # Grab the files with a single-digit index first to sort them correctly
    # Assumes file name format 'psd.dat.#' and 'psd.dat.##'
    # Returns a 3D array, formatted (PSD index, frequency, channel)
    #  first column is the frequency
    print('\tImporting ' + time_dir[-11:-1] + '...')
    psd_files = sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9]'))) + \
        sorted(glob.glob(os.path.join(time_dir, 'psd.dat.[0-9][0-9]')))
    # Import PSD files into 3D array
    time_data = np.array([np.loadtxt(psd_file) for psd_file in psd_files])
    # Strip rows of 2s
    return time_data[:,np.min(time_data!=2., axis=(0,2))]
    
def summarize_psd(time_data, channel):
    '''
    Returns a 2D array with the median and credible intervals for one time.
    The columns are | frequency | median | 50% C.I. low | 50% C.I. high | 
    90% C.I. low | 90% C.I. high |. Credible intervals are calculated using
    pymc3's highest posterior density (HPD) function, where alpha is the 
    desired probability of type I error (so, 1 - C.I.).
    
    Input
    -----
      time_data: a 3D array of all PSDs for a single time
      channel: int, index of the channel of index
    '''
    return np.hstack((
        time_data[0,:,0:1], # frequencies
        np.array([np.median(time_data[:,:,channel], axis=0)]).T, # medians
        hpd(chan_data, alpha=0.5), # 50% credible interval
        hpd(chan_data, alpha=0.1)  # 90% credible interval
    ))
    
def summarize_run(run, channel):
    # Directory stuff
    top_dir = os.getcwd()
    time_dirs = sorted(glob.glob(os.path.join(top_dir, 'data', run, run+'*/')))
    
    # Time arrays
    gps_times = np.array([int(time_dir[-11:-1]) for time_dir in time_dirs])
    delta_t_days = (gps_times - gps_times[0]) / (60 * 60 * 24)
    
    # Pull PSD files from target run
    print('Importing ' + run + '...')
    # List of summary PSDs (each a 2D array), one for each time
    # Takes a long time
    summaries = [summarize_psd(import_time(d), channel) for d in time_dirs]
    print('Adjusting arrays...')
    # Make all arrays the same length and turn into 3D array
    min_rows = min([summary.shape[0] for summary in summaries])
    return np.array([summary[:min_rows] for summary in summaries])
    
def save_summary(run, channel):
    '''
    Calls summarize_run() and writes output to binary .npy file
    
    Input
    -----
      run : string, name of the run directory
      channel : int, index of the channel of interest
    '''
    # Column headers
    cols = ['freq', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    
    # Summarize run
    summaries = summarize_run(run, channel)
    
    # Save summary file
    print('Writing to PSD summaries file...')
    np.save(
        os.path.join(os.getcwd(),'summaries',run,'summary.'+cols[channel]+'.npy'),
        summaries
    )
    

