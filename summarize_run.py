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
    # Parameters:
    #  run: a 3D array of all PSDs for a single time
    #  channel: int from 1-6, the channel index we're interested in
    # Returns:
    #  summary_psd: a 2D array with the mean PSD function and credible intervals
    #  | frequency | median | mean - CI1 | mean + CI1 | mean - CI2 | ...
    # Frequencies column
    frequencies = time_data[0,:,0:1]
    # Create 2D array with format (frequency, chain index)
    chan_data = time_data[:,:,channel]
    # Medians column
    medians = np.array([np.median(chan_data, axis=0)]).T
    # Credible intervals columns
    # Uses the highest posterior density (HPD), or minimum width Bayesian CI
    # pymc3 uses alpha to mean Type I error probability, so adjust
    alpha = (0.5, 0.9)
    credible_intervals = [hpd(chan_data, alpha=1-a) for a in alpha]
    return np.hstack((frequencies, medians) + tuple(credible_intervals))
    
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
      channel : string, name of the channel of interest
    '''
    # Column headers
    cols = ['freq', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    
    # Summarize run
    summaries = summarize_run(run, cols.index(channel))
    
    # Save summary file
    print('Writing to PSD summaries file...')
    np.save(
        os.path.join(os.getcwd(),'summaries',run,'summary.'+channel+'.npy'),
        summaries
    )
    

