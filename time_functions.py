import numpy as np
import os
import glob
from astropy.time import Time

def get_time_dirs(run):
    '''
    Returns a list of paths to all time directories in the desired run,
    relative to the current working directory (usually that of the script).
    '''
    return sorted(glob.glob(os.path.join('data', run, '*/')))

def get_gps_times(run):
    '''
    Returns a list of gps times in the desired run.
    '''
    # Get a list of the time directories
    time_dirs = get_time_dirs(run)
    # Array of run times
    return np.array([int(time_dir[-11:-1]) for time_dir in time_dirs])
    
def get_days_elapsed(gps_times):
    '''
    Converts list of gps times to days elapsed from the first time in the list.
    '''
    return (gps_times - gps_times[0]) / (60 * 60 * 24)
    
def get_iso_date(gps_int):
    '''
    Converts a gps time (int) to an ISO date.
    '''
    gps_time = Time(gps_int, format='gps')
    return Time(gps_time, format='iso')
