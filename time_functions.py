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
    return [int(time_dir[-11:-1]) for time_dir in time_dirs]
    
def get_days_elapsed(gps_times):
    '''
    Converts list of gps times to days elapsed from the first time in the list.
    '''
    return [(int(t) - int(gps_times[0])) / (60*60*24) for t in gps_times]
    
def get_iso_date(gps_int):
    '''
    Converts a gps time (int) to an ISO date.
    '''
    gps_time = Time(gps_int, format='gps')
    return Time(gps_time, format='iso')

def approx_time(run, approx_day):
    '''
    Converts an approximate days elapsed to gps time. Returns tuple of
    gps time and exact days elapsed.

    Input
    ----
      run : name of the run
      approx_day : approximate number of days from start of run
    '''
    gps_times = get_gps_times(run)
    days_elapsed = get_days_elapsed(gps_times)
    time_index = int(approx_day / max(days_elapsed) * len(days_elapsed))
    return gps_times[time_index], days_elapsed[time_index]
