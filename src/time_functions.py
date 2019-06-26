import os
import glob
import numpy as np
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

def gps2day(run, gps_time):
    '''
    Takes an exact gps time and returns the number of days from the start of run
    '''
    gps_times = get_gps_times(run)
    return (gps_time - gps_times[0]) / (60*60*24)
    
def gps2day_list(gps_times):
    '''
    Converts list of gps times to days elapsed from the first time in the list.
    '''
    return [(t - gps_times[0]) / (60*60*24) for t in gps_times]
    
def gps2iso(gps_int):
    '''
    Converts a gps time (int) to an ISO date.
    '''
    gps_time = Time(gps_int, format='gps')
    return Time(gps_time, format='iso')

def iso2gps(iso_date):
    return int(Time(iso_date, format='gps'))

def day2gps(run, day):
    gps_times = get_gps_times(run)
    return int(60 * 60 * 24 * day + gps_times[0])

def get_exact_gps(run, approx_gps):
    gps_times = get_gps_times(run)
    time_index = round(
        (approx_gps - gps_times[0]) 
        / (gps_times[-1] - gps_times[0]) 
        * len(gps_times)
    )
    return gps_times[time_index]

def get_exact_time(summary, approx_day):
    '''
    Converts an approximate days elapsed to gps time. Returns tuple of
    gps time and exact days elapsed.

    Input
    ----
      run : name of the run
      approx_day : approximate number of days from start of run
    '''
    gps_times = list(summary.index.get_level_values(0).unique())
    days_elapsed = get_days_elapsed(gps_times)
    time_index = round(approx_day / max(days_elapsed) * len(days_elapsed))
    return gps_times[time_index], days_elapsed[time_index]

