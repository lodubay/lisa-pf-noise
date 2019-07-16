import sys
import glob
import os

import numpy as np
from astropy.time import Time

class Progress:
    ''' A loop progress indicator class '''
    def __init__(self, iterable, message=''):
        # Set up the progress indicator with a status message
        self.progress = 0
        self.steps = len(iterable)
        sys.stdout.write(f'{message}   {self.progress}%\b')
        sys.stdout.flush()
        
    def update(self, i):
        # Call inside loop to update progress indicator
        self.progress = int((i+1) / self.steps * 100)
        prog_str = str(self.progress)
        sys.stdout.write('\b' * len(prog_str) + prog_str)
        sys.stdout.flush()
        if self.progress == 100: self.end()

    def end(self):
        # Terminate progress indicator; automatically called by self.update()
        sys.stdout.write('\n')
        sys.stdout.flush()

class Log:
    ''' A class for outputting to a log file '''
    def __init__(self, log_file=None, header='Log'):
        self.log_file = log_file
        if log_file:
            print(f'Logging output to {log_file}')
            with open(log_file, 'w+') as f:
                f.write(header)
                f.write('\n\n')
    
    def log(self, message=''):
        if self.log_file:
            with open(self.log_file, 'a+') as f:
                f.write(message)
                f.write('\n')

class Run:
    ''' A class to store information about a given run '''
    channels = np.array(range(6))
    
    def __init__(self, name, parent_dir='data'):
        self.name = name
        run_dir = os.path.join(parent_dir, name)
        if os.path.exists(run_dir):
            # Get time directories which contain the data
            self.time_dirs = sorted(glob.glob(os.path.join(run_dir, '*/')))
            # Various time formats
            self.gps_times = np.array([int(d[-11:-1]) for d in self.time_dirs])
            self.days_elapsed = self.gps2day(self.gps_times)
            self.iso_dates = self.gps2iso(self.gps_times)
            # Median time step in seconds
            self.dt = np.median(np.diff(self.gps_times))
            # List of GPS times missing from the run
            self.missing_times = self.get_missing_times()
            # Run start ISO date
            self.start_date = self.iso_dates[0]
        else:
            print(f'Could not find {name} in {parent_dir}!')
        
    def gps2day(self, gps_time):
        ''' Convert GPS time to days elapsed since run start '''
        return (gps_time - self.gps_times[0]) / (60*60*24)
    
    def day2gps(self, day):
        return int(60 * 60 * 24 * day + self.gps_times[0])
    
    def gps2iso(self, gps_time):
        ''' Convert GPS time to ISO date '''
        gps_time = Time(gps_time, format='gps')
        return Time(gps_time, format='iso')
    
    def get_exact_gps(self, approx_gps):
        ''' Converts approximate day elapsed to exact GPS time '''
        time_index = round(
            (approx_gps - self.gps_times[0])
            / (self.gps_times[-1] - self.gps_times[0]) 
            * len(self.gps_times)
        )
        return self.gps_times[time_index]
    
    def get_missing_times(self, gps_times=None, dt=None):
        missing_times = []
        if not gps_times: gps_times = self.gps_times
        if not dt: dt = self.dt
        for i in range(len(gps_times) - 1):
            diff = gps_times[i+1] - gps_times[i]
            if diff >= 2 * dt:
                # Number of new times to insert
                n = int(np.floor(diff / dt))
                # List of missing times, with same time interval
                missing_times += [gps_times[i]+dt * k for k in range(1, n+1)]
        return missing_times

