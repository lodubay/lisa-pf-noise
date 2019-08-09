import sys
from glob import glob
import os
import argparse
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    def __init__(self, log_file, header='Log'):
        self.log_file = log_file
        print(f'Logging output to {log_file}')
        with open(log_file, 'w+') as f:
            f.write(header)
            f.write('\n')
    
    def log(self, message=''):
        with open(self.log_file, 'a+') as f:
            f.write(message)
            f.write('\n')

class Run:
    ''' A class to store information about a given run '''
    channels = np.array(['x', 'y', 'z', 'θ', 'η', 'ϕ'])
    
    def __init__(self, path, name=None):
        if os.path.exists(path):
            # Path names
            self.path = path
            split_path = path.split(os.sep)
            self.parent_dir = split_path[0]
            self.mode = split_path[1]
            self.name = split_path[2]
            
            # Output directories
            self.output_dir = os.path.join('out', self.mode, self.name)
            self.summary_dir = os.path.join(self.output_dir, 'summaries')
            if not os.path.exists(self.summary_dir): 
                os.makedirs(self.summary_dir)
            self.plot_dir = os.path.join(self.output_dir, 'plots')
            if not os.path.exists(self.plot_dir): 
                os.makedirs(self.plot_dir)
            self.log_dir = os.path.join(self.output_dir, 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            
            # Summary file paths
            self.psd_file = os.path.join(self.summary_dir, 'psd.pkl')
            self.psd_log = os.path.join(self.summary_dir, 'psd.log')
            self.fft_log = os.path.join(self.summary_dir, 'fft.log')
            self.linecounts_file = os.path.join(self.summary_dir, 'linecounts.pkl')
            self.linechain_file = os.path.join(self.summary_dir, 'linechain.pkl')
            
            # Get time directories which contain the data
            self.time_dirs = sorted(glob(os.path.join(path, '*'+os.sep)))
            # Various time formats
            self.gps_times = np.array(
                sorted([self.get_time(d) for d in self.time_dirs])
            )
            self.days_elapsed = self.gps2day(self.gps_times)
            self.iso_dates = self.gps2iso(self.gps_times)
            # Median time step in seconds
            self.dt = np.median(np.diff(self.gps_times))
            # List of GPS times missing from the run
            self.missing_times = self.get_missing_times()
            # Run start ISO date
            self.start_date = self.iso_dates[0]
            
        else:
            raise FileNotFoundError(f'{path} does not exist')
    
    def get_time(self, time_dir):
        return int(time_dir[-11:-1])
        
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
        if not dt: dt = int(self.dt)
        for i in range(len(gps_times) - 1):
            diff = gps_times[i+1] - gps_times[i]
            if diff > dt + 1:
                # Number of new times to insert
                n = int(np.ceil(diff / dt)-1)
                # List of missing times, with same time interval
                missing_times += [gps_times[i] + dt * k for k in range(1, n+1)]
        return missing_times
    
    def get_channel_index(self, channel):
        return self.channels.tolist().index(channel)


def init_runs(paths):
    '''
    Initializes run objects from a list of data paths. Skips directories that
    don't exist.
    '''
    runs = []
    for path in paths:
        try:
            run = Run(path)
            runs.append(run)
        except FileNotFoundError:
            print(f'{path} not found, skipping...')
    return runs


def gps2day(gps_times):
    ''' Convert numpy array of GPS times to days elapsed since start of list '''
    return (gps_times - gps_times[0]) / (60*60*24)


def get_exact_freq(summary, approx_freqs):
    '''
    Takes an approximate input frequency and returns the closest measured
    frequency in the data.
    '''
    freqs = np.array(sorted(summary.index.unique(level='FREQ')))
    freq_indices = np.round(
            approx_freqs / (np.max(freqs) - np.min(freqs)) * freqs.shape[0]
    ).astype(int)
    return freqs[freq_indices]

def get_impacts(impacts_file):
    cols = ['DATE', 'GPS', 'P_MED', 'P_CI_LO', 'P_CI_HI', 'FACE', 'LOCAL', 
            'LAT_SC', 'LON_SC', 'LAT_SSE', 'LON_SSE', 'LPF_X', 'LPF_Y', 'LPF_Z']
    impacts = pd.read_csv(impacts_file, sep=' ', names=cols, na_values='-')
    return impacts

def add_parser(description, comparison=True, title=False):
    # Argument parser
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('runs', type=str, nargs='*', 
        help='run directory name (default: all folders in "data/" directory)')
    if comparison:
        parser.add_argument('-c', '--compare', dest='compare', action='store_true',
                help='generate additional side-by-side run comparison plots')
    parser.add_argument('--overwrite-all', dest='overwrite', 
        action='store_true',
        help='re-generate summary files even if they already exist (default: \
              ask for each run)')
    parser.add_argument('--keep-all', dest='keep', action='store_true',
        help='do not generate summary file if it already exists (default: ask \
              for each run)')
    if title:
        parser.add_argument('--no-title', dest='title', action='store_false',
                help='do not include a plot title')
    args = parser.parse_args()
    
    # Add all runs in data directory if none are specified
    if len(args.runs) == 0: 
        args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')
    
    return args


def gridplot(fn, df, channel, iterable, suptitle, xlabel, ylabel, config):
    '''
    Plots a given function for a list of values in a grid.
    
    Input
    -----
      fn : the plotting function
      df : DataFrame, the full summary output from import_psd.py
      channel : str, the channel to investigate
      iterable : list, the values to pass to fn
      suptitle : str, the plot title
      xlabel : str, the x axis label
      ylabel : str, the y axis label
      config : ConfigParser
    '''
    # Automatically create grid of axes
    nrows = int(np.floor(float(len(iterable)) ** 0.5))
    ncols = int(np.ceil(1. * len(iterable) / nrows))
    # Set up figure
    fig = plt.figure(
            figsize=(config.getfloat('Placement', 'fig_x_scale') * ncols, 
            config.getfloat('Placement', 'fig_y_scale') * nrows))
    fig.suptitle(suptitle, fontsize=config.getfloat('Font', 'fig_title_size'))
    
    # Subplots
    for i, item in enumerate(iterable):
        # Set up subplot
        ax = fig.add_subplot(nrows, ncols, i+1)
        
        # Plot function
        fn(fig, ax, df, channel, item)
        
        # Subplot config
        ax.set_title(ax.get_title(),
                fontsize=config.getfloat('Font', 'subplot_title_size'))
        if i >= len(iterable) - ncols: # x axis label on bottom plots only
            ax.set_xlabel(xlabel, 
                    fontsize=config.getfloat('Font', 'ax_label_size'))
        if i % ncols == 0: # y axis label on left plots only
            ax.set_ylabel(ylabel, 
                    fontsize=config.getfloat('Font', 'ax_label_size'))
        ax.tick_params(axis='both', which='major',
                labelsize=config.getfloat('Font', 'tick_label_size'),
                length=config.getfloat('Tick','major_tick_length'))
        ax.tick_params(axis='both', which='minor', 
                length=config.getfloat('Tick', 'minor_tick_length'))
        
    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend
    if len(handles) == 3:
        order = [0, 2, 1]
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                fontsize=config.getfloat('Font', 'legend_label_size'))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    
    return fig


def compareplot(fn, runs, dfs, channel, iterable, suptitle, xlabels, ylabel,
        config):
    # Set up figure
    nrows = len(iterable)
    ncols = len(runs)
    fig, axes = plt.subplots(nrows, ncols, sharex='col',
            figsize=(config.getfloat('Placement', 'fig_x_scale') * ncols, 
                     config.getfloat('Placement', 'fig_y_scale') * nrows))
    fig.suptitle(suptitle, fontsize=config.getfloat('Font', 'fig_title_size'))

    # Subplots
    for r, run in enumerate(runs):
        xlabel = xlabels[r]
        df = dfs[r]
        for i, item in enumerate(iterable):
            ax = axes[i, r]

            # Plot function
            fn(fig, ax, df, channel, item)
            
            # Subplot config
            ax.set_title(ax.get_title(),
                    fontsize=config.getfloat('Font', 'subplot_title_size'))
            if i >= len(iterable) - ncols: # x axis label on bottom plots only
                ax.set_xlabel(xlabel, 
                        fontsize=config.getfloat('Font', 'ax_label_size'))
            if i % ncols == 0: # y axis label on left plots only
                ax.set_ylabel(ylabel, 
                        fontsize=config.getfloat('Font', 'ax_label_size'))
            ax.tick_params(axis='both', which='major',
                    labelsize=config.getfloat('Font', 'tick_label_size'),
                    length=config.getfloat('Tick','major_tick_length'))
            ax.tick_params(axis='both', which='minor', 
                    length=config.getfloat('Tick', 'minor_tick_length'))
        
    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend
    if len(handles) == 3:
        order = [0, 2, 1]
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                fontsize=config.getfloat('Font', 'legend_label_size'))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    
    return fig

