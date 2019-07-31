#!/usr/bin/env python3

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import utils
import fft

# Argument parser
parser = argparse.ArgumentParser(description='FFT analysis.')
parser.add_argument('runs', type=str, nargs='*', 
    help='run directory name (default: all folders in "data/" directory)'
)
args = parser.parse_args()
# Add all runs in data directory if none are specified
if len(args.runs) == 0: 
    args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')

# Initialize run objects; skip missing directories
runs = utils.init_runs(args.runs)

# Frequencies to examine
frequencies = np.array([3e-2, 5e-3, 1e-3])

# Set up log file
log_dir = os.path.join('out', 'multirun', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log = utils.Log(os.path.join('out', 'multirun', 'logs', 'fft.log'), 
        f'fft.log file for run comparison')

# One plot per channel
for c, channel in enumerate(runs[0].channels):
    print(f'\n-- Channel {channel} --')
    log.log(f'\nChannel {channel}')    
    
    nrows = len(frequencies)
    ncols = len(runs)
    fig, axes = plt.subplots(nrows, ncols, sharex='all')
    fig.suptitle(f'Channel {channel}\nFFT of power at selected frequencies',
            fontsize=22)
    
    # Each run gets its own column of subplots
    for r, run in enumerate(runs):
        # Import PSD summaries
        df = pd.read_pickle(run.psd_file)
        # Remove large gap in LTP run
        if run.name == 'run_b' and run.mode == 'ltp':
            df = df[df.index.get_level_values('TIME') >= 1143962325]
        
        # Each frequency gets its own row
        for i, freq in enumerate(frequencies):
            freq = utils.get_exact_freq(df, freq)
            freq_str = f'%s mHz' % float('%.3g' % (freq * 1000.))
            log.log(f'\n{freq_str}')
            ax = axes[i, r]
            
            # Plot FFT at specific frequency
            rfftfreq, rfft = fft.fft(df, channel, freq, log)
            psd = np.absolute(rfft)**2
            ax.plot(rfftfreq, psd, color='#0077c8')
            
            # Axis configuration
            if i == 0:
                ax.set_title(f'{run.mode.upper()}', fontsize=22)
            ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
            if i+1 == len(frequencies):
                ax.set_xlabel('Frequency (Hz)', fontsize=20)
            ax.set_yscale('log')
            ax.yaxis.set_minor_locator(tkr.NullLocator())
            ax.set_ylabel(f'PSD at ' + freq_str, fontsize=20)
    
    plot_file = os.path.join('out', 'multirun', 'plots', f'fft{c}.png')
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()

'''
ltp = utils.Run('data/ltp/run_b')
ltpsum = pd.read_pickle(ltp.psd_file)
ltp.psd_summary = ltpsum[ltpsum.index.get_level_values('TIME') >= 1143962325]

drs = utils.Run('data/drs/run_b')
drs.psd_summary = pd.read_pickle(drs.psd_file)

runs = [ltp, drs]
'''
