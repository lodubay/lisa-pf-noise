#!/usr/bin/env python3

import os
from glob import glob
import sys
import argparse

import numpy as np
import pandas as pd
from pymc3.stats import hpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.ticker as tkr

import linechain as lc
import import_psd
import plot
import utils

def main():
    '''
    Initializes the argument parser and sets up the log file for the analysis
    scripts. Then calls functions to analyze individual runs and compare
    multiple runs.
    '''
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Generate PSD summaries and plots.'
    )
    parser.add_argument('runs', type=str, nargs='*', 
        help='run directory name (default: all folders in "data/" directory)'
    )
    parser.add_argument('--overwrite-all', dest='overwrite', action='store_true',
        help='re-generate summary files even if they already exist (default: \
              ask for each run)'
    )
    parser.add_argument('--keep-all', dest='keep', action='store_true',
        help='do not generate summary file if it already exists (default: ask \
              for each run)'
    )
    args = parser.parse_args()
    # Add all runs in data directory if none are specified
    if len(args.runs) == 0: 
        args.runs = glob(f'data{os.sep}*{os.sep}*{os.sep}')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    
    # Import impacts file, if any
    impacts_file = 'impacts.dat'
    impacts = np.array([])
    if os.path.exists(impacts_file):
        impacts = get_impacts(impacts_file)
    
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
        # Log output file
        log_file = os.path.join(run.summary_dir, 'psd.log')
        log = utils.Log(log_file, f'psd.py log file for {run.name}')
        # Confirm to overwrite if summary already exists
        if args.keep: overwrite = False
        elif args.overwrite: overwrite = True
        elif os.path.exists(run.psd_file):
            over = input('Found psd.pkl for this run. Overwrite? (y/N) ')
            overwrite = True if over == 'y' else False
        else: overwrite = True

        # Import / generate summary PSD DataFrame
        if overwrite:
            df = import_psd.summarize(run)
        else:
            df = pd.read_pickle(run.psd_file)

def single_run(run):
    '''
    Generate analysis plots for a single run.
    '''

def compare(runs):
    '''
    Generate comparison plots for a list of runs.
    '''
        
