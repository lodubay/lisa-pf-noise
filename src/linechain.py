import pandas as pd
import numpy as np
import csv
import os
import sys
import time_functions as tf
import itertools

def gen_evidence_df(run, line_evidence_file, evidence_threshold=0.5):
    '''
    Returns a DataFrame with times as rows and channels as columns. A cell is
    marked True if that time and column showed sufficient evidence for a
    line, determined by the evidence threshold
    '''
    time_dirs = tf.get_time_dirs(run)
    gps_times = tf.get_gps_times(run)
    
    # Set up progress indicator
    max_iter = len(time_dirs) * 6
    
    # Create empty DataFrame
    df = pd.DataFrame(index=gps_times, columns=range(6))
    for t, c in list(itertools.product(range(len(time_dirs)), range(6))):
        # Update progress
        sys.stdout.write('\r' + str(t * 6 + c + 1) + '/' + str(max_iter))
        sys.stdout.flush()
        # linechain file name
        lc_file = os.path.join(
            time_dirs[t], 'linechain_channel' + str(c) + '.dat'
        )
        with open(lc_file, 'r') as f:
            # Use incorrect separator to load uneven lines
            lc = pd.read_csv(f, usecols=[0], engine='c', header=None, 
                squeeze=True, dtype=str
            )
        # Calculate fraction of steps with non-zero line counts
        counts = [int(row[0]) for row in lc if int(row[0]) > 0]
        detection_ratio = len(counts) / len(lc)
        # Record whether this file provides evidence for lines
        df.loc[gps_times[t], c] = detection_ratio > evidence_threshold
    df.to_csv(line_evidence_file, sep=' ')
    return df

def get_lines(run, channel, line_evidence_file, threshold=0.5):
    '''
    Returns a list of times with likely lines in the given run and channel.
    '''
    try:
        print('Line evidence file found. Reading...')
        df = pd.read_csv(line_evidence_file, sep=' ', index_col=0)
    except FileNotFoundError:
        print('No line evidence file found. Generating...')
        df = gen_evidence_df(run, line_evidence_file, threshold)
    # Return list of times
    return [int(t) for t in df.index if df.loc[t,str(channel)]]
