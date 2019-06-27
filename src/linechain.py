import pandas as pd
import numpy as np
import csv
import os
import sys
import time_functions as tf
import itertools
from sklearn.cluster import MiniBatchKMeans

def get_counts(lc_file):
    '''
    Returns just the first column of the linechain file as a list of ints
    as a pandas Series
    '''
    # Use incorrect separator to load uneven lines
    lc = pd.read_csv(lc_file, usecols=[0], header=None, squeeze=True, dtype=str)
    return pd.Series([int(row[0]) for row in lc])

def gen_evidence_df(run, line_evidence_file, evidence_threshold=0.5):
    '''
    Returns a DataFrame with times as rows and channels as columns. A cell is
    marked True if that time and column showed sufficient evidence for a
    line, determined by the evidence threshold
    '''
    time_dirs = tf.get_time_dirs(run)
    gps_times = tf.get_gps_times(run)
    # Create empty DataFrame
    df = pd.DataFrame(index=gps_times, columns=range(6))
    for t, c in list(itertools.product(range(len(time_dirs)), range(6))):
        # Update progress
        sys.stdout.write('\r' + str(t*6 + c+1) + '/' + str(len(time_dirs)*6))
        sys.stdout.flush()
        # linechain file name
        lc_file = os.path.join(
            time_dirs[t], 'linechain_channel' + str(c) + '.dat'
        )
        # Calculate fraction of steps with non-zero line counts
        counts = [count for count in get_counts(lc_file) if count > 0]
        detection_ratio = len(counts) / len(lc)
        # Record whether this file provides evidence for lines
        df.loc[gps_times[t], c] = detection_ratio > evidence_threshold
    sys.stdout.write('\n')
    sys.stdout.flush()
    df.to_csv(line_evidence_file, sep=' ')
    return df

def gen_model_df(run, model_file):
    '''
    Returns a DataFrame with times as rows and channels as columns. Cells
    are filled with the most likely model number
    '''
    time_dirs = tf.get_time_dirs(run)
    gps_times = tf.get_gps_times(run)
    # Create empty DataFrame
    df = pd.DataFrame(index=gps_times, columns=range(6))
    for t, c in list(itertools.product(range(len(time_dirs)), range(6))):
        # Update progress
        sys.stdout.write('\r' + str(t*6 + c+1) + '/' + str(len(time_dirs)*6))
        sys.stdout.flush()
        # linechain file name
        lc_file = os.path.join(
            time_dirs[t], 'linechain_channel' + str(c) + '.dat'
        )
        # Find the mode
        count = int(get_counts(lc_file).mode())
        df.loc[gps_times[t], c] = count
    sys.stdout.write('\n')
    sys.stdout.flush()
    df.to_csv(model_file, sep=' ')
    return df

def get_lines(run, channel, model_file):
    '''
    Returns a list of times with likely lines in the given run and channel.
    '''
    if os.path.exists(model_file):
        print('Line evidence file found. Reading...')
        df = pd.read_csv(model_file, sep=' ', index_col=0)
    else:
        print('No line evidence file found. Generating...')
        #df = gen_evidence_df(run, line_evidence_file, threshold)
        df = gen_model_df(run, model_file)
    # Return list of times
    return df[df.iloc[:,channel] > 0].index
    #return [int(t) for i, t in enumerate(df.index) if df.iloc[i,channel] > 0]

def best_line_model(run, time, channel):
    # Get time directory
    time_index = tf.get_gps_times(run).index(time)
    time_dir = tf.get_time_dirs(run)[time_index]
    # File name
    lc_file = os.path.join(
        time_dir, 'linechain_channel' + str(channel) + '.dat'
    )
    # Import first column
    counts = pd.Series(get_counts(lc_file))
    # Runoff: eliminates the least likely model, then transfers those counts
    # to the model with one fewer line (unless there is none).
    # Tries to give a bit more balance to the less-likely models
    while len(set(counts)) > 1:
        # Find the least-likely model (the count which appears the least)
        unlikely = min(set(counts), key=list(counts).count)
        # Find the nearest larger model, if it exists
        if unlikely + 1 in set(counts):
            counts[counts == unlikely] = unlikely - 1
        else:
            counts = counts[counts != unlikely]
    return max(counts)

def get_line_params(run, time, channel):
    # Get time directory
    time_index = tf.get_gps_times(run).index(time)
    time_dir = tf.get_time_dirs(run)[time_index]
    # File name
    lc_file = os.path.join(
        time_dir, 'linechain_channel' + str(channel) + '.dat'
    )
    # Import first column to determine how wide DataFrame should be
    counts = get_counts(lc_file)
    # Import entire data file, accounting for uneven rows
    lc = pd.read_csv(lc_file, header=None, names=range(max(counts)*3+1), sep=' ')
    # Column headers
    headers = ['FREQ', 'AMP', 'QF']
    # Make 3 column DataFrame by stacking sections vertically
    df = pd.concat([
        lc.iloc[:,c*3+1:c*3+4].set_axis(headers, axis=1, inplace=False) 
        for c in range(max(counts))
    ], ignore_index=True)
    # Remove NaNs and reset index
    return df[df.iloc[:,0].notna()].reset_index()

def get_model_params(run, time, channel, model):
    # Get time directory
    time_index = tf.get_gps_times(run).index(time)
    time_dir = tf.get_time_dirs(run)[time_index]
    # File name
    lc_file = os.path.join(
        time_dir, 'linechain_channel' + str(channel) + '.dat'
    )
    # Import first column to determine how wide DataFrame should be
    counts = get_counts(lc_file)
    # Import entire data file, accounting for uneven rows
    lc = pd.read_csv(lc_file, header=None, names=range(max(counts)*3+1), sep=' ')
    # Strip of all rows that don't match the model
    lc = lc[lc.iloc[:,0] == model]
    # Column headers
    headers = ['FREQ', 'AMP', 'QF']
    # Make 3 column DataFrame by stacking sections vertically
    df = pd.concat([
        lc.iloc[:,c*3+1:c*3+4].set_axis(headers, axis=1, inplace=False) 
        for c in range(model)
    ], ignore_index=True)
    # Remove NaNs and reset index
    return df[df.iloc[:,0].notna()].reset_index(drop=True)

def get_param_centroids(run, time, channel, model):
    # Uses scikit-learn K-means algorithm
    param_df = get_model_params(run, time, channel, model)
    init = np.array([[6.944806e-02, 3.072776e-20, 3.067084e+02], [3.189902e-03, 6.450174e-19, 5.410645e+03]])
    centroids = MiniBatchKMeans(n_clusters=model, init=init).fit(param_df)
    return centroids.cluster_centers_
