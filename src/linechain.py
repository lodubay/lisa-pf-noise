import pandas as pd
import numpy as np
import csv
import os
import sys
import time_functions as tf
import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from chainconsumer import ChainConsumer

def get_counts(lc_file):
    '''
    Returns just the first column of the linechain file as a pandas Series
    '''
    # Use incorrect separator to load uneven lines
    lc = pd.read_csv(lc_file, usecols=[0], header=None, squeeze=True, dtype=str)
    return pd.Series([int(row[0]) for row in lc])

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
    # Finish progress indicator
    sys.stdout.write('\n')
    sys.stdout.flush()
    # Write to CSV
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
        df = gen_model_df(run, model_file)
    # Return list of times
    return df[df.iloc[:,channel] > 0].index

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
    # Get most likely line model (i.e., the number of spectral lines)
    model = int(counts.mode())
    # Import entire data file, accounting for uneven rows
    lc = pd.read_csv(lc_file, header=None, names=range(max(counts)*3+1), sep=' ')
    # Strip of all rows that don't match the model
    lc = lc[lc.iloc[:,0] == model].dropna(1).reset_index(drop=True).rename_axis('IDX')
    # Rearrange DataFrame so same lines are grouped together
    # Make 3 column DataFrame by stacking sections vertically
    df = pd.concat([
        lc.iloc[:,c*3+1:c*3+4].set_axis(
            ['FREQ','AMP','QF'], axis=1, inplace=False
        ) for c in range(model)
    ], keys=pd.Series(range(model), name='LINE'))
    # Sort first by index, then by frequency to preserve index order
    df = df.sort_values(by=['IDX', 'FREQ'])
    # Re-sort line index to bin similar line frequencies together
    df.index = pd.MultiIndex.from_arrays(
        [list(range(model)) * len(lc), df.index.get_level_values('IDX')], 
        names=['LINE', 'IDX']
    )
    return df.sort_values(by=['LINE', 'IDX'])

def get_param_centroids(param_df, model):
    # Uses scikit-learn K-means algorithm
    #init = np.array([param_df.median().to_numpy(), param_df.median().to_numpy()])
    #print(init)
    centroids = KMeans(n_clusters=model).fit(param_df)
    #freqs = KernelDensity(kernel='gaussian').fit(param_df.iloc[:,0:1])
    #return freqs.get_params()
    return centroids.cluster_centers_

