import pandas as pd
import numpy as np
import csv
import os
import sys
import time_functions as tf
import itertools
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

def get_line_params(time_dir, channel):
    time = int(time_dir[-11:-1])
    # File name
    lc_file = os.path.join(
        time_dir, 'linechain_channel' + str(channel) + '.dat'
    )
    # Import first column to determine how wide DataFrame should be
    counts = get_counts(lc_file)
    # Get most likely line model (i.e., the number of spectral lines)
    model = int(counts.mode())
    if model > 0:
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
            [[time] * len(df), list(range(model)) * len(lc), 
                df.index.get_level_values('IDX')
            ], 
            names=['TIME', 'LINE', 'IDX']
        )
        return df.sort_values(by=['LINE', 'IDX'])
    else:
        return pd.DataFrame()

def cluster_params(time_dir, channel):
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
    col1 = lc.iloc[:,1] < 1e-2
    col4 = lc.iloc[:,4] < 1e-2
    print(lc[col1 & col4])
    # KMeans for peak frequencies
    freq_df = lc[[m*3+1 for m in range(model)]]
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(freq_df)
    plt.scatter(freq_df.iloc[:,0], freq_df.iloc[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.ylim(0, 0.1)
    plt.xlim(0, 0.1)
    plt.show()
    return kmeans.cluster_centers_

def summarize_params(line_df):
    '''
    Takes a DataFrame with FREQ, AMP, and QF columns for one spectral line
    Returns percentiles for each parameter in a single-row DataFrame
    Returns multiple rows if multiple lines are present in the df
    '''
    summary = []
    time = line_df.index.get_level_values('TIME')[0]
    for i in line_df.index.get_level_values('LINE').unique():
        summary.append(pd.concat(
            [pd.Series([time], name='TIME')] +
            [pd.Series(np.percentile(line_df.loc[pd.IndexSlice[:, i], col], p), 
                    name=col+'_P'+str(p) # Col name, e.g. FREQ_P10
                ) for col in line_df.columns for p in [10, 25, 50, 75, 90]
            ], axis=1
        ))
    return pd.concat(summary).set_index('TIME')

def save_summary(run, channel, summary_file):
    time_dirs = tf.get_time_dirs(run)
    summaries = []
    for i, t in enumerate(time_dirs):
        line_df = get_line_params(t, channel)
        if len(line_df) > 0:
            print(t)
            summaries.append(summarize_params(line_df))
    summaries = pd.concat(summaries)
    # Output to file
    print('Writing to ' + summary_file + '...')
    summaries.to_pickle(summary_file)
    return summaries

