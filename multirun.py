# Import and summarize a list of runs

print('Importing libraries...')
import time_functions as tf
import psd
import plot
import os

runs = [os.path.join('ltp', 'run_c'), os.path.join('ltp', 'run_b2'), 'run_k',
    os.path.join('drs', 'run_q'), os.path.join('drsCrutch', 'run_q'),
    os.path.join('ltp', 'run_b1')]
overwrite = False # Overwrite existing summary files?

for run in runs:
    summary_dir = os.path.join('summaries', run)
    summary_file = os.path.join(summary_dir, 'summary.pkl')
    # Make directory if it doesn't exist
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if os.path.exists(summary_file) and not overwrite:
        print('Summary file for ' + run + ' already found. Skipping...')
    else:
        psd.save_summary(run, summary_file)

