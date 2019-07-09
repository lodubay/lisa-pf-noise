import time_functions as tf
import psd
import plot
import linechain as lc
import pandas as pd
import numpy as np
import os
from pymc3.stats import hpd
import matplotlib.pyplot as plt

# Testing variables
run = 'ltp_run_b'
channel = 4
output_dir = os.path.join('out', run)
if not os.path.exists(output_dir): os.makedirs(output_dir)
model_file = os.path.join(output_dir, run + '_line_evidence.dat')
summary_file = os.path.join(output_dir, run + '_linechain_summary_' + str(channel) + '.pkl')

print(lc.gmm_cluster('data/ltp_run_b/run_b_1143768233/', 4))

# Import / generate summary PSD DataFrame
#if os.path.exists(summary_file):
#    df = pd.read_pickle(summary_file)
#    print('Imported PSD summaries file.')
#else:
#    print('No PSD summaries file found. Generating...')
#    df = lc.save_summary(run, channel, summary_file)

#plot.line_param(df, 'AMP', run, channel)

