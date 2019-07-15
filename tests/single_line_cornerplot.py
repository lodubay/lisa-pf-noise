import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd
import os
import time_functions as tf

# Looking at a cornerplot for a single spectral line
time_dir = 'data/ltp_run_b/run_b_1143766594/'
channel = 4
model = 1

time = int(time_dir[-11:-1])
# File name
lc_file = os.path.join(time_dir, 'linechain_channel' + str(channel) + '.dat')
# Import first column to determine how wide DataFrame should be
# Use incorrect separator to load uneven lines
lc = pd.read_csv(lc_file, usecols=[0], header=None, squeeze=True, dtype=str)
counts = pd.Series([int(row[0]) for row in lc])
# Import entire data file, accounting for uneven rows
lc = pd.read_csv(lc_file, header=None, names=range(max(counts)*3+1), sep=' ')
# Strip of all rows that don't match the model
lc = lc[lc.iloc[:,0] == model].dropna(1).reset_index(drop=True).rename_axis('IDX')
# Log scales
lc.iloc[:,1] = np.log(lc.iloc[:,1])
lc.iloc[:,2] = np.log(lc.iloc[:,2])
lc.iloc[:,3] = np.log(lc.iloc[:,3])
# Cornerplot
corner.corner(lc.iloc[:,1:4], 
    labels=['log(Frequency (Hz))', 'log(Amplitude)', 'log(Quality factor)'],
    #range=[(3e-3, 3.7e-3), (8e-21, 2.4e-20), (0, 800)]
)
plt.suptitle(time_dir + ' channel ' + str(channel) + 
    ', N='+str(model) + ', ' + str(len(lc)) + ' samples')
plt.show()
