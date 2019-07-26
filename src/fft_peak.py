import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psd
import utils

# Plot parameters
run_dir = 'data/ltp/run_b'
channel = 'y'
freq = [1e-2]

# Analysis parameters
bin_width = 10 # number of points on either side of peak to bin
f_min = 2e-5 # minimum estimated frequency of peak
f_max = 2.5e-5 # maximum estimated frequency of peak

# Setup
run = utils.Run(run_dir)
run.psd_summary = pd.read_pickle(run.psd_file)
# Log output file
log_file = os.path.join(run.summary_dir, 'psd.log')
log = utils.Log(log_file, f'psd.py log file for {run.name}')
# Exact frequency
freq = psd.get_exact_freq(run.psd_summary, freq)
# FFT
rfftfreq, rfft = psd.fft(run, channel, freq, log)
psd = np.absolute(rfft)**2

# Find background mean and variance
b1 = psd[rfftfreq < f_min][-bin_width:]
b2 = psd[rfftfreq > f_max][:bin_width]
b_mean = np.mean(np.concatenate([b1, b2]))
var1 = np.var(b1)
var2 = np.var(b2)
var_mean = np.mean([var1, var2])

# Plot
plt.plot(rfftfreq[1:], np.absolute(rfft)[1:]**2)
#plt.yscale('log')
plt.show()
