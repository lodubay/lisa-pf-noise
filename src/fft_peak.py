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
bin_width = 20 # number of points on either side of peak to bin
f_min = 8.5e-5 # minimum estimated frequency of peak
f_max = 9e-5 # maximum estimated frequency of peak

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
fft_psd = np.absolute(rfft)**2

# Find background mean and variance
b1 = fft_psd[rfftfreq < f_min]
bin_width = min(bin_width, b1.shape[0]) # make sure start doesn't go below 0
b1 = b1[-bin_width:]
b2 = fft_psd[rfftfreq > f_max][:bin_width]
b_mean = np.mean(np.concatenate([b1, b2]))
b1_var = np.var(b1)
b2_var = np.var(b2)
b_var = np.mean([b1_var, b2_var])
b_std = np.sqrt(b_var)

# Find peak value
peak_val = np.max(fft_psd[(rfftfreq > f_min) & (rfftfreq < f_max)])
significance = (peak_val - b_mean) / b_std
print(significance)

# Plot
plt.plot(rfftfreq[1:], fft_psd[1:])
#plt.yscale('log')
plt.show()
