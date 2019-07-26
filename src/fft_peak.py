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
bin_width = 8 # number of points on either side of peak to bin
f_step = 5e-6 # amount to step between each check
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

# Number of peak bins to check
n_bins = int((rfftfreq[-bin_width] - rfftfreq[bin_width]) / f_step)
# Cycle through frequency space looking for peaks
f_peaks = []
peaks = []
sigs = []
for i in range(n_bins):
    # Define minimum and maximum peak frequencies
    f_min = rfftfreq[bin_width] + i * f_step
    f_max = f_min + f_step
    
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
    peak_psd = fft_psd[(rfftfreq > f_min) & (rfftfreq < f_max)]
    peak_range = rfftfreq[(rfftfreq > f_min) & (rfftfreq < f_max)]
    print(peak_range)
    peak_val = np.max(peak_psd)
    f_peak = peak_range[np.argmax(peak_psd)]
    significance = (peak_val - b_mean) / b_std
    print(f'{f_peak}: {peak_val}, significance {significance} sigma')
    print(f'Background level: {b_mean}\n')
    if significance >= 3:
        f_peaks.append(f_peak)
        peaks.append(peak_val)
        sigs.append(significance)

# Significance table
sig_df = pd.DataFrame({'FREQ': f_peaks, 'POWER': peaks, 'SIG': sigs})
print(sig_df)

# Plot
plt.plot(rfftfreq[1:], fft_psd[1:])
plt.scatter(f_peaks, peaks, c='r', marker='.')
plt.ylabel('PSD')
plt.xlabel('Frequency (Hz)')
plt.title(f'Bin width = {bin_width}')
#plt.yscale('log')
plt.show()

