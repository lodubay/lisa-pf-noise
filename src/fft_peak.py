import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psd
import utils

# Plot parameters
run_dir = 'data/drs/run_k'
run = utils.Run(run_dir)
channel = run.channels[1]
freq = [1e-2]

# Analysis parameters
bin_width = 10 # number of points on either side of peak to bin
f_step = 5e-6 # amount to step between each check
min_sig = 3 # minimum significance for peak identification

# Setup
run.psd_summary = pd.read_pickle(run.psd_file)
# Log output file
log = utils.Log(run.fft_log, f'{run.mode.upper()} {run.name}')
log.log(f'Channel {channel}, %s mHz' % float('%.3g' % (freq[0] * 1000.)))
log.log(f'Bin width = {bin_width}, peak width = {f_step}, minimum significance = {min_sig}\n')
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
background = []
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
    peak_val = np.max(peak_psd)
    f_peak = peak_range[np.argmax(peak_psd)]
    significance = (peak_val - b_mean) / b_std
    log.log(f'{f_peak}: {peak_val}, significance {significance} sigma')
    log.log(f'Background level: {b_mean}\n')
    if significance >= min_sig:
        f_peaks.append(f_peak)
        peaks.append(peak_val)
        sigs.append(significance)
        background.append(b_mean)

# Significance table
f_peaks = np.array(f_peaks)
sig_df = pd.DataFrame({'FREQ': f_peaks, 'PERIOD': 1/f_peaks, 'POWER': peaks, 
        'SIG': sigs, 'BACKGROUND': background})
print(sig_df)
log.log(sig_df.to_string())

# Plot
plt.plot(rfftfreq[1:], fft_psd[1:])
plt.scatter(f_peaks, peaks, c='r', marker='.')
plt.ylabel('PSD')
plt.xlabel('Frequency (Hz)')
plt.title(f'{run.mode.upper()} channel {channel}\nBin width = {bin_width}')
#plt.yscale('log')
plt.show()

