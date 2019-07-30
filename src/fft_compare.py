import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.ticker as tkr

import psd
import utils

channel = 'y'

ltp = utils.Run('data/ltp/run_b')
ltpsum = pd.read_pickle(ltp.psd_file)
ltp.psd_summary = ltpsum[ltpsum.index.get_level_values('TIME') >= 1143962325]

drs = utils.Run('data/drs/run_b')
drs.psd_summary = pd.read_pickle(drs.psd_file)

runs = [ltp, drs]

frequencies = np.array([1e-3, 5e-3, 3e-2])
frequencies = psd.get_exact_freq(ltp.psd_summary, frequencies)

nrows = len(frequencies)
ncols = 2
fig, axes = plt.subplots(nrows, ncols, sharex='all')
fig.suptitle(f'Channel {channel}\nFFT of power at selected frequencies',
        fontsize=18)

for r, run in enumerate(runs):
    for i, freq in enumerate(frequencies):
        ax = axes[i, r]
        ax.set_title(f'%s mHz' % float('%.3g' % (freq * 1000.)), fontsize=16)
        rfftfreq, rfft = psd.fft(run, channel, [freq])
        ax.plot(rfftfreq, np.absolute(rfft)**2, color='#0077c8')
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(tkr.NullLocator())
        if r == 0:
            ax.set_ylabel('PSD', fontsize=14)
        if i+1 == len(frequencies):
            ax.set_xlabel('Frequency (Hz)', fontsize=14)

plt.show()
    

def fft(rfftfreq, rfft, run, channel, frequencies, 
        plot_file=None, show=False, logfreq=True):
    # Automatically create grid of axes
    nrows = int(np.floor(float(len(frequencies)) ** 0.5))
    ncols = int(np.ceil(1. * len(frequencies) / nrows))
    # Set up figure
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    fig.suptitle(f'{run.mode.upper()} channel {channel}', fontsize=fig_title_size)
    
    # Subplots
    for i, freq in enumerate(frequencies):
        ax = fig.add_subplot(nrows, ncols, i+1)
        psd_vals = np.absolute(rfft[i])**2
        ax.plot(rfftfreq[i], psd_vals, color='#0077c8')
        # Axis title
        ax.title.set_text(f'FFT of power at %s mHz' % float('%.3g' % (freq * 1000.)))
        # Vertical axis label on first plot in each row
        if i % ncols == 0:
            ax.set_ylabel('PSD', fontsize=ax_label_size)
        # Horizontal axis label on bottom plot in each column
        if i >= len(frequencies) - ncols:
            ax.set_xlabel('Frequency (Hz)', fontsize=ax_label_size)
        if logfreq: 
            ax.set_xscale('log')
        else:
            # Minor ticks
            ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
        ax.set_yscale('log')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    if plot_file: plt.savefig(plot_file, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
