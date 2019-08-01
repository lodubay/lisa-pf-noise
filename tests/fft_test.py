import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd

import utils

#pd.set_option('display.max_rows', 1000)

run = utils.Run('data/ltp/run_b')
summary_file = run.psd_file
df = pd.read_pickle(summary_file)

# Pick specific frequency and channel
freq = 0.01038
channel = run.channels[0]
values = df.loc[channel].xs(freq, level='FREQ')['MEDIAN']
# Remove NaN values
values = values[values.notna()]
values = values.loc[1143962325:]
times = values.index.to_numpy()
values = values.to_numpy()

# Find time differences between each observation
diffs = np.array([times[i] - times[i-1] for i in range(1, len(times))])
# Find the mean time difference, excluding outliers
dt = np.mean(diffs[diffs < 1640])
print(f'dt = {dt}')
n = values.shape[0]
# Interpolate data at same dt
n = int((times[-1] - times[0]) / dt)
new_times = np.array([times[0] + i * dt for i in range(n)])

# Plot original
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle(f'Time series and FFT for {run.mode} {run.name} channel {channel}\n{freq} Hz; manual adjustment')
ax1.plot(times, values, marker='.', label='Original')
ax1.set_xlabel('GPS time')
ax1.set_ylabel('Power')
ax1.set_ylim((1e-16, 6e-16))

interp_types = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
for kind in interp_types:
    f = interpolate.interp1d(times, values, kind=kind)
    new_values = f(new_times)
    # Plot time series
    ax1.plot(new_times, new_values, marker='.', label=kind, alpha=0.8)
    
    # FFT
    rfftfreq = np.fft.rfftfreq(n, dt)
    rfft = np.fft.rfft(new_values)
    ax2.plot(rfftfreq, np.absolute(rfft))
    ax2.set_yscale('log')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('FFT')

new_values = np.interp(new_times, times, values)
# Plot interpolated and original data
handles, labels = ax1.get_legend_handles_labels()
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()

'''
n = int(4000)
tmax = int(100)
dt = tmax / n
print(dt)
t = np.append(np.linspace(0, tmax/2-5, n/2), np.linspace(tmax/2+5, tmax, n/2))
print(t)
dt = np.array([t[i] - t[i-1] for i in range(1, len(t))])
print(dt)
dt = np.mean(dt)
values = np.sin(4 * np.pi * t) + 2 * np.random.normal(size=n)
plt.subplot(2, 1, 1)
plt.plot(t, values)
plt.title('noisy sin(t)')
plt.xlabel('t', position=(1, 0))
plt.ylabel('y')
plt.subplot(2, 1, 2)
'''

