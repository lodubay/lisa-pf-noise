import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd

import utils

#pd.set_option('display.max_rows', 1000)

run = utils.Run('data/drs/run_b')
summary_file = run.psd_file
df = pd.read_pickle(summary_file)

# Pick specific frequency and channel
freq = 0.01038
channel = run.channels[0]
values = df.loc[channel].xs(freq, level='FREQ')['MEDIAN']
# Remove NaN values
values = values[values.notna()]
#values = values.loc[1143962325:]
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
fig, ax = plt.subplots(1, 1)
ax.plot(times, values, marker='.', label='Original')
ax.set_xlabel('GPS time')
ax.set_ylabel('Power')

interp_types = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
for kind in interp_types:
    f = interpolate.interp1d(times, values, kind=kind)
    new_values = f(new_times)
    ax.plot(new_times, new_values, marker='.', label=kind, alpha=0.8)

new_values = np.interp(new_times, times, values)
# Plot interpolated and original data
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels)
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

rfftfreq = np.fft.rfftfreq(n, dt)
rfft = np.fft.rfft(new_values)

print(f'Max frequency: {max(rfftfreq)}')
print(f'1/(2*dt) = {1. / (2 * dt)}')

plt.plot(rfftfreq, np.absolute(rfft))
#plt.plot(rfftfreq, abs(np.fft.rfft(values)), color='r')
plt.title(f'FFT for {run.mode} {run.name} channel {channel}')
plt.xlabel('f')
plt.ylabel('abs(rfft)')
plt.yscale('log')
plt.xlim((2e-6, 3e-4))
plt.show()

