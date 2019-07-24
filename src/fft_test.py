import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils

#pd.set_option('display.max_rows', 1000)

summary_file = 'out/drs/run_k/summaries/psd.pkl'
df = pd.read_pickle(summary_file)

# Pick specific frequency and channel
freq = 0.010376
channel = 'x'
values = df.loc[channel].xs(freq, level='FREQ')['MEDIAN']
# Remove NaN values
values = values[values.notna()]
#values = values.loc[1143962325:]
times = values.index
# Find time differences between each observation
diffs = np.array([times[i] - times[i-1] for i in range(1, len(times))])
# Find the mean time difference, excluding outliers
dt = np.mean(diffs[diffs < 1640])
print(f'dt = {dt}')
n = values.shape[0]
# Interpolate data at same dt
new_times = np.array([times[0] + i * dt for i in range(n)])
new_values = np.interp(new_times, times, values)
# Plot interpolated and original data
fig, ax = plt.subplots(1, 1)
ax.plot(times, values, 'b-*', label='Original')
ax.plot(new_times, new_values, 'r-*', label='Interpolated')
ax.set_xlabel('GPS time')
ax.set_ylabel('Power')
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

plt.plot(rfftfreq, abs(rfft))
#plt.plot(rfftfreq, abs(np.fft.rfft(values)), color='r')
plt.title(f'FFT for {run.mode} {run.name} channel {channel}')
plt.xlabel('f')
plt.ylabel('abs(rfft)')
plt.show()

