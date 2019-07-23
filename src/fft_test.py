import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils

#pd.set_option('display.max_rows', 1000)

summary_file = 'out/drs/run_q/summaries/psd.pkl'
df = pd.read_pickle(summary_file)

# Pick specific frequency and channel
freq = 0.010376
channel = 'x'
values = df.loc[channel].xs(freq, level='FREQ')['MEDIAN']
#values = values[values.notna()]
#values = values.loc[1143962325:]
times = values.index
diffs = np.array([times[i] - times[i-1] for i in range(1, len(times))])
dt = np.mean(diffs[diffs > 1637])
print(dt)
#values = pd.DataFrame({'MEDIAN' : values, 'DT' : diffs}, index=values.index)
#values = values['MEDIAN']
n = values.shape[0]
new_times = np.array([times[0] + i * dt for i in range(n)])
new_values = np.interp(new_times, times, values)
plt.plot(times, values, 'b-*')
plt.plot(new_times, new_values, 'r-*')
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
rfft = np.fft.rfft(values)

print(f'Max frequency: {max(rfftfreq)}')
print(f'1/(2*dt) = {1. / (2 * dt)}')

plt.plot(rfftfreq, abs(rfft))
#plt.plot(rfftfreq, abs(np.fft.rfft(values)), color='r')
plt.title('FFT')
plt.xlabel('f')
plt.ylabel('abs(rfft)')
plt.show()

