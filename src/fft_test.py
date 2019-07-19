import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils

pd.set_option('display.max_rows', 1000)

summary_file = 'out/ltp/run_b/summaries/psd.pkl'
df = pd.read_pickle(summary_file)

freq = 0.003052
channel = 'x'
values = df.loc[channel].xs(freq, level='FREQ')['MEDIAN']
values = values[values.notna()]
times = values.index
diffs = [times[i] - times[i-1] for i in range(1, len(times))] + [0]
dt = np.median(diffs)
values = pd.DataFrame({'MEDIAN' : values, 'DT' : diffs}, index=values.index)

values = values.loc[1143962325:1144058991]['MEDIAN']
#print(values)
n = values.shape[0]

'''
n = 40000
tmax = 1000
dt = tmax / n
t = np.linspace(0, tmax, n)
values = np.sin(4 * np.pi * t) + np.random.normal(size=n)
'''

rfftfreq = np.fft.rfftfreq(n, dt)
rfft = np.fft.rfft(values)

print(max(rfftfreq))
print(1. / (2 * dt))

plt.plot(rfftfreq, abs(rfft))
plt.show()

'''
n = 40000
tmax = 1000
dt = tmax / n
t = np.linspace(0, tmax, n)
y = np.sin(4 * np.pi * t) + np.random.normal(size=n)

plt.scatter(t, y)
plt.show()

rfftfreq = np.fft.rfftfreq(n, dt)
rfft = np.fft.rfft(y)

print(rfft.shape)
print(dt)
print(np.max(rfftfreq))

plt.plot(rfftfreq, abs(rfft))
plt.xlim(0, 4)
plt.show()
'''
