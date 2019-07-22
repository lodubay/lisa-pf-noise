import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils

pd.set_option('display.max_rows', 1000)

summary_file = 'out/drs/run_k/summaries/psd.pkl'
df = pd.read_pickle(summary_file)

'''
freq = 0.003052
channel = 'x'
values = df.loc[channel].xs(freq, level='FREQ')['MEDIAN']
values = values[values.notna()]
times = values.index
diffs = [times[i] - times[i-1] for i in range(1, len(times))] + [0]
dt = np.mean(diffs)
values = pd.DataFrame({'MEDIAN' : values, 'DT' : diffs}, index=values.index)

values = values['MEDIAN']
#values = values.loc[1143962325:1144058991]['MEDIAN']
n = values.shape[0]

'''
n = 4000
tmax = 100
dt = tmax / n
t = np.linspace(0, tmax, n)
values = np.sin(4 * np.pi * t) + 2 * np.random.normal(size=n)
plt.subplot(2, 1, 1)
plt.plot(t, values)
plt.title('noisy sin(t)')
plt.xlabel('t', position=(1, 0))
plt.ylabel('y')
plt.subplot(2, 1, 2)


rfftfreq = np.fft.rfftfreq(n, dt)
rfft = np.fft.rfft(values)

print(f'Max frequency: {max(rfftfreq)}')
print(f'1/(2*dt) = {1. / (2 * dt)}')

plt.plot(rfftfreq, abs(rfft))
plt.title('FFT')
plt.xlabel('f')
plt.ylabel('abs(rfft)')
plt.show()

