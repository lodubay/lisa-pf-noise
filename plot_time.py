import matplotlib.pyplot as plt
import psd

time_dir = 'data/ltp/run_b2/run_b_1143963964/'
df = psd.import_time(time_dir).loc['a_x']
summary = psd.summarize_psd(time_dir).loc['a_x']
freqs = df.index.get_level_values('FREQ')

fig, axs = plt.subplots(1, 2)
for i in range(100):
    axs[0].scatter(freqs, df[i], marker='.', color='b')
axs[0].set_ylim((1e-17, 2e-15))
axs[0].set_xlim((1e-3, 1))
axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[1].plot(freqs, summary['MEDIAN'], color='r')
axs[1].fill_between(freqs, 
    summary['CI_50_LO'], 
    summary['CI_50_HI'],
    color='r', 
    alpha=0.5,
)
axs[1].fill_between(freqs, 
    summary['CI_90_LO'], 
    summary['CI_90_HI'],
    color='r', 
    alpha=0.1,
)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
plt.show()

