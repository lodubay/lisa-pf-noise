import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import os
import linechain as lc

# `normed` is being deprecated in favor of `density` in histograms
if LooseVersion(matplotlib.__version__) >= '2.1':
    density_param = {'density': True}
else:
    density_param = {'normed': True}
    
#----------------------------------------------------------------------
# Import data
time_dir = 'data/ltp_run_b/run_b_1143768233/'
channel = 4
lc_file = os.path.join(time_dir, 'linechain_channel' + str(channel) + '.dat')
# Import first column to determine how wide DataFrame should be
counts = lc.get_counts(lc_file)
# Get most likely line model (i.e., the number of spectral lines)
model = int(counts.mode())
# Import entire data file, accounting for uneven rows
lc = pd.read_csv(lc_file, header=None, names=range(max(counts)*3+1), sep=' ')
# Strip of all rows that don't match the model
lc = lc[lc.iloc[:,0] == model].dropna(1).reset_index(drop=True).rename_axis('IDX')
# Rearrange DataFrame so same lines are grouped together
# Make 3 column DataFrame by stacking sections vertically
df = pd.concat([
    lc.iloc[:,c*3+1:c*3+4].set_axis(
        ['FREQ','AMP','QF'], axis=1, inplace=False
    ) for c in range(model)
], keys=pd.Series(range(model), name='LINE'))

#----------------------------------------------------------------------
# Kernel density estimation

X = df['FREQ'].to_numpy()[:, np.newaxis]
train_idx = int(len(X) / 2 * 0.25)
mid_idx = int(len(X) / 2)
X_train = np.vstack((X[:train_idx], X[mid_idx:mid_idx+train_idx]))
N = len(X)
X_test = np.vstack((X[train_idx:mid_idx], X[mid_idx+train_idx:]))
X_plot = np.linspace(0, 1, 1638)[:, np.newaxis]

fig, ax = plt.subplots()
ax.hist(X[:, 0], bins=100, range=(0.068, 0.071), density=True)

kde = KernelDensity(kernel='gaussian', bandwidth=1e-4).fit(X_train)
log_dens = kde.score_samples(X_plot)
ax.plot(X_plot[:, 0], np.exp(log_dens), '-', label="kernel = 'gaussian'")

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X_test[:, 0], -10 - 40 * np.random.random(X_test.shape[0]), '+k')
ax.plot(X_train[:, 0], -10 - 40 * np.random.random(X_train.shape[0]), '+r')


ax.set_xlim(0.068, 0.071)
#ax.set_ylim(-0.02, 1)
#ax.set_xscale('log')
plt.show()
