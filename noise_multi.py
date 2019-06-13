import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import glob

# Pull PSD files from target run
print('Importing data...')
run = 1159724317
run_directory = 'run_k_' + str(run)
os.chdir(run_directory)
# Grab the files with a single-digit index first to sort them correctly
# Assumes file name format 'psd.dat.#' and 'psd.dat.##'
psd_files = sorted(glob.glob('psd.dat.[0-9]'))
psd_files += sorted(glob.glob('psd.dat.[0-9][0-9]'))
# Import PSD files
run_data = []
for psd_file in psd_files:
	run_data.append(np.loadtxt(psd_file))
run_data = np.array(run_data)
# Contacenate all data into single 2D array
combined = np.concatenate(run_data)

# Set up figure
print('Initializing figure...')
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlim(np.min(combined[:-1,0]), np.max(combined[:-1,0]))
ax.set_ylim(np.min(combined[:-1,1:]), np.max(combined[:-1,1:]))
ncols = combined.shape[-1]
colors = ['red','orange','yellow','green','blue','purple']
for k in range(1, ncols):
	plt.scatter(combined[:-1,0], combined[:-1,k], color=colors[k-1], marker='.')
	#for i in range(run_data.shape[0]):
	#	plt.plot(run_data[i,:-1,0], run_data[i,:-1,k], color=colors[k-1])
plt.yscale('log')
plt.show()
