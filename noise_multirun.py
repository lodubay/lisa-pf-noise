import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D # Registers the 3D projection
import corner
import numpy as np
import os
import glob

# The index of the channel we're interested in
channel = 1
# Run directories
top_dir = os.getcwd()
run_dirs = sorted(glob.glob('run_k_*/'))[0:4] # only a few for testing purposes

# Pull PSD files from target run
all_data = []
times = []
for run_dir in run_dirs:
	run = int(run_dir[6:-1])
	times.append(run)
	print('Importing ' + str(run) + '...')
	os.chdir(top_dir + '/' + run_dir)
	# Grab the files with a single-digit index first to sort them correctly
	# Assumes file name format 'psd.dat.#' and 'psd.dat.##'
	psd_files = sorted(glob.glob('psd.dat.[0-9]'))
	psd_files += sorted(glob.glob('psd.dat.[0-9][0-9]'))
	# Import PSD files
	run_data = []
	for psd_file in psd_files:
		run_data.append(np.loadtxt(psd_file))
	# 3D array: (chain index, frequency index, channel)
	# If the number of rows differs from previous runs, adjust
	run_data = np.array(run_data)
	if run_dirs.index(run_dir) > 0:
		difference = all_data[-1].shape[-2] - run_data.shape[-2]
		# Create filler array assigned value 2, which will later be masked
		# Filler array shape: (# of psds, difference between arrays, # columns in run_data)
		filler_array = np.full((run_data.shape[0], np.abs(difference), run_data.shape[2]), 2)
		# If it's smaller, add extra rows:
		if difference > 0:
			run_data = np.hstack((run_data, filler_array))
		# If it's larger, add extra rows to all previous runs:
		elif difference < 0:
			all_data = [np.hstack((data, filler_array)) for data in all_data]					
	all_data.append(np.array(run_data))
# 4D array: (run index, chain index, frequency index, channel)
# Replace trailing 2s with empty cells
all_data = np.ma.masked_equal(np.array(all_data), 2)

# Set up figure
print('Initializing figure...')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
# Axes limits
all_freq = all_data[:,:,:,0].flatten()
all_channel = all_data[:,:,:,channel].flatten()
ax.set_xlim(np.min(all_freq), np.max(all_freq))
ax.set_ylim(np.min(all_channel), np.max(all_channel))
for run in range(len(all_data)):
	plt.scatter(all_data[run,:,:,0], all_data[run,:,:,channel], marker='.', s=1)
plt.show()

# 3D figure
print('Initializing 3D figure...')
fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
times_long = np.array(sorted(times * all_data.shape[1] * all_data.shape[2]))
ax.scatter(all_data[:,:,:,0], times_long, all_data[:,:,:,channel], marker='.', s=1)
ax.set_xlabel('Frequency')
ax.set_ylabel('GPS time')
ax.set_zlabel('Channel ' + str(channel) + ' noise')
plt.show()
