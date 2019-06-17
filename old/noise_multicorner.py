import matplotlib.pyplot as plt
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
for run_dir in run_dirs:
	print('Importing ' + run_dir[:-1] + '...')
	# Change working directory to next run
	os.chdir(top_dir + '/' + run_dir)
	# GPS time
	time = int(run_dir[6:-1])
	# Grab the files with a single-digit index first to sort them correctly
	# Assumes file name format 'psd.dat.#' and 'psd.dat.##'
	psd_files = sorted(glob.glob('psd.dat.[0-9]'))
	psd_files += sorted(glob.glob('psd.dat.[0-9][0-9]'))
	# Import PSD files
	run_data = []
	for i in range(len(psd_files)):
		psd_data = np.loadtxt(psd_files[i])
		rows = psd_data.shape[0] # Number of rows in the psd.dat file
		time_col = np.full((rows, 1), time) # GPS time column
		chain_index = np.full((rows, 1), i) # Chain index column
		psd_data = np.hstack((time_col, chain_index, psd_data))
		run_data.append(psd_data)
	# 2D array: GPS time | chain index | frequency | channel 1 | channel 2 | ...
	run_data = np.concatenate(run_data)					
	all_data.append(np.array(run_data))
# Replace trailing 2s with empty cells
all_data = np.ma.masked_equal(np.concatenate(all_data), 2)
print(all_data)

corner.corner(all_data)
plt.show()
