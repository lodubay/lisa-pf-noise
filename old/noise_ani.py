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

# Set up figure
print('Initializin figure...')
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlim(np.min(run_data[:,:-1,0]), np.max(run_data[:,:-1,0]))
ax.set_ylim(np.min(run_data[:,:-1,1:]), np.max(run_data[:,:-1,1:]))
# Initialize paths
ncols = run_data.shape[-1]
paths = ()
for k in range(1, ncols):
	paths += (ax.plot([],[],'-'),)

def animate(i):
	for k in range(1, ncols):
		# Strip last point, which would throw off the y scale
		print(paths[k])
		paths[k].set_xdata(run_data[i,:-1,0])
		paths[k].set_ydata(run_data[i,:-1,k])
	return paths

ani = animation.FuncAnimation(fig, animate, frames=len(run_data), interval=20)
plt.show()
