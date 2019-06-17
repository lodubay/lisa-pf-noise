import matplotlib.pyplot as plt
import matplotlib.animation as ani
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
print(run_data)

# Set up figure
fig = plt.figure(1)
ax = fig.add_subplot(111)
path, = ax.plot([],[],'-')

def animate(i):
	path.set_xdata(run_data[i,:,0])
	path.set_ydata(run_data[i,:,1])
	return path

	
fig1 = plt.figure(1)

filename = 'run_k_1159724317/psd.dat.0'
data = np.loadtxt(filename)
freq = data[:-1,0]

plt.title(filename)
for i in range(1,data.shape[1]):
	plt.plot(freq, data[:-1,i])

plt.yscale('log')
plt.xlabel('frequency')
plt.ylabel('noise')
plt.show()

filename = 'run_k_1159724317/psd.dat.1'
data = np.loadtxt(filename)
freq = data[:-1,0]

plt.title(filename)
for i in range(1,data.shape[1]):
	plt.plot(freq, data[:-1,i])

plt.yscale('log')
plt.xlabel('frequency')
plt.ylabel('noise')
plt.show()
