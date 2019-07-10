import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#time = 1144203931
#time_dir = './data/ltp_run_b/run_b_' + str(time) + '/linechain_channel'
# Has 3 spectral lines
time = 1176962857
time_dir = './data/drsCrutch_run_q/run_q_' + str(time) + '/linechain_channel'

#Only looking at CHANNEL 4! **********************************************
for k in range(6):
#for k in range(4,5):
    print('\n-- CHANNEL ' + str(k) + ' --')
    dim=[]
    with open(time_dir+str(k)+'.dat') as fp:
        for line in fp:
            z=line.split()
            dim.append(int(z[0]))
    dim=np.asarray(dim)
    dim_u=np.arange(np.max(dim)+1)
    g=[]
    for i in dim_u:
        g.append(len(dim[dim==i]))
    print(g)

    most_freq_dim=int(find_nearest(g, max(g)))
    print(most_freq_dim)
    
    if most_freq_dim == 0:
        print('No spectral lines found, skipping...')
    else:
        # Import all rows with dim == most_freq_dim
        #most_freq_dim = 2
        with open(time_dir+str(k)+'.dat') as fp:
            params = [l.split() for l in fp if int(l.split()[0]) == most_freq_dim]
        
        # Configure array
        params = np.asarray(params, dtype='float64')[:,1:]
        
        # Make an array of just frequencies - for testing purposes
        freqs = np.hstack([params[:,3*c:3*c+1] for c in range(most_freq_dim)])
        freqs.sort(axis=1)
        
        # Calculate modes for each column
        df = 1e-3
        modes = []
        for c in range(freqs.shape[1]):
            f = freqs[:,c]
            hist, bin_edges = np.histogram(f, bins=int((np.max(f)-np.min(f))/(2*df)))
            hist_max = np.where(hist == np.max(hist))[0][0]
            mode = np.mean(bin_edges[hist_max:hist_max+2])
            modes.append(mode)
        modes = np.array(modes)
        
        print(modes)
        print(np.median(freqs, axis=0))
        
        # Iterate through rows and sort values to correct columns
        if most_freq_dim > 1:
            for i, row in enumerate(freqs):
                # Compute row permutations
                perm = np.array(list(itertools.permutations(row)))
                # Calculate the distances between each permutation and the modes
                dist = np.abs(perm - modes) / modes
                # Compute the total distance magnitudes
                # Inverting to lessen penalty for one value that doesn't match
                sums = np.sum(dist ** -1, axis=1) ** -1
                # Use permutation that minimizes total distance
                freqs[i] = perm[sums == np.min(sums)][0]
        
        # Summary statistics
        medians = np.median(freqs, axis=0)
        percentiles = np.array([5, 25, 50, 75, 95])
        f_summary = np.percentile(freqs, percentiles, axis=0)
        midx = pd.MultiIndex.from_product(
            [[k], [time], list(range(most_freq_dim))],
            names=['CHANNEL', 'TIME', 'LINE']
        )
        f_summary = pd.DataFrame(
            f_summary.T, 
            columns=pd.Series(percentiles.astype('str'), name='PERCENTILE'), 
            index=midx
        )
        print(f_summary)
        
        # Plot
        colors = ['r', 'g', 'b']
        for i in range(len(medians)):
        #for i in [2]:
            plt.scatter(freqs[:,i], np.random.random(freqs.shape[0]), 
                marker='.', alpha=0.2, s=1, c=colors[i]
            )
            plt.axvline(medians[i], c=colors[i])
        
        #plt.xlim(0.25,0.28)
        plt.xlim(1e-3,1)
        #plt.xscale('log')
        plt.show()
