import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

time = 1144203931
time_dir = './data/ltp_run_b/run_b_' + str(time) + '/linechain_channel'
# Has 3 spectral lines
#time = 1176962857
#time_dir = './data/drsCrutch_run_q/run_q_' + str(time) + '/linechain_channel'

#Only looking at CHANNEL 4! **********************************************
for channel in range(6):
#for channel in range(4,5):
    print('\n-- CHANNEL ' + str(channel) + ' --')
    lc_file = time_dir + str(channel) + '.dat'
    # Find the most frequently sampled line model number
    with open(lc_file) as lc:
        # Only use the first column
        dim = np.array([int(line[0]) for line in lc])
    
    # List of unique model numbers through max from lc file
    dim_u = np.arange(np.max(dim)+1)
    # Count how often each model is used
    counts = np.array([len(dim[dim == m]) for m in dim_u])
    print('Line model histogram:')
    print(counts)
    # Get the most common model
    model = counts.argmax()
    print(str(model) + ' spectral lines found.')
    
    if model > 0:
        # Import all rows with dim == model
        with open(lc_file) as lc:
            lines = [line.split() for line in lc if int(line[0]) == model]
        
        # Configure array
        line_array = np.asarray(lines, dtype='float64')[:,1:]
        # Create 3D array with index order [index, line, parameter]
        params = []
        for p in range(3):
            param = [line_array[:,3*c+p:3*c+p+1] for c in range(model)]
            params.append(np.hstack(param))
        params = np.dstack(params)
        
        # Calculate modes for each column
        # This should give a rough value for the location of each spectral line
        modes = []
        for c in range(params.shape[1]):
            f = params[:,c,0]
            hist, bin_edges = np.histogram(f, bins=2000)
            hist_max = hist.argmax()
            mode = np.mean(bin_edges[hist_max:hist_max+2])
            modes.append(mode)
        modes = np.sort(np.array(modes))
        # For debugging
        print('Spectral line modal frequencies: ')
        print(modes)
        
        # Iterate through rows and sort values to correct columns
        if model > 1:
            for i, row in enumerate(params):
                # Compute permutations of all frequencies
                f = row[:,0]
                perm = np.array(list(itertools.permutations(f)))
                # Permutations of indices
                idx = np.array(list(itertools.permutations(range(len(f)))))
                # Calculate the distances between each permutation and the modes
                dist = np.abs(perm - modes) / modes
                # Compute the total distance magnitudes
                # Inverting to lessen penalty for one value that doesn't match
                sums = np.sum(dist ** -1, axis=1) ** -1
                # Use permutation that minimizes total distance
                min_idx = idx[sums.argmin()]
                params[i] = row[min_idx]
        
        # Summary statistics
        percentiles = np.array([5, 25, 50, 75, 95])
        parameters = ['FREQ', 'AMP', 'QF']
        summary = np.percentile(params, percentiles, axis=0)
        # Transpose to index as [line, param, index]
        summary = np.transpose(summary, axes=(1,2,0))
        midx = pd.MultiIndex.from_product(
            [[channel], [time], list(range(model)), parameters],
            names=['CHANNEL', 'TIME', 'LINE', 'PARAMETER']
        )
        summary = pd.DataFrame(
            np.vstack(summary), 
            columns=pd.Series(percentiles.astype('str'), name='PERCENTILE'), 
            index=midx
        )
        print(summary)
        
        # Plot
        colors = ['r', 'g', 'b']
        medians = summary.xs('FREQ', level='PARAMETER')['50']
        for i in range(model):
        #for i in [2]:
            plt.scatter(params[:,i,0], np.random.random(params.shape[0]), 
                marker='.', alpha=0.2, s=1, c=colors[i]
            )
            plt.axvline(medians[i], c=colors[i])
        
        plt.xlim(1e-3,1)
        plt.xscale('log')
        plt.show()

