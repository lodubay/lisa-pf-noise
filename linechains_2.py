import numpy as np
import matplotlib.pyplot as plt
import itertools

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

time_dir = './data/drsCrutch_run_q/run_q_1176962857/linechain_channel'

#Only looking at CHANNEL 4! **********************************************
#for k in range(0,6,1):
for k in range(4,5,1):
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

    most_freq_dim=int(find_nearest(g,max(g[1:])))
    print(most_freq_dim)
    
    # Import all rows with dim == most_freq_dim
    with open(time_dir+str(k)+'.dat') as fp:
        params = [l.split() for l in fp if int(l.split()[0]) == most_freq_dim]
        params = np.asarray(params, dtype='float64')[:,1:]
    
    # Make an array of just frequencies - for testing purposes
    freqs = np.hstack([params[:,3*c:3*c+1] for c in range(most_freq_dim)])
    
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
    #for c in range(freqs.shape[1] - 1):
    #    f = freqs[:,c]
    for i, row in enumerate(freqs):
        # Compute row permutations
        perm = np.array(list(itertools.permutations(row)))
        # Calculated the distances between each permutation and the modes
        dist = np.abs(perm - modes)
        # Compute the total distance magnitudes
        sums = np.sqrt(np.sum(dist ** 2, axis=1))
        # Use permutation that minimizes total distance
        freqs[i] = perm[sums == np.min(sums)][0]
    
    print(freqs)
    medians = np.median(freqs, axis=0)
    print(medians)
    
    # Plot
    colors = ['r', 'g', 'b']
    #for i in range(len(medians)):
    #    plt.scatter(freqs[:,i], np.random.random(freqs.shape[0]), 
    #        marker='.', alpha=0.2, s=1, c=colors[i]
    #    )
        #plt.axvline(medians[i], c=colors[i])
        
    plt.scatter(freqs[:,0], np.random.random(freqs.shape[0]), 
        marker='.', alpha=0.2, s=1, c=colors[0]
    )
    
    #plt.xlim(0.25,0.28)
    plt.xlim(0,1)
    #plt.xscale('log')
    plt.show()
