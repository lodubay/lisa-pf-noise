import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    
    # Iterate through rows and sort values to correct columns
    for c in range(freqs.shape[1] - 1):
        f = freqs[:,c]
        for i in range(len(f)):
            # Find difference between this value and each mode
            difs = np.abs(f[i] - modes)
            # If the closest mode isn't in this column, swap with next value
            if difs[c] != np.min(difs):
                freqs[i,c], freqs[i,c+1] = freqs[i,c+1], freqs[i,c]
    
    param_stack = np.vstack([params[:,3*c:3*c+3] for c in range(most_freq_dim)])
    #f = param_stack[:,0]
    f = params[:,6]
    amp = param_stack[:,1]
    Q = param_stack[:,2]
    index=np.arange(len(f))


    #mid_pt=(min(f)+max(f))/2
    #mid_pt = np.mean(f)
    #df = np.std(f)
    df = 1e-3
    hist, bin_edges = np.histogram(f, bins=int((np.max(f)-np.min(f))/(2*df)))
    #mode = stats.mode(f)[0]
    hist_max = np.where(hist == np.max(hist))[0][0]
    print(hist_max)
    mode = np.mean(bin_edges[hist_max:hist_max+2])
    print(mode)
    lower = mode - df
    upper = mode + df
    mids = np.quantile(f, [1./3., 2./3.])

    #f_1=f[f<mid_pt]
    #f_2=f[f>=mid_pt]
    f_1 = f[f<mids[0]]
    f_2 = f[(f>=mids[0]) & (f<mids[1])]
    f_3 = f[f>=mids[1]]
    f_prime = f[(f>lower) & (f<upper)]
     
    #amp_1=amp[f<mid_pt]
    #amp_2=amp[f>=mid_pt]

    #q_1=Q[f<mid_pt]
    #q_2=Q[f>=mid_pt]

    #plt.scatter(f_1,q_1, marker='.', alpha=0.2, s=1)
    #plt.scatter(f_2,q_2, marker='.', alpha=0.2, s=1)
    #plt.hist(f_1, range=(0,0.1), bins=1000)
    #plt.hist(f_2, range=(0,0.1), bins=1000)
    plt.scatter(f_1, np.random.random(len(f_1)), marker='.', alpha=0.2, s=1)
    plt.scatter(f_2, np.random.random(len(f_2)), marker='.', alpha=0.2, s=1)
    plt.scatter(f_3, np.random.random(len(f_3)), marker='.', alpha=0.2, s=1)
    #plt.axvline(mids[0], c='r')
    #plt.axvline(mids[1], c='purple')
    plt.axvline(mode, c='g')
    plt.axvline(lower, c='purple')
    plt.axvline(upper, c='purple')
    plt.xlim(0,1)
    #plt.xscale('log')

    #plt.scatter(f_1,q_1)
    #plt.scatter(f_2,q_2)
    #plt.show()
