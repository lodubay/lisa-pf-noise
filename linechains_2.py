import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#Only looking at CHANNEL 4! **********************************************
#for k in range(0,6,1):
for k in range(4,5,1):
    dim=[]
    with open('./data/drsCrutch_run_q/run_q_1176962857/linechain_channel'+str(k)+'.dat') as fp:
        for line in fp:
            z=line.split()
            dim.append(int(z[0]))
    dim=np.asarray(dim)
    dim_u=np.unique(dim)
    g=[]
    for i in dim_u:
        g.append(len(dim[dim==i]))
    print(g)

    most_freq_dim=int(find_nearest(g,max(g[1:])))
    print(most_freq_dim)

    f=[]
    amp=[]
    Q=[]
    #most_freq_dim=3
    with open('./data/drsCrutch_run_q/run_q_1176962857/linechain_channel'+str(k)+'.dat') as fp:
        #cnt = 1
        params = [l.split() for l in fp if int(l.split()[0]) == most_freq_dim]
        params = np.asarray(params).astype('float64')[:,1:]
        print(params)
        #for line in fp:
        #    z=line.split()
        #    print(z[0])
        #    if float(z[0]) == most_freq_dim:
        #        f.append(float(z[1]))
        #        #f.append(float(z[4]))
        #        #f.append(float(z[7]))
        #        amp.append(float(z[2]))
        #        #amp.append(float(z[5]))
        #        #amp.append(float(z[8]))
        #        Q.append(float(z[3]))
        #        #Q.append(float(z[6]))
        #        #Q.append(float(z[9]))
            #cnt += 1
    #f=np.asarray(f)
    #amp=np.asarray(amp)
    #Q=np.asarray(Q)
    f = params[:,0]
    amp = params[:,1]
    Q = params[:,2]
    index=np.arange(len(f))


    #mid_pt=(min(f)+max(f))/2
    mid_pt = np.mean(f)
    mode = stats.mode(f)
    print(mode)

    f_1=f[f<mid_pt]
    f_2=f[f>mid_pt]
     
    amp_1=amp[f<mid_pt]
    amp_2=amp[f>mid_pt]

    q_1=Q[f<mid_pt]
    q_2=Q[f>mid_pt]

    #plt.scatter(f_1,q_1, marker='.', alpha=0.2, s=1)
    #plt.scatter(f_2,q_2, marker='.', alpha=0.2, s=1)
    #plt.hist(f_1, range=(0,0.1), bins=1000)
    #plt.hist(f_2, range=(0,0.1), bins=1000)
    plt.scatter(f_1, np.random.random(len(f_1)), marker='.', alpha=0.2, s=1)
    plt.scatter(f_2, np.random.random(len(f_2)), marker='.', alpha=0.2, s=1)
    #plt.axvline(mid_pt, c='r')
    #plt.axvline(mode[0], c='g')
    #plt.axvline(mode[0] + np.std(f), c='purple')
    plt.xlim(1e-3, 1)
    plt.xscale('log')

    #plt.scatter(f_1,q_1)
    #plt.scatter(f_2,q_2)
    plt.show()
