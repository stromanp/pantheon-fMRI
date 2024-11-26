# need to convert B values betweeen Sinput that have been scaled to have normalized
# variance, and Sinput that have not been scaled.
import numpy as np
import matplotlib.pyplot as plt
import copy

resultsname = r'E:\SAPMresults_Dec2022\AllPain_original_3242423012_results.npy'
resultsname_norm = r'E:\SAPMresults_Dec2022\AllPain_3242423012_results.npy'

paramsname = r'E:\SAPMresults_Dec2022\AllPain_original_3242423012_params.npy'

results = np.load(resultsname, allow_pickle=True)
results_norm = np.load(resultsname_norm, allow_pickle=True)

params = np.load(paramsname, allow_pickle=True).flat[0]

nn = 1

Sinput = results[nn]['Sinput']
Sconn = results[nn]['Sconn']
Mconn = results[nn]['Mconn']
betavals = results[nn]['betavals']
Minput = results[nn]['Minput']

Sinput_norm = results_norm[nn]['Sinput']
Sconn_norm = results_norm[nn]['Sconn']
Mconn_norm = results_norm[nn]['Mconn']
betavals_norm = results_norm[nn]['betavals']
Minput_norm = results_norm[nn]['Minput']

plt.close(60)
fig = plt.figure(60)
plt.plot(betavals, betavals_norm,'og')

nr,tsize = np.shape(Sinput)
V = np.zeros((nr,nr))
vlist = np.zeros(nr)
for rr in range(nr):
	vv = np.std(Sinput[rr,:])
	V[rr,rr] = 1.0/(vv + 1.0e-20)
	vlist[rr] = copy.deepcopy(vv)

# G = np.linalg.inv(Minput @ Minput.T) @ Minput.T @ V @ Minput
#   ==> problem:  Minput @ Minput.T is singular

nr, N = np.shape(Minput)
M = Minput.T @ Minput
Minv = sortof_inverse(M, order = 'invM M')
G = Minv @ Minput.T @ V @ Minput
GG = (G.T @ G)
GGinv = sortof_inverse(GG, order = 'invG G')
Mconn2 = GGinv @ G.T @ Mconn_norm @ G
# this doesnt work



windownum = 10
plt.close(windownum)
fig = plt.figure(windownum)
plt.plot(betavals,betavals_norm,'ob')





nregions = len(params['cluster_properties'])
countlist = np.zeros(nregions)
for rr in range(nregions):
	cx = params['cluster_properties'][rr]['cx']
	IDX = params['cluster_properties'][rr]['IDX']
	nclusters = params['cluster_properties'][rr]['nclusters']
	rname = params['cluster_properties'][rr]['rname']
	vcount = np.zeros(nclusters)
	for rn in range(nclusters):
		c = np.where(IDX == rn)[0]
		vcount[rn] = len(c)
	countlist[rr] = np.mean(vcount)
	print('{} voxels in each cluster: {},  std: {:.3f}'.format(rname, vcount,vlist[rr]))


plt.close(70)
fig = plt.figure(70)
plt.plot(np.sqrt(countlist),1.0/vlist,'og')