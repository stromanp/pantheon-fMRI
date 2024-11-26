import numpy as np
import copy
import matplotlib.pyplot as plt

rname = r'D:\Howie_FM2_Brain_Data\HCstim_region_data.npy'
rname_out = r'D:\Howie_FM2_Brain_Data\HCstim_region_data2.npy'

rdata = np.load(rname, allow_pickle=True).flat[0]

tsize = 135
nvolmask = 3

nregions = len(rdata['region_properties'])

for nn in range(nregions):
	tc = rdata['region_properties'][nn]['tc']
	nclusters, ntp = np.shape(tc)
	nruns = int(ntp/tsize)
	for xx in range(nclusters):
		for rr in range(nruns) :
			tc1 = rr*tsize
			tc2 = (rr+1)*tsize
			temp = tc[xx,tc1:tc2]

			for tt in range(nvolmask): temp[tt] = temp[nvolmask]

			md = np.mean(temp)
			temp -= md

			tc[xx, tc1:tc2] = copy.deepcopy(temp)

	rdata['region_properties'][nn]['tc'] = copy.deepcopy(tc)

np.save(rname_out, rdata)


# look at region properties
rname = r'D:\Howie_FM2_Brain_Data\HCstim_region_data2.npy'

rdata = np.load(rname, allow_pickle=True).flat[0]

NP = len(rdata['region_properties'])
rnum = 12
clusternum = 3
clusternum = 4
windownum = 24

tc = rdata['region_properties'][rnum]['tc'][clusternum,:]

ntp = len(tc)
nruns = int(ntp / tsize)
tc_reshape = np.zeros((nruns,tsize))
for rr in range(nruns):
	tc1 = rr * tsize
	tc2 = (rr + 1) * tsize
	tc_reshape[rr,:] = tc[tc1:tc2]

mtc = np.mean(tc_reshape,axis=0)
stc = np.std(tc_reshape,axis=0)/np.sqrt(nruns)

plt.close(windownum)
fig = plt.figure(windownum)
plt.errorbar(range(tsize), mtc,yerr = stc)