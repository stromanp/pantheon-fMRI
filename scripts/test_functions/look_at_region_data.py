import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydatabase
import pyclustering


regiondataname1 = r'E:\FM2021data\HCstim_region_data_Dec2023.npy'
regiondata1 = np.load(regiondataname1, allow_pickle=True).flat[0]
region_properties1 = regiondata1['region_properties']
nregions1 = len(region_properties1)
rnamelist1 = [region_properties1[xx]['rname'] for xx in range(nregions1)]
index1 = rnamelist1.index(region_name)
tc1 = region_properties1[index1]['tc']
nruns_per_person1 = region_properties1[index1]['nruns_per_person']
tsize1 = region_properties1[index1]['tsize']
nclusters1, tsize_full1 = np.shape(tc1)
NP1 = len(nruns_per_person1)
nruns_total1 = np.sum(nruns_per_person1)

tc_bssc = np.mean(np.reshape(tc1, (nclusters1, nruns_total1, tsize1)), axis = 1)


# compare with brain data
regiondatanameb = r'D:\Howie_FM2_Brain_Data\HCstim_region_data_Dec2023.npy'
regiondatab = np.load(regiondatanameb, allow_pickle=True).flat[0]
region_propertiesb = regiondatab['region_properties']
nregionsb = len(region_propertiesb)
rnamelistb = [region_propertiesb[xx]['rname'] for xx in range(nregionsb)]
indexb = rnamelistb.index(region_name)
tcb = region_propertiesb[indexb]['tc']
nruns_per_personb = region_propertiesb[indexb]['nruns_per_person']
tsizeb = region_propertiesb[indexb]['tsize']
nclustersb, tsize_fullb = np.shape(tcb)
NPb = len(nruns_per_personb)
nruns_totalb = np.sum(nruns_per_personb)
tsize_fullb = nruns_totalb*tsizeb

tc_brain = np.mean(np.reshape(tcb, (nclustersb, nruns_totalb, tsizeb)), axis = 1)

windownum = 20
for cnum in range(5):
	plt.close(windownum+cnum)
	fig = plt.figure(windownum+cnum)
	ax1 = plt.subplot(211)
	plt.plot(range(tsize1),tc_bssc[cnum,:], '-xr')

	ax1 = plt.subplot(212)
	plt.plot(range(tsizeb),tc_brain[cnum,:], '-or')