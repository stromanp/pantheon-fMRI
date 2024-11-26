import matplotlib
import matplotlib.pyplot as plt
import numpy as np


regiondataname = r'E:/FM2021data/allstim_region_data_allstim_Jan21_2024.npy'
regiondataname = r'E:/FM2021data/allstim_region_data_allstim_Jan21_2024_check.npy'
regiondataname = r'E:/FM2021data/allstim_region_data_allstim_Jan21_2024_check3.npy'
regionname = 'LC'
windownum = 12

regiondata = np.load(regiondataname, allow_pickle=True).flat[0]
region_properties = regiondata['region_properties']
nregions = len(region_properties)
rnamelist = [region_properties[xx]['rname'] for xx in range(nregions)]
nruns_per_person = region_properties[0]['nruns_per_person']
nruns_total = np.sum(nruns_per_person)
tsize = region_properties[0]['tsize']

index = rnamelist.index(regionname)

nc, tsize_total = np.shape(region_properties[index]['tc'])
tsize = (tsize_total/nruns_total).astype(int)
tc = np.reshape(region_properties[index]['tc'],(nc,nruns_total,tsize))
tc_avg = np.mean(tc, axis=1)

cols = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,0,0]])
plt.close(windownum)
fig = plt.figure(windownum)
for cc in range(nc):
	plt.plot(range(tsize),tc_avg[cc,:],linestyle = '-', color = cols[cc,:])