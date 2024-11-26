# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pydatabase
import nibabel as nib

clusterdeffile =r'Y:\PS2023_analysis\HCfast_clusters_v2.npy'
regiondatafile = r'Y:\PS2023_analysis\HCfast_v6new_regiondata.npy'
regiondatafile = r'Y:\PS2023_analysis\HCfast_v3new_regiondata.npy'
regiondatafile = r'Y:\PS2023_analysis\HCfast_highvnew_regiondata.npy'
windownum = 30

clusterdata = np.load(clusterdeffile, allow_pickle=True).flat[0]
cluster_properties = clusterdata['cluster_properties']
regiondata = np.load(regiondatafile, allow_pickle=True).flat[0]
region_properties = regiondata['region_properties']

DBname = regiondata['DBname']
DBname = r'Y:\Copy_of_PS2023\PS2023_database_corrected.xlsx'   # override this
DBnum = regiondata['DBnum']

# clusterdef_entry = {'cx' :cx, 'cy' :cy, 'cz' :cz ,'IDX' :IDX, 'nclusters' :nclusters, 'rname' :rname, 'regionindex' :regionindex, 'regionnum' :regionnum, 'occurrence' :occurrence}
# regiondata_entry = {'tc' :tc, 'tc_sem' :tc_sem, 'tc_original' :tc_original, 'tc_sem_original' :tc_sem_original, 'nruns_per_person' :nruns_per_person, 'tsize' :tsize, 'rname' :rname, 'DBname' :DBname, 'DBnum' :DBnum, 'prefix' :prefix, 'occurrence' :occurrence}

# check the data for a selected region/person
region = 'PAG'
cluster = 1
person = 5
run = 2
prefix = 'xptc'

tsize = region_properties[0]['tsize']
nruns_per_person = region_properties[0]['nruns_per_person']

# get the region/cluster data
namelist = [region_properties[xx]['rname'] for xx in range(len(region_properties))]
regionnum = namelist.index(region)
tc = region_properties[regionnum]['tc']
tc_sem = region_properties[regionnum]['tc_sem']
runindex = np.sum(nruns_per_person[:person]) + run
tt1 = runindex*tsize
tt2 = (runindex + 1)*tsize

tcdata = tc[cluster,tc1:tc2]
tcdata_sem = tc_sem[cluster,tc1:tc2]
tt = np.array(list(range(tsize)))


cx = cluster_properties[regionnum]['cx']
cy = cluster_properties[regionnum]['cy']
cz = cluster_properties[regionnum]['cz']
IDX = cluster_properties[regionnum]['IDX']
cc = np.where(IDX == cluster)[0]

# get the NIfTI data
datanamelist, dbnumlist, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix)
namelist = [name for name in datanamelist.keys()]

name = namelist[person]
dbnums = dbnumlist[name]
dbnum = dbnums[run]
fname = datanamelist[name][run]   # name of the nifti file for this particular person/run

input_img = nib.load(fname)
input_data = input_img.get_fdata()

xs,ys,zs,ts = np.shape(input_data)
tc_original = np.zeros(ts)
tcsem_original = np.zeros(ts)

voxeldata = np.zeros((len(cc),tsize))
# for ttt in range(ts):
# 	voxeldata[:,ttt] = input_data[cx[cc],cy[cc],cz[cc],ttt]

voxeldata = input_data[cx[cc],cy[cc],cz[cc],:]

# mask out the initial volumes, if wanted
nvolmask = 2
if nvolmask > 0:
	for ttt in range(nvolmask): voxeldata[:, ttt] = voxeldata[:, nvolmask]

# # convert to signal change from the average----------------
# if data have been cleaned they are already percent signal changes
mean_data = np.mean(voxeldata, axis=1)
mean_data = np.repeat(mean_data[:, np.newaxis], tsize, axis=1)
voxeldata = voxeldata - mean_data

varcheck = np.var(voxeldata, axis = 1)
mvar = np.median(varcheck)

mcheck = np.where(varcheck < 5.0*mvar)[0]
tc_original = np.mean(voxeldata[mcheck,:], axis = 0)
tcsem_original = np.std(voxeldata[mcheck,:], axis = 0)/np.sqrt(len(mcheck))

plt.close(windownum)
fig = plt.figure(windownum)

plt.errorbar(tt, tcdata, tcdata_sem, color = [1,0,0])
plt.errorbar(tt, tc_original, tcsem_original, color = [0,0,1])