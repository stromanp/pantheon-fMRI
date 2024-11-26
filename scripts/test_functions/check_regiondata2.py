

import numpy as np
import os
import pandas as pd
import nibabel as nib
import pybasissets
import pydatabase
import GLMfit
from sklearn.cluster import KMeans
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pyclustering


#-----------------------------------------------------

clusterdeffile =r'Y:\PS2023_analysis\HCfast_clusters_v2.npy'
regiondatafile = r'Y:\PS2023_analysis\HCfast_v6new_regiondata.npy'
regiondatafile = r'Y:\PS2023_analysis\HCfast_v3new_regiondata.npy'
regiondatafile = r'Y:\PS2023_analysis\HCfast_highvnew_regiondata.npy'
windownum = 30

networkmodel = r'Y:\PS2023_analysis\network_model_June2023_SAPM.xlsx'

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


# check the data for a selected region/person
region = 'C6RD'
cluster = 2
person = 10
run = 1
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

tcdata = tc[cluster,tt1:tt2]
tcdata_sem = tc_sem[cluster,tt1:tt2]
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
#
# plt.close(windownum)
# fig = plt.figure(windownum)
#
# plt.errorbar(tt, tcdata, tcdata_sem, color = [1,0,0])
# plt.errorbar(tt, tc_original, tcsem_original, color = [0,0,1])


#-----------------------------------------------------
# now check how it is done in pyclustering
#------------------------------------------------------

# the voxels in the regions of interest need to be extracted
filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
nruns_per_person = np.zeros(NP).astype(int)
for nn in range(NP):
	nruns_per_person[nn] = len(filename_list[nn])
nruns_total = np.sum(nruns_per_person)

# load information about the network
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel, exclude_latent = True)

# -------------------replace section of cluster definition function-----------------------
# identify the voxels in the regions of interest
region_properties0 = []
# cluster_properties = []
region_coordinate_list = []
region_start = []
region_end = []
ncluster_list2 = []
vox_count = 0
for nn, rname in enumerate(sem_region_list):
	print('loading data for region {}'.format(rname))
	rname_check = copy.deepcopy(cluster_properties[nn]['rname'])
	regionindex = copy.deepcopy(cluster_properties[nn]['regionindex'])
	regionnumlist = copy.deepcopy(cluster_properties[nn]['regionnum'])
	cx = copy.deepcopy(cluster_properties[nn]['cx'])
	cy = copy.deepcopy(cluster_properties[nn]['cy'])
	cz = copy.deepcopy(cluster_properties[nn]['cz'])
	IDX = copy.deepcopy(cluster_properties[nn]['IDX'])
	nclusters = copy.deepcopy(cluster_properties[nn]['nclusters'])
	print('loading data for {} clusters'.format(nclusters))
	region_coordinate_list.append({'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence' :0, 'IDX' :IDX, 'nclusters' :nclusters})
	region_start += [vox_count]
	region_end += [vox_count +len(cx)]
	vox_count += len(cx)

	if (nn == 0):
		ncluster_list2 = [nclusters]
		cx_all = copy.deepcopy(cx)
		cy_all = copy.deepcopy(cy)
		cz_all = copy.deepcopy(cz)
	else:
		ncluster_list2 += [nclusters]
		cx_all = np.concatenate((cx_all, cx), axis=0)
		cy_all = np.concatenate((cy_all, cy), axis=0)
		cz_all = np.concatenate((cz_all, cz), axis=0)

# -------------------end of part that replaced cluster definition function---------------------

mode = 'concatenate'
allregiondata = pyclustering.load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all)

# for comparing with earlier part------------------
rstart = region_start[regionnum]
rend = region_end[regionnum]
loaded_regiondata = allregiondata[rstart:rend,:]

cx = cluster_properties[regionnum]['cx']
cy = cluster_properties[regionnum]['cy']
cz = cluster_properties[regionnum]['cz']
IDX = cluster_properties[regionnum]['IDX']
cc = np.where(IDX == cluster)[0]

regiondata_1cluster = loaded_regiondata[cc,tt1:tt2]
tc_regiondata = np.mean(regiondata_1cluster,axis=0)
tcsem_regiondata = np.std(regiondata_1cluster,axis=0)/np.sqrt(len(cc))

plt.close(windownum)
fig = plt.figure(windownum)
plt.errorbar(tt,tcdata+0.5,tcdata_sem, color = [1,0,0])
plt.errorbar(tt,tc_original,tcsem_original, color = [0,1,0])
plt.errorbar(tt,tc_regiondata -0.5,tcsem_regiondata, color = [0,0,1])
#--------------------------------------------------







nvox ,ts = np.shape(allregiondata)
print('nvox = {}   ts = {}'.format(nvox ,ts))

region_name_list = [region_coordinate_list[x]['rname'] for x in range(len(region_coordinate_list))]
for nn, rname in enumerate(region_name_list):
	print('loading data from region {}'.format(rname))
	nvox = copy.deepcopy(region_coordinate_list[nn]['nvox'])
	cx = copy.deepcopy(region_coordinate_list[nn]['cx'])
	cy = copy.deepcopy(region_coordinate_list[nn]['cy'])
	cz = copy.deepcopy(region_coordinate_list[nn]['cz'])
	occurrence = copy.deepcopy(region_coordinate_list[nn]['occurrence'])
	nclusters = copy.deepcopy(region_coordinate_list[nn]['nclusters'])
	n1 = copy.deepcopy(region_start[nn])
	n2 = copy.deepcopy(region_end[nn])

	regiondata = copy.deepcopy(allregiondata[n1:n2 ,:])

	# -----------------check for high variance------------
	tsize = int(ts /nruns_total)
	print('ts = {}  nruns_total = {}   tsize = {}'.format(ts ,nruns_total, t s /nruns_total))
	# rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order = 'F').copy()
	rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
	varcheck2 = np.var(rdtemp, axis = 2)
	nvarvalstotal = nvox *nruns_total

	if varcheckmethod == 'median':
		typicalvar2 = np.median(varcheck2)
	else:
		typicalvar2 = np.mean(varcheck2)
	varlimit = varcheckthresh * typicalvar2

	cv ,cp = np.where(varcheck2 > varlimit)  # voxels with high variance
	if len(cv) > 0:
		for vv in range(len(cv)):
			rdtemp[cv[vv] ,cp[vv] ,:] = np.zeros \
				(tsize)   # replace with zeros so the crazy variance does not mess up clustering
		for vv in range(len(cv)):
			meanrun = np.mean(rdtemp[cv[vv], :, :], axis=0)
			rdtemp[cv[vv], cp[vv], :] = copy.deepcopy \
				(meanrun)  # replace with the average now, with the high variance runs zeroed out
		print('---------------------------------!!!!!--------------------------------------');
		print('Variance check found {} voxels with high variance ({:.1f} percent of total)'.format(len(cv), 100. * len
			(cv ) /nvarvalstotal))
		print('---------------------------------!!!!!--------------------------------------\n');
	else:
		print('Variance check did not find any voxels with high variance');

	# regiondata = rdtemp.reshape(nvox, ts, order = 'F').copy()
	regiondata = np.reshape(rdtemp, (nvox, ts))
	# ------------done correcting for crazy variance - -------------------

	# ------------replace in define_cluster_and_load_data function---------------
	IDX = copy.deepcopy(region_coordinate_list[nn]['IDX'])
	# ------------end of replace in define_cluster_and_load_data function--------

	tc = np.zeros([nclusters ,ts])
	tc_sem = np.zeros([nclusters ,ts])
	for aa in range(nclusters):
		# cc = [i for i in range(len(IDX)) if IDX[i] == aa]
		cc = np.where(IDX == aa)[0]
		nvox = len(cc)
		if nvox > 0:
			tc[aa ,:] = np.mean(regiondata[cc, :], axis=0)
			tc_sem[aa ,:] = np.std(regiondata[cc, :], axis=0 ) /np.sqrt(nvox)
		else:
			print('---------------CHECK THIS!-----------------------------')
			print('region {} cluster {} does not contain any data!'.format(rname ,aa))
			print('-------------------------------------------------------')
	tc_original = copy.deepcopy(tc)
	tc_sem_original = copy.deepcopy(tc_sem)


	# handle high variance voxels differently
	# rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
	cv2 ,cp2 = np.where(varcheck2 <= varlimit)  # voxels without high variance
	tcr = np.zeros([nclusters ,nruns_total ,tsize])
	tcr_sem = np.zeros([nclusters ,nruns_total ,tsize])

	for aa in range(nclusters):
		# cc = [i for i in range(len(IDX)) if IDX[i] == aa]
		# cc = np.where(IDX == aa)[0]
		for bb in range(nruns_total):
			# rcheck = np.where(cp2 == bb)[0]   # find the entries for this run
			# vcheck = cv2[rcheck]    # find the good voxels for this run
			# cc2 = [i for i in cc if i in vcheck]
			# cc2r = np.where( (varcheck2[:,bb] <= varlimit) & (IDX == aa))  # good voxels for this run

			cc2 = [xx for xx in range(len(IDX)) if (varcheck2[xx ,bb] <= varlimit) and (IDX[xx] == aa)]
			nvox2 = len(cc2)
			if nvox2 > 0:
				tcr[aa ,bb ,:] = np.mean(rdtemp[cc2, bb, :], axis=0)
				tcr_sem[aa ,:] = np.std(rdtemp[cc2, bb, :], axis=0 ) /np.sqrt(nvox2)
	tc = np.reshape(tcr ,(nclusters ,ts))
	tc_sem = np.reshape(tcr_sem ,(nclusters ,ts))

	clusterdef_entry = {'cx' :cx, 'cy' :cy, 'cz' :cz ,'IDX' :IDX, 'nclusters' :nclusters, 'rname' :rname, 'regionindex' :regionindex, 'regionnum' :regionnum, 'occurrence' :occurrence}
	regiondata_entry = {'tc' :tc, 'tc_sem' :tc_sem, 'tc_original' :tc_original, 'tc_sem_original' :tc_sem_original, 'nruns_per_person' :nruns_per_person, 'tsize' :tsize, 'rname' :rname, 'DBname' :DBname, 'DBnum' :DBnum, 'prefix' :prefix, 'occurrence' :occurrence}
	region_properties.append(regiondata_entry)
	cluster_properties.append(clusterdef_entry)

print('loading cluster data complete.')
