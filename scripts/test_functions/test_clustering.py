# testing clustering methods

import pyclustering
import numpy as np
import load_templates
import os
import pandas as pd
import nibabel as nib
import pybasissets
import pydatabase
import GLMfit
from sklearn.cluster import KMeans
import copy


# modified clustering method - for roughly equal size clusters
# Thanks to Eyal Shulman who shared on StackOverflow  https://stackoverflow.com/users/6247548/eyal-shulman
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

DBname = r'E:\graded_pain_database_May2022.xlsx'
listname = r'E:\Pain_filtered_list.npy'
dbnumlist = np.load(listname,allow_pickle=True).flat[0]
DBnum = dbnumlist['dbnumlist']
prefix = 'xptc'
networkmodel = r'E:\SAPMresults_Dec2022\network_model_Jan2023.xlsx'

template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
	load_templates.load_template_and_masks('ccbs', 1)



#---------------------------------------------------------------------------------------------------
# pyclustering.define_clusters_and_load_data(DBname, DBnum, prefix, networkmodel, regionmap_img, anatlabels)

# the voxels in the regions of interest need to be extracted
filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
nruns_per_person = np.zeros(NP).astype(int)
for nn in range(NP):
	nruns_per_person[nn] = len(filename_list[nn])
nruns_total = np.sum(nruns_per_person)

# load information about the network
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)

anatnamelist = []
for name in anatlabels['names']:
	anatnamelist.append(name)

# identify the voxels in the regions of interest
region_properties = []
cluster_properties = []
region_coordinate_list = []
region_start = []
region_end = []
ncluster_list2 = []
vox_count = 0
for nn, rname in enumerate(sem_region_list):
	# regionindex = anatnamelist.index(rname)
	regionindex = [x for x, name in enumerate(anatnamelist) if name == rname]
	regionnum = anatlabels['numbers'][regionindex]
	print('searching for region {} {}'.format(rname, regionnum))
	if len(regionnum) > 1:
		# if the number of clusters divides evenly into the number of regions
		# then split the clusters amongst the regions (to maintain R/L divisions for example)
		# otherwise, put all the regions together
		clusters_per_region = ncluster_list[nn]['nclusters'] / len(regionnum)
		if clusters_per_region == np.floor(clusters_per_region):
			for aa, rr in enumerate(regionnum):
				cx, cy, cz = np.where(regionmap_img == rr)
				region_start += [vox_count]
				region_end += [vox_count + len(cx)]
				vox_count += len(cx)
				region_coordinate_list.append(
					{'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence': aa})
				if (nn == 0) and (aa == 0):
					ncluster_list2 = [np.ceil(ncluster_list[nn]['nclusters'] / len(regionnum)).astype(int)]
					cx_all = cx
					cy_all = cy
					cz_all = cz
				else:
					ncluster_list2 += [np.ceil(ncluster_list[nn]['nclusters'] / len(regionnum)).astype(int)]
					cx_all = np.concatenate((cx_all, cx), axis=0)
					cy_all = np.concatenate((cy_all, cy), axis=0)
					cz_all = np.concatenate((cz_all, cz), axis=0)
		else:
			for aa, rr in enumerate(regionnum):
				cx, cy, cz = np.where(regionmap_img == rr)
				if aa == 0:
					cx_full = cx
					cy_full = cy
					cz_full = cz
				else:
					cx_full = np.concatenate((cx_full, cx), axis=0)
					cy_full = np.concatenate((cy_full, cy), axis=0)
					cz_full = np.concatenate((cz_full, cz), axis=0)

			print('shape of cx_full is {}'.format(np.shape(cx_full)))
			cx = np.unique(cx_full)
			cy = np.unique(cy_full)
			cz = np.unique(cz_full)

			cx = copy.deepcopy(cx_full)
			cy = copy.deepcopy(cy_full)
			cz = copy.deepcopy(cz_full)
			print('shape of cx after unique function is {}'.format(np.shape(cx)))

			region_start += [vox_count]
			region_end += [vox_count + len(cx)]
			vox_count += len(cx)
			region_coordinate_list.append(
				{'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence': 0})

			if nn == 0:
				ncluster_list2 = [ncluster_list[nn]['nclusters']]
				cx_all = cx
				cy_all = cy
				cz_all = cz
			else:
				ncluster_list2 += [ncluster_list[nn]['nclusters']]
				cx_all = np.concatenate((cx_all, cx), axis=0)
				cy_all = np.concatenate((cy_all, cy), axis=0)
				cz_all = np.concatenate((cz_all, cz), axis=0)
	else:
		cx, cy, cz = np.where(regionmap_img == regionnum.values[0])

		region_start += [vox_count]
		region_end += [vox_count + len(cx)]
		vox_count += len(cx)
		region_coordinate_list.append({'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence': 0})
		if nn == 0:
			ncluster_list2 = [ncluster_list[nn]['nclusters']]
			cx_all = cx
			cy_all = cy
			cz_all = cz
		else:
			ncluster_list2 += [ncluster_list[nn]['nclusters']]
			cx_all = np.concatenate((cx_all, cx), axis=0)
			cy_all = np.concatenate((cy_all, cy), axis=0)
			cz_all = np.concatenate((cz_all, cz), axis=0)

# regiondata = group_data[cx,cy,cz,:]   # nvox x tsize
# load the data one region at a time to save memory - necessary for large data sets
mode = 'concatenate'
nvolmask = 2
allregiondata = pyclustering.load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all)
nvox, ts = np.shape(allregiondata)

region_name_list = [region_coordinate_list[x]['rname'] for x in range(len(region_coordinate_list))]
for nn, rname in enumerate(region_name_list):
	nvox = region_coordinate_list[nn]['nvox']
	cx = region_coordinate_list[nn]['cx']
	cy = region_coordinate_list[nn]['cy']
	cz = region_coordinate_list[nn]['cz']
	occurrence = region_coordinate_list[nn]['occurrence']
	n1 = region_start[nn]
	n2 = region_end[nn]

	regiondata = allregiondata[n1:n2, :]

	# -----------------check for extreme variance------------
	tsize = int(ts / nruns_total)
	rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order='F').copy()
	varcheck2 = np.var(rdtemp, axis=1)
	typicalvar2 = np.mean(varcheck2)
	varlimit = 5.0 * typicalvar2
	cv, cp = np.where(varcheck2 > varlimit)  # voxels with crazy variance
	if len(cv) > 0:
		for vv in range(len(cv)):
			rdtemp[cv[vv], :, cp[vv]] = np.zeros(tsize)
		print('---------------!!!!!----------------------');
		print('Variance check found {} crazy voxels'.format(len(cv)))
		print('---------------!!!!!----------------------\n');
	else:
		print('Variance check did not find any crazy voxels');
	regiondata = rdtemp.reshape(nvox, ts, order='F').copy()
	# ------------done correcting for crazy variance - -------------------

	# now do the clustering for this region
	varcheck = np.var(regiondata, axis=1)
	# cvox = np.where(varcheck > 0) # exclude voxels with effectively constant values
	cvox = [i for i in range(len(varcheck)) if varcheck[i] > 0]
	print('using {} voxels of {} with non-zero variance for defining clusters'.format(len(cvox), nvox))
	if len(cvox) > 0:
		regiondata = regiondata[cvox, :]
		cx = cx[cvox]
		cy = cy[cvox]
		cz = cz[cvox]



	#-----------------------------------------------------------------------
	# divide each region into N clusters with similar timecourse properties
	nclusters = ncluster_list2[nn]
	kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(regiondata)
	IDX = kmeans.labels_
	cluster_tc = kmeans.cluster_centers_

	nvoxels, tsizefull = np.shape(regiondata)
	cluster_size = np.floor(nvoxels/nclusters).astype(int)

	# from Eyal Shulman
	centers = kmeans.cluster_centers_
	centers = centers.reshape(-1, 1, regiondata.shape[-1]).repeat(cluster_size, 1).reshape(-1, regiondata.shape[-1])
	distance_matrix = cdist(regiondata, centers)
	IDX = linear_sum_assignment(distance_matrix)[1] // cluster_size


	tc = np.zeros([nclusters, ts])
	tc_sem = np.zeros([nclusters, ts])
	for aa in range(nclusters):
		cc = [i for i in range(len(IDX)) if IDX[i] == aa]
		nvox = len(cc)
		tc[aa, :] = np.mean(regiondata[cc, :], axis=0)
		tc_sem[aa, :] = np.std(regiondata[cc, :], axis=0) / np.sqrt(nvox)

	clusterdef_entry = {'cx': cx, 'cy': cy, 'cz': cz, 'IDX': IDX, 'nclusters': nclusters, 'rname': rname,
						'regionindex': regionindex, 'regionnum': regionnum, 'occurrence': occurrence}
	regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,
						'rname': rname, 'DBname': DBname, 'DBnum': DBnum, 'prefix': prefix, 'occurrence': occurrence}
	region_properties.append(regiondata_entry)
	cluster_properties.append(clusterdef_entry)

# combine repeated occurrences, if they occur
occurrences = [cluster_properties[x]['occurrence'] for x in range(len(cluster_properties))]
if (np.array(occurrences) > 0).any():
	rnamelist = [cluster_properties[x]['rname'] for x in range(len(cluster_properties))]
	cluster_properties2 = []
	region_properties2 = []
	for nn, rname in enumerate(sem_region_list):
		cr = [x for x in range(len(rnamelist)) if rnamelist[x] == rname]
		if len(cr) > 1:
			ncluster_total = 0
			for aa, cc in enumerate(cr):
				if aa == 0:
					cx = cluster_properties[cc]['cx']
					cy = cluster_properties[cc]['cy']
					cz = cluster_properties[cc]['cz']
					IDX = cluster_properties[cc]['IDX']
					nclusters = cluster_properties[cc]['nclusters']
					rname = cluster_properties[cc]['rname']
					regionindex = cluster_properties[cc]['regionindex']
					regionnum = cluster_properties[cc]['regionnum']
					tc = region_properties[cc]['tc']
					tc_sem = region_properties[cc]['tc_sem']
					nruns_per_person = region_properties[cc]['nruns_per_person']
					tsize = region_properties[cc]['tsize']
					DBname = region_properties[cc]['DBname']
					DBnum = region_properties[cc]['DBnum']
					prefix = region_properties[cc]['prefix']
					ncluster_total += nclusters
				else:
					cx2 = cluster_properties[cc]['cx']
					cy2 = cluster_properties[cc]['cy']
					cz2 = cluster_properties[cc]['cz']
					IDX2 = cluster_properties[cc]['IDX']
					nclusters2 = cluster_properties[cc]['nclusters']
					tc2 = region_properties[cc]['tc']
					tc_sem2 = region_properties[cc]['tc_sem']

					cx = np.concatenate((cx, cx2), axis=0)
					cy = np.concatenate((cy, cy2), axis=0)
					cz = np.concatenate((cz, cz2), axis=0)
					IDX = np.concatenate((IDX, IDX2 + ncluster_total), axis=0)
					ncluster_total += nclusters2
					tc = np.concatenate((tc, tc2), axis=0)
					tc_sem = np.concatenate((tc, tc_sem2), axis=0)

			clusterdef_entry_temp = {'cx': cx, 'cy': cy, 'cz': cz, 'IDX': IDX, 'nclusters': ncluster_total,
									 'rname': rname, 'regionindex': regionindex, 'regionnum': regionnum}
			regiondata_entry_temp = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,
									 'rname': rname, 'DBname': DBname, 'DBnum': DBnum, 'prefix': prefix}

			cluster_properties2.append(clusterdef_entry_temp)
			region_properties2.append(regiondata_entry_temp)
		else:
			cluster_properties2.append(cluster_properties[cr[0]])
			region_properties2.append(region_properties[cr[0]])

	cluster_properties = cluster_properties2
	region_properties = region_properties2

print('cluster definition complete.')
# return cluster_properties, region_properties