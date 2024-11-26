import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyclustering
import load_templates
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# def define_clusters_and_load_data(DBname, DBnum, prefix, nvolmask, networkmodel, regionmap_img, anatlabels,
# 								  varcheckmethod='median', varcheckthresh=3.0):

DBname = r'E:\FM2021data\FMS2_database_July27_2022b.xlsx'
DBnum = [10, 12, 13, 16, 17]   # HC get the data for the first person in the set
prefix = 'xptc'
networkmodel = r'E:\FM2021data\network_model_June2023_SAPM.xlsx'

normtemplatename = 'ccbs'
resolution = 1
template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
	load_templates.load_template_and_masks(normtemplatename, resolution)


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

# the voxels in the regions of interest need to be extracted
filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
nruns_per_person = np.zeros(NP).astype(int)
for nn in range(NP):
	nruns_per_person[nn] = len(filename_list[nn])
nruns_total = np.sum(nruns_per_person)

# load information about the network
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel, exclude_latent=True)

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
	regionnum = copy.deepcopy(anatlabels['numbers'][regionindex])
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

			cx = copy.deepcopy(cx_full)
			cy = copy.deepcopy(cy_full)
			cz = copy.deepcopy(cz_full)

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

# regiondata = group_data[cx,cy,cz,:]   # nvox x tsize
# load all the data from all regions, all data sets
mode = 'concatenate'
nvolmask = 2
allregiondata = pyclustering.load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all)
nvox, ts = np.shape(allregiondata)

region_name_list = [region_coordinate_list[x]['rname'] for x in range(len(region_coordinate_list))]
for nn, rname in enumerate(region_name_list):
	print('loading data from region {}'.format(rname))
	nvox = region_coordinate_list[nn]['nvox']
	cx = region_coordinate_list[nn]['cx']
	cy = region_coordinate_list[nn]['cy']
	cz = region_coordinate_list[nn]['cz']
	occurrence = region_coordinate_list[nn]['occurrence']
	n1 = region_start[nn]
	n2 = region_end[nn]

	regiondata = allregiondata[n1:n2, :]

	# -----------------check for high variance------------
	tsize = int(ts / nruns_total)
	# rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order = 'F').copy()
	rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
	varcheck2 = np.var(rdtemp, axis=2)
	nvarvalstotal = nvox * nruns_total

	if varcheckmethod == 'median':
		typicalvar2 = np.median(varcheck2)
	else:
		typicalvar2 = np.mean(varcheck2)
	varlimit = varcheckthresh * typicalvar2

	cv, cp = np.where(varcheck2 > varlimit)  # voxels with high variance
	high_var_record = {'cv': cv, 'cp': cp}
	if len(cv) > 0:
		for vv in range(len(cv)):
			rdtemp[cv[vv], cp[vv], :] = np.zeros(tsize)  # replace with zeros so the high variance does not mess up clustering
		for vv in range(len(cv)):
			meanrun = np.mean(rdtemp[cv[vv], :, :], axis=0)
			rdtemp[cv[vv], cp[vv], :] = copy.deepcopy(meanrun)  # replace with the average now, with the high variance runs zeroed out
		print('---------------------------------!!!!!--------------------------------------')
		print('Variance check found {} voxels with high variance ({:.1f} percent of total)'.format(len(cv), 100. * len(cv) / nvarvalstotal))
		print('---------------------------------!!!!!--------------------------------------\n')
	else:
		print('Variance check did not find any voxels with high variance')
	# regiondata = rdtemp.reshape(nvox, ts, order = 'F').copy()
	regiondata = np.reshape(rdtemp, (nvox, ts))
	# ------------done correcting for crazy variance - -------------------

	# now do the clustering for this region
	# -----------------remove this part--------------------------
	# varcheck = np.var(regiondata, axis = 1)
	# # cvox = np.where(varcheck > 0) # exclude voxels with effectively constant values
	# cvox = [i for i in range(len(varcheck)) if varcheck[i] > 0]
	# print('using {} voxels of {} with non-zero variance for defining clusters'.format(len(cvox), nvox))
	# if len(cvox)>0:
	#     regiondata = regiondata[cvox, :]
	#     cx = cx[cvox]
	#     cy = cy[cvox]
	#     cz = cz[cvox]
	# -----------------------------------------------------------

	# divide each region into N clusters with similar timecourse properties
	nclusters = ncluster_list2[nn]
	kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(regiondata)
	IDX = kmeans.labels_
	cluster_tc = kmeans.cluster_centers_

	make_equal_size_clusters = True
	if make_equal_size_clusters:
		# modified clustering method - for roughly equal size clusters
		# Thanks to Eyal Shulman who shared on StackOverflow  https://stackoverflow.com/users/6247548/eyal-shulman
		# method for making clusters approximately equal size
		print('identifying approximately equal sized clusters...')
		nvoxels, tsizefull = np.shape(regiondata)
		cluster_size = np.floor(nvoxels / nclusters).astype(int)
		nvox_trunc = cluster_size * nclusters
		centers = kmeans.cluster_centers_
		centers = centers.reshape(-1, 1, regiondata.shape[-1]).repeat(cluster_size, 1).reshape(-1,
																							   regiondata.shape[-1])
		distance_matrix = cdist(regiondata[:nvox_trunc, :], centers)
		val = linear_sum_assignment(distance_matrix)
		IDX = val[1] // cluster_size

		# add in remaining voxels to the nearest clusters
		nresidual = nvoxels - nvox_trunc
		IDXresidual = []
		for xx in range(nresidual):
			tc = regiondata[-xx, :]
			dist = [np.sqrt(np.sum(tc - kmeans.cluster_centers_[dd, :]) ** 2) for dd in range(nclusters)]
			IDXresidual += [np.argmin(dist)]
		IDX = np.concatenate((IDX, np.array(IDXresidual[::-1])), axis=0)

	tc = np.zeros([nclusters, ts])
	tc_sem = np.zeros([nclusters, ts])
	for aa in range(nclusters):
		cc = [i for i in range(len(IDX)) if IDX[i] == aa]
		nvox = len(cc)
		tc[aa, :] = np.mean(regiondata[cc, :], axis=0)
		tc_sem[aa, :] = np.std(regiondata[cc, :], axis=0) / np.sqrt(nvox)
	tc_original = copy.deepcopy(tc)
	tc_sem_original = copy.deepcopy(tc_sem)

	# handle high variance voxels differently
	# rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
	cv2, cp2 = np.where(varcheck2 <= varlimit)  # voxels without high variance
	tcr = np.zeros([nclusters, nruns_total, tsize])
	tcr_sem = np.zeros([nclusters, nruns_total, tsize])
	for aa in range(nclusters):
		# cc = [i for i in range(len(IDX)) if IDX[i] == aa]
		cc = np.where(IDX == aa)[0]
		for bb in range(nruns_total):
			rcheck = np.where(cp2 == bb)[0]  # find the entries for this run
			vcheck = cv2[rcheck]  # find the good voxels for this run
			cc2 = [i for i in cc if i in vcheck]
			nvox2 = len(cc2)
			if nvox2 > 0:
				tcr[aa, bb, :] = np.mean(rdtemp[cc2, bb, :], axis=0)
				tcr_sem[aa, :] = np.std(rdtemp[cc2, bb, :], axis=0) / np.sqrt(nvox2)
	tc = np.reshape(tcr, (nclusters, ts))
	tc_sem = np.reshape(tcr_sem, (nclusters, ts))

	clusterdef_entry = {'cx': cx, 'cy': cy, 'cz': cz, 'IDX': IDX, 'nclusters': nclusters, 'rname': rname,
						'regionindex': regionindex, 'regionnum': regionnum, 'occurrence': occurrence}
	regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'tc_original': tc_original, 'tc_sem_original': tc_sem_original,
						'nruns_per_person': nruns_per_person, 'tsize': tsize, 'rname': rname, 'DBname': DBname,
						'DBnum': DBnum, 'prefix': prefix, 'occurrence': occurrence}
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
			regiondata_entry_temp = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person,
									 'tsize': tsize, 'rname': rname, 'DBname': DBname, 'DBnum': DBnum,
									 'prefix': prefix}

			cluster_properties2.append(clusterdef_entry_temp)
			region_properties2.append(regiondata_entry_temp)
		else:
			cluster_properties2.append(cluster_properties[cr[0]])
			region_properties2.append(region_properties[cr[0]])

	cluster_properties = cluster_properties2
	region_properties = region_properties2

print('cluster definition complete.')
return cluster_properties, region_properties
