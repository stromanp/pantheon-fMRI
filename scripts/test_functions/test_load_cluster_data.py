# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\venv'])

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
import matplotlib
import matplotlib.pyplot as plt
import load_templates


def test_load_cluster_data_methods():
	clusterdeffile = r'Y:\PS2023_analysis\HCfast_clusters_v2.npy'
	regiondatafile = r'Y:\PS2023_analysis\HCfast_v3new_regiondata.npy'
	regiondatafile = r'Y:\PS2023_analysis\HCfast_highvnew_regiondata.npy'
	windownum = 30

	networkmodel = r'Y:\PS2023_analysis\network_model_June2023_SAPM.xlsx'

	clusterdata = np.load(clusterdeffile, allow_pickle=True).flat[0]
	cluster_properties = clusterdata['cluster_properties']
	regiondata = np.load(regiondatafile, allow_pickle=True).flat[0]
	region_properties = regiondata['region_properties']

	normtemplatename = 'ccbs'
	resolution = 1
	template_img, regionmap_img, template_affine, anatlabels = load_templates.load_template(normtemplatename, resolution)

	DBname = regiondata['DBname']
	DBname = r'Y:\Copy_of_PS2023\PS2023_database_corrected.xlsx'  # override this
	DBnum = regiondata['DBnum']

	DBnum = DBnum[:15]

	prefix = 'xptc'
	nvolmask = 2
	varcheckmethod = 'median'
	varcheckthresh = 3.0

	# first test the actual functions
	varcheckthresh = 2.0
	cluster_properties_test, region_properties_test = pyclustering.define_clusters_and_load_data(DBname, DBnum, prefix, nvolmask, networkmodel, regionmap_img, anatlabels, varcheckmethod = 'median', varcheckthresh = varcheckthresh)

	region_properties_new = pyclustering.load_cluster_data_new(cluster_properties_test, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = varcheckthresh)
	region_properties_original = pyclustering.load_cluster_data(cluster_properties_test, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = varcheckthresh)


	#----------------------------------------------------------------
	# original method------------------------------------------------
	#----------------------------------------------------------------
	# the voxels in the regions of interest need to be extracted
	filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list',
																			  separate_conditions=True)
	nruns_per_person = np.zeros(NP).astype(int)
	for nn in range(NP):
		nruns_per_person[nn] = len(filename_list[nn])
	nruns_total = np.sum(nruns_per_person)

	# load information about the network
	network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel, exclude_latent=True)

	# identify the voxels in the regions of interest
	region_properties = []
	temp_data_original = []
	for nn, rname in enumerate(sem_region_list):
		print('loading data for region {}'.format(rname))
		rname_check = cluster_properties[nn]['rname']
		regionindex = cluster_properties[nn]['regionindex']
		regionnum = cluster_properties[nn]['regionnum']
		cx = cluster_properties[nn]['cx']
		cy = cluster_properties[nn]['cy']
		cz = cluster_properties[nn]['cz']
		IDX = cluster_properties[nn]['IDX']
		nclusters = cluster_properties[nn]['nclusters']


		# load the data one region at a time to save memory - necessary for large data sets
		mode = 'concatenate'
		# nvolmask = 2  # changed this to be an input parameter
		regiondata = pyclustering.load_data_from_region(filename_list, nvolmask, mode, cx, cy, cz)
		nvox, ts = np.shape(regiondata)

		# regiondata = group_data[cx, cy, cz, :]  # nvox x tsize

		# -----------------check for extreme variance------------
		tsize = int(ts / nruns_total)
		nvox = len(cx)
		rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order='F').copy()
		varcheck2 = np.var(rdtemp, axis=1)

		# options for identifying voxels with inordinately high variance
		# mean or median?
		# multiple of typical variance?
		# make these inputs?
		if varcheckmethod == 'median':
			typicalvar2 = np.median(varcheck2)
		else:
			typicalvar2 = np.mean(varcheck2)
		varlimit = varcheckthresh * typicalvar2

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

		tc = np.zeros([nclusters, ts])
		tc_sem = np.zeros([nclusters, ts])
		for aa in range(nclusters):
			# cc = [i for i in range(len(IDX)) if IDX[i] == aa]
			cc = np.where(IDX == aa)[0]
			nvox = len(cc)
			tc[aa, :] = np.mean(regiondata[cc, :], axis=0)
			tc_sem[aa, :] = np.std(regiondata[cc, :], axis=0) / np.sqrt(nvox)

		tc_original = copy.deepcopy(tc)
		tcsem_original = copy.deepcopy(tc_sem)

		temp_data_original.append({'tc':tc, 'tc_sem':tc_sem})

	#----------------------------------------------------------------
	# new method ----------------------------------------------------
	#----------------------------------------------------------------

	# the voxels in the regions of interest need to be extracted
	filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
	nruns_per_person = np.zeros(NP).astype(int)
	for nn in range(NP):
		nruns_per_person[nn] = len(filename_list[nn])
	nruns_total = np.sum(nruns_per_person)

	# load information about the network
	network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel, exclude_latent=True)

	# -------------------replace section of cluster definition function-----------------------
	# identify the voxels in the regions of interest
	region_properties = []
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
		regionnum = copy.deepcopy(cluster_properties[nn]['regionnum'])
		cx = copy.deepcopy(cluster_properties[nn]['cx'])
		cy = copy.deepcopy(cluster_properties[nn]['cy'])
		cz = copy.deepcopy(cluster_properties[nn]['cz'])
		IDX = copy.deepcopy(cluster_properties[nn]['IDX'])
		nclusters = copy.deepcopy(cluster_properties[nn]['nclusters'])
		print('loading data for {} clusters'.format(nclusters))
		region_coordinate_list.append(
			{'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence': 0, 'IDX': IDX,
			 'nclusters': nclusters})
		region_start += [vox_count]
		region_end += [vox_count + len(cx)]
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
	nvox, ts = np.shape(allregiondata)
	print('nvox = {}   ts = {}'.format(nvox, ts))

	region_name_list = [region_coordinate_list[x]['rname'] for x in range(len(region_coordinate_list))]
	temp_data_new = []
	temp_data_new2 = []
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

		regiondata = copy.deepcopy(allregiondata[n1:n2, :])

		# -----------------check for high variance------------
		tsize = int(ts / nruns_total)
		print('ts = {}  nruns_total = {}   tsize = {}'.format(ts, nruns_total, ts / nruns_total))
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
		if len(cv) > 0:
			for vv in range(len(cv)):
				rdtemp[cv[vv], cp[vv], :] = np.zeros(
					tsize)  # replace with zeros so the crazy variance does not mess up clustering
			for vv in range(len(cv)):
				meanrun = np.mean(rdtemp[cv[vv], :, :], axis=0)
				rdtemp[cv[vv], cp[vv], :] = copy.deepcopy(
					meanrun)  # replace with the average now, with the high variance runs zeroed out
			print('---------------------------------!!!!!--------------------------------------');
			print('Variance check found {} voxels with high variance ({:.1f} percent of total)'.format(len(cv),
																									   100. * len(
																										   cv) / nvarvalstotal))
			print('---------------------------------!!!!!--------------------------------------\n');
		else:
			print('Variance check did not find any voxels with high variance');

		# regiondata = rdtemp.reshape(nvox, ts, order = 'F').copy()
		regiondata = np.reshape(rdtemp, (nvox, ts))
		# ------------done correcting for crazy variance - -------------------

		# ------------replace in define_cluster_and_load_data function---------------
		IDX = copy.deepcopy(region_coordinate_list[nn]['IDX'])
		# ------------end of replace in define_cluster_and_load_data function--------

		tc = np.zeros([nclusters, ts])
		tc_sem = np.zeros([nclusters, ts])
		for aa in range(nclusters):
			# cc = [i for i in range(len(IDX)) if IDX[i] == aa]
			cc = np.where(IDX == aa)[0]
			nvox = len(cc)
			if nvox > 0:
				tc[aa, :] = np.mean(regiondata[cc, :], axis=0)
				tc_sem[aa, :] = np.std(regiondata[cc, :], axis=0) / np.sqrt(nvox)
			else:
				print('---------------CHECK THIS!-----------------------------')
				print('region {} cluster {} does not contain any data!'.format(rname, aa))
				print('-------------------------------------------------------')
		tc_new = copy.deepcopy(tc)
		tc_sem_new = copy.deepcopy(tc_sem)

		temp_data_new.append({'tc':tc, 'tc_sem':tc_sem})

		# handle high variance voxels differently
		# rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
		cv2, cp2 = np.where(varcheck2 <= varlimit)  # voxels without high variance
		tcr = np.zeros([nclusters, nruns_total, tsize])
		tcr_sem = np.zeros([nclusters, nruns_total, tsize])

		for aa in range(nclusters):
			# cc = [i for i in range(len(IDX)) if IDX[i] == aa]
			# cc = np.where(IDX == aa)[0]
			for bb in range(nruns_total):
				# rcheck = np.where(cp2 == bb)[0]   # find the entries for this run
				# vcheck = cv2[rcheck]    # find the good voxels for this run
				# cc2 = [i for i in cc if i in vcheck]
				# cc2r = np.where( (varcheck2[:,bb] <= varlimit) & (IDX == aa))  # good voxels for this run

				cc2 = [xx for xx in range(len(IDX)) if (varcheck2[xx, bb] <= varlimit) and (IDX[xx] == aa)]
				nvox2 = len(cc2)
				if nvox2 > 0:
					tcr[aa, bb, :] = np.mean(rdtemp[cc2, bb, :], axis=0)
					tcr_sem[aa, :] = np.std(rdtemp[cc2, bb, :], axis=0) / np.sqrt(nvox2)
		tc = np.reshape(tcr, (nclusters, ts))
		tc_sem = np.reshape(tcr_sem, (nclusters, ts))

		temp_data_new2.append({'tc':tc, 'tc_sem':tc_sem})



# def load_cluster_data     (cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
def load_cluster_data_new(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
    # NEW VERSION FOR TESTING
    '''
    Function to load data from a group, based on previous cluster definitions.
    define_clusters_and_load_data in pyclustering.py
    region_properties = load_cluster_data(DBname, DBnum, prefix, networkmodel)
    :param DBname:  name of the database file (probably an excel file)
    :param DBnum:   list of the database entry numbers to use
    :param prefix:   prefix of the nifti format image files to read (indicates the preprocessing
                        steps that have been applied)
    :param networkmodel:  the network definition file name (probably an excel file)
    :return:  output is region_properties
            region_properties is an array of dictionaries (one entry per cluster),
                        with keys: tc, tc_sem, nruns_per_person, and tsize
                        tc and tc_sem are the average and standard error of the time-course for each cluster
                        nruns_per_person lists the number of data sets that are concatenated per person
                        tsize is the number of time points in each individual fMRI run
    This new version (Jan 2024) is designed to run faster and is based on the "define_clusters_and_load_data" function.
    '''

    # the voxels in the regions of interest need to be extracted
    filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
    nruns_per_person = np.zeros(NP).astype(int)
    for nn in range(NP):
        nruns_per_person[nn] = len(filename_list[nn])
    nruns_total = np.sum(nruns_per_person)

    # load information about the network
    network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel, exclude_latent = True)

    #-------------------replace section of cluster definition function-----------------------
    # identify the voxels in the regions of interest
    region_properties = []
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
        regionnum = copy.deepcopy(cluster_properties[nn]['regionnum'])
        cx = copy.deepcopy(cluster_properties[nn]['cx'])
        cy = copy.deepcopy(cluster_properties[nn]['cy'])
        cz = copy.deepcopy(cluster_properties[nn]['cz'])
        IDX = copy.deepcopy(cluster_properties[nn]['IDX'])
        nclusters = copy.deepcopy(cluster_properties[nn]['nclusters'])
        print('loading data for {} clusters'.format(nclusters))
        region_coordinate_list.append({'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence':0, 'IDX':IDX, 'nclusters':nclusters})
        region_start += [vox_count]
        region_end += [vox_count+len(cx)]
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
    nvox,ts = np.shape(allregiondata)
    print('nvox = {}   ts = {}'.format(nvox,ts))

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

        regiondata = copy.deepcopy(allregiondata[n1:n2,:])

        #-----------------check for high variance------------
        tsize = int(ts/nruns_total)
        print('ts = {}  nruns_total = {}   tsize = {}'.format(ts,nruns_total, ts/nruns_total))
        # rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order = 'F').copy()
        rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
        varcheck2 = np.var(rdtemp, axis = 2)
        nvarvalstotal = nvox*nruns_total

        if varcheckmethod == 'median':
            typicalvar2 = np.median(varcheck2)
        else:
            typicalvar2 = np.mean(varcheck2)
        varlimit = varcheckthresh * typicalvar2

        cv,cp = np.where(varcheck2 > varlimit)  # voxels with high variance
        if len(cv) > 0:
            for vv in range(len(cv)):
                rdtemp[cv[vv],cp[vv],:] = np.zeros(tsize)   # replace with zeros so the crazy variance does not mess up clustering
            for vv in range(len(cv)):
                meanrun = np.mean(rdtemp[cv[vv], :, :], axis=0)
                rdtemp[cv[vv], cp[vv], :] = copy.deepcopy(meanrun)  # replace with the average now, with the high variance runs zeroed out
            print('---------------------------------!!!!!--------------------------------------');
            print('Variance check found {} voxels with high variance ({:.1f} percent of total)'.format(len(cv), 100. * len(cv)/nvarvalstotal))
            print('---------------------------------!!!!!--------------------------------------\n');
        else:
            print('Variance check did not find any voxels with high variance');

        # regiondata = rdtemp.reshape(nvox, ts, order = 'F').copy()
        regiondata = np.reshape(rdtemp, (nvox, ts))
        # ------------done correcting for crazy variance - -------------------

        #------------replace in define_cluster_and_load_data function---------------
        IDX = copy.deepcopy(region_coordinate_list[nn]['IDX'])
        #------------end of replace in define_cluster_and_load_data function--------

        tc = np.zeros([nclusters,ts])
        tc_sem = np.zeros([nclusters,ts])
        for aa in range(nclusters):
            # cc = [i for i in range(len(IDX)) if IDX[i] == aa]
            cc = np.where(IDX == aa)[0]
            nvox = len(cc)
            if nvox > 0:
                tc[aa,:] = np.mean(regiondata[cc, :], axis=0)
                tc_sem[aa,:] = np.std(regiondata[cc, :], axis=0)/np.sqrt(nvox)
            else:
                print('---------------CHECK THIS!-----------------------------')
                print('region {} cluster {} does not contain any data!'.format(rname,aa))
                print('-------------------------------------------------------')
        tc_original = copy.deepcopy(tc)
        tc_sem_original = copy.deepcopy(tc_sem)


        # handle high variance voxels differently
        # rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
        cv2,cp2 = np.where(varcheck2 <= varlimit)  # voxels without high variance
        tcr = np.zeros([nclusters,nruns_total,tsize])
        tcr_sem = np.zeros([nclusters,nruns_total,tsize])

        for aa in range(nclusters):
            # cc = [i for i in range(len(IDX)) if IDX[i] == aa]
            # cc = np.where(IDX == aa)[0]
            for bb in range(nruns_total):
                # rcheck = np.where(cp2 == bb)[0]   # find the entries for this run
                # vcheck = cv2[rcheck]    # find the good voxels for this run
                # cc2 = [i for i in cc if i in vcheck]
                # cc2r = np.where( (varcheck2[:,bb] <= varlimit) & (IDX == aa))  # good voxels for this run

                cc2 = [xx for xx in range(len(IDX)) if (varcheck2[xx,bb] <= varlimit) and (IDX[xx] == aa)]
                nvox2 = len(cc2)
                if nvox2 > 0:
                    tcr[aa,bb,:] = np.mean(rdtemp[cc2, bb, :], axis=0)
                    tcr_sem[aa,:] = np.std(rdtemp[cc2, bb, :], axis=0)/np.sqrt(nvox2)
        tc = np.reshape(tcr,(nclusters,ts))
        tc_sem = np.reshape(tcr_sem,(nclusters,ts))

        clusterdef_entry = {'cx':cx, 'cy':cy, 'cz':cz,'IDX':IDX, 'nclusters':nclusters, 'rname':rname, 'regionindex':regionindex, 'regionnum':regionnum, 'occurrence':occurrence}
        regiondata_entry = {'tc':tc, 'tc_sem':tc_sem, 'tc_original':tc_original, 'tc_sem_original':tc_sem_original, 'nruns_per_person':nruns_per_person, 'tsize':tsize, 'rname':rname, 'DBname':DBname, 'DBnum':DBnum, 'prefix':prefix, 'occurrence':occurrence}
        region_properties.append(regiondata_entry)
        cluster_properties.append(clusterdef_entry)

    print('loading cluster data complete.')
    return region_properties


def load_cluster_data(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
    '''
    Function to load data from a group, using a predefined cluster definition
    load_cluster_data in pyclustering.py
    region_properties = load_cluster_data(cluster_properties, DBname, DBnum, prefix, networkmodel, template_img, ...
                                        regionmap_img, anatlabels)
    :param cluster_properties:  cluster definition data, created in define_clusters_and_load_data
    :param DBname:  name of the database file (probably an excel file)
    :param DBnum:   list of the database entry numbers to use
    :param prefix:   prefix of the nifti format image files to read (indicates the preprocessing
                        steps that have been applied)
    :param networkmodel:  the network definition file name (probably an excel file)
    :return:  output is cluster_properties, region_properties
            cluster_properties is an array of dictionaries (one entry per cluster),
                with keys cx, cy, cz, IDX, nclusters, rname, regionindex, regionnum
                cx, cy, cz are the 3D coordinates of the voxels in each cluster
                IDX is the list of cluster labels for each voxel
                nclusters is the number of clusters for the region
                rname is the region name, regionindex in the index in the anat template definition, and
                    regionnum is the number index used to indicate voxels in the anat template
            region_properties is an array of dictionaries (one entry per cluster),
                        with keys: tc, tc_sem, nruns_per_person, and tsize
                        tc and tc_sem are the average and standard error of the time-course for each cluster
                        nruns_per_person lists the number of data sets that are concatenated per person
                        tsize is the number of time points in each individual fMRI run
    '''

    #  data will be concatenated across the entire group, with all runs in each person:  tsize  = N x ts
    # need to save data on how much runs are loaded per person
    # mode = 'concatenate_group'
    # # nvolmask is similar to removing the initial volumes, except this is the number of volumes that are replaced
    # # by a later volume, so that the effects of the initial volumes are not present, but the total number of volumes
    # # has not changed
    # nvolmask = 2
    # print('load_cluster_data:  DBname = ', DBname)
    # group_data = GLMfit.compile_data_sets(DBname, DBnum, prefix, mode, nvolmask)  # from GLMfit.py
    # # group_data is now xs, ys, zs, tsize
    # xs, ys, zs, ts = group_data.shape

    # the voxels in the regions of interest need to be extracted
    filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list', separate_conditions = True)
    nruns_per_person = np.zeros(NP).astype(int)
    for nn in range(NP):
        nruns_per_person[nn] = len(filename_list[nn])
    nruns_total = np.sum(nruns_per_person)

    # load information about the network
    network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel, exclude_latent = True)

    # identify the voxels in the regions of interest
    region_properties = []
    for nn, rname in enumerate(sem_region_list):
        print('loading data for region {}'.format(rname))
        rname_check = cluster_properties[nn]['rname']
        regionindex = cluster_properties[nn]['regionindex']
        regionnum = cluster_properties[nn]['regionnum']
        cx = cluster_properties[nn]['cx']
        cy = cluster_properties[nn]['cy']
        cz = cluster_properties[nn]['cz']
        IDX = cluster_properties[nn]['IDX']
        nclusters = cluster_properties[nn]['nclusters']

        if rname_check != rname:
            print('Problem with inconsistent cluster and network definitions!')
            return region_properties  # to this point region_properties will be incomplete
        else:
            # load the data one region at a time to save memory - necessary for large data sets
            mode = 'concatenate'
            # nvolmask = 2  # changed this to be an input parameter
            regiondata = load_data_from_region(filename_list, nvolmask, mode, cx, cy, cz)
            nvox,ts = np.shape(regiondata)

            # regiondata = group_data[cx, cy, cz, :]  # nvox x tsize

            # -----------------check for extreme variance------------
            tsize = int(ts / nruns_total)
            nvox = len(cx)
            rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order='F').copy()
            varcheck2 = np.var(rdtemp, axis=1)

            # options for identifying voxels with inordinately high variance
            # mean or median?
            # multiple of typical variance?
            # make these inputs?
            if varcheckmethod == 'median':
                typicalvar2 = np.median(varcheck2)
            else:
                typicalvar2 = np.mean(varcheck2)
            varlimit = varcheckthresh * typicalvar2

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

            tc = np.zeros([nclusters, ts])
            tc_sem = np.zeros([nclusters, ts])
            for aa in range(nclusters):
                # cc = [i for i in range(len(IDX)) if IDX[i] == aa]
                cc = np.where(IDX == aa)[0]
                nvox = len(cc)
                tc[aa, :] = np.mean(regiondata[cc, :], axis=0)
                tc_sem[aa, :] = np.std(regiondata[cc, :], axis=0) / np.sqrt(nvox)

            regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize, 'rname':rname, 'DBname':DBname, 'DBnum':DBnum, 'prefix':prefix}
            region_properties.append(regiondata_entry)

    return region_properties


