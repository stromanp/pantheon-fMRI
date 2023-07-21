"""
pyclustering.py

This is a set of programs for clustering data for predefined ROIs, based on voxel time-course properties
1) identify data sets to load
2) identify regions, and number of clusters per region
3) load all voxel data and perform clustering

"""

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# "Pantheon" is a python software repository for complete analysis of functional
# magnetic resonance imaging data at all level of the central nervous system,
# including the brain, brainstem, and spinal cord.
#
# The software in this repository was written by P. Stroman, and the bulk of the methods in this
# package have been developed by P. W. Stroman, Queen's University at Kingston, Ontario, Canada.
#
# Some of the methods have been adapted from other freely available packages
# as noted in the documentation.
#
# This software is for research purposes only, and no guarantees are given that it is
# free of bugs or errors.
#
# Use this software as needed, with the condition that you reference it in any
# published works or presentations, with the following citations:
#
# Proof-of-concept of a novel structural equation modelling approach for the analysis of
# functional MRI data applied to investigate individual differences in human pain responses
# P. W. Stroman, J. M. Powers, G. Ioachim
# Human Brain Mapping, 44:2523–2542 (2023). https://doi.org/10.1002/hbm.26228
#
#  Ten key insights into the use of spinal cord fMRI
#  J. M Powers, G. Ioachim, P. W. Stroman
#  Brain Sciences 8(9), (DOI: 10.3390/brainsci8090173 ) 2018.
#
#  Validation of structural equation modeling (SEM) methods for functional MRI data acquired in the human brainstem and spinal cord
#  P. W. Stroman
#  Critical Reviews in Biomedical Engineering 44(4): 227-241 (2016).
#
#  Assessment of data acquisition parameters, and analysis techniques for noise
#  reduction in spinal cord fMRI data
#  R.L. Bosma & P.W. Stroman
#  Magnetic Resonance Imaging, 2014 (10.1016/j.mri.2014.01.007).
#
# also see https://www.queensu.ca/academia/stromanlab/
#
# Patrick W. Stroman, Queen's University, Centre for Neuroscience Studies
# stromanp@queensu.ca
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


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



def load_network_model(networkmodel, exclude_latent = False):
    '''
    load_network_model in pyclustering.py
    network, ncluster_list, sem_region_list = load_network_model(networkmodel)
    :param networkmodel:  the name of the excel format file that contains the network definition
            This file must contain two excel sheets, named 'connections' and 'nclusters'
            The 'connections' sheet has a row for each target. The target name is in the first column, the 2nd
                column is blank, and the next columns list the source names, one source per column
            The 'nclusters' sheet has a row for each region in the network.  The 1st column contains the name of the
            region, and the 2nd column has the number of clusters to be used for that region
    :return: network, ncluster_list, sem_region_list
            network - this is an array of dictionary items keys: target, sources, targetnum, sourcenums
                that contains the network model information for use by other functions
            ncluster_list - this is an array of dictionary items with keys: name, nclusters
            sem_region_list - a list of region names in the order that will be used for data, and the same
                order that is used for teh network and ncluster_list data
    '''
    xls = pd.ExcelFile(networkmodel, engine = 'openpyxl')
    dnet = pd.read_excel(xls, 'connections')
    dnet.pop('Unnamed: 0')   # remove this blank field from the beginning
    column_names = dnet.columns
    for checkname in column_names:
        if 'Unnamed' in checkname:
            print('found and removed invalid column:  {}'.format(checkname))
            dnet.pop(checkname)   # remove invalid columns
    dnclusters = pd.read_excel(xls, 'nclusters')

    nregions = len(dnclusters)
    ntargets, ncols = dnet.shape
    nsources_max = ncols-1

    sem_region_list = []
    ncluster_list = []
    for nn in range(nregions):
        check_latent = ('intrinsic' in dnclusters.loc[nn,'name']) or ('latent' in dnclusters.loc[nn,'name'])
        if check_latent and exclude_latent:
            print('latent component of network model not included: {}'.format(dnclusters.loc[nn,'name']))
        else:
            entry = {'name':dnclusters.loc[nn,'name'],'nclusters':dnclusters.loc[nn,'nclusters']}
            ncluster_list.append(entry)
            sem_region_list.append(dnclusters.loc[nn,'name'])

    network = []
    for nn in range(ntargets):
        targetname = dnet.loc[nn,'target']
        targetnum = sem_region_list.index(targetname)
        sourcelist = []
        sourcenumlist = []
        for ss in range(nsources_max):
            tag = 'source'+str(ss+1)
            try:
                sourcename = dnet.loc[nn,tag]
                if not str(sourcename) == 'nan':
                    sourcelist.append(sourcename)
                    sourcenum = sem_region_list.index(sourcename)
                    sourcenumlist.append(sourcenum)
            except:
                print('source {} for target {} ignored in network definition - invalid source'.format(sourcename,targetname))

        entry = {'target':targetname, 'sources':sourcelist, 'targetnum':targetnum, 'sourcenums':sourcenumlist}
        network.append(entry)

    return network, ncluster_list, sem_region_list



def define_clusters_and_load_data(DBname, DBnum, prefix, networkmodel, regionmap_img, anatlabels, varcheckmethod = 'median', varcheckthresh = 3.0):
    '''
    Function to load data from a group, and define clusters based on voxel time-course properties
    define_clusters_and_load_data in pyclustering.py
    cluster_properties, region_properties = define_clusters_and_load_data(DBname, DBnum, prefix, networkmodel, ...
                regionmap_img, anatlabels)
    :param DBname:  name of the database file (probably an excel file)
    :param DBnum:   list of the database entry numbers to use
    :param prefix:   prefix of the nifti format image files to read (indicates the preprocessing
                        steps that have been applied)
    :param networkmodel:  the network definition file name (probably an excel file)
    :param regionmap_img:  the spatially normalized region map, from the function 'load_template' in
                        module 'load_templates'
    :param anatlabels:  the list of region label names, and the corresponding number used to identify them,
                        also from 'load_template'
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
    # nvolmask is similar to removing the initial volumes, except this is the number of volumes that are replaced
    # by a later volume, so that the effects of the initial volumes are not present, but the total number of volumes
    # has not changed

    # old method ... (uses more memory)
    # nvolmask = 2
    # group_data = GLMfit.compile_data_sets(DBname, DBnum, prefix, mode, nvolmask)   # from GLMfit.py
    # # group_data is now xs, ys, zs, tsize
    # xs,ys,zs,ts = group_data.shape

    # the voxels in the regions of interest need to be extracted
    filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
    nruns_per_person = np.zeros(NP).astype(int)
    for nn in range(NP):
        nruns_per_person[nn] = len(filename_list[nn])
    nruns_total = np.sum(nruns_per_person)

    # load information about the network
    network, ncluster_list, sem_region_list = load_network_model(networkmodel, exclude_latent = True)

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
        print('searching for region {} {}'.format(rname,regionnum))
        if len(regionnum) > 1:
            # if the number of clusters divides evenly into the number of regions
            # then split the clusters amongst the regions (to maintain R/L divisions for example)
            # otherwise, put all the regions together
            clusters_per_region = ncluster_list[nn]['nclusters']/len(regionnum)
            if clusters_per_region == np.floor(clusters_per_region):
                for aa,rr in enumerate(regionnum):
                    cx,cy,cz = np.where(regionmap_img == rr)
                    region_start += [vox_count]
                    region_end += [vox_count + len(cx)]
                    vox_count += len(cx)
                    region_coordinate_list.append({'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence':aa})
                    if (nn == 0) and (aa == 0):
                        ncluster_list2 = [np.ceil(ncluster_list[nn]['nclusters']/len(regionnum)).astype(int)]
                        cx_all = cx
                        cy_all = cy
                        cz_all = cz
                    else:
                        ncluster_list2 += [np.ceil(ncluster_list[nn]['nclusters']/len(regionnum)).astype(int)]
                        cx_all = np.concatenate((cx_all, cx), axis=0)
                        cy_all = np.concatenate((cy_all, cy), axis=0)
                        cz_all = np.concatenate((cz_all, cz), axis=0)
            else:
                for aa,rr in enumerate(regionnum):
                    cx,cy,cz = np.where(regionmap_img == rr)
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
            cx,cy,cz = np.where(regionmap_img == regionnum.values[0])

            region_start += [vox_count]
            region_end += [vox_count+len(cx)]
            vox_count += len(cx)
            region_coordinate_list.append({'rname':rname, 'nvox':len(cx), 'cx':cx, 'cy':cy, 'cz':cz, 'occurrence':0})
            if nn == 0:
                ncluster_list2 = [ncluster_list[nn]['nclusters']]
                cx_all = cx
                cy_all = cy
                cz_all = cz
            else:
                ncluster_list2 += [ncluster_list[nn]['nclusters']]
                cx_all = np.concatenate((cx_all,cx),axis=0)
                cy_all = np.concatenate((cy_all,cy),axis=0)
                cz_all = np.concatenate((cz_all,cz),axis=0)

    # regiondata = group_data[cx,cy,cz,:]   # nvox x tsize
    # load the data one region at a time to save memory - necessary for large data sets
    mode = 'concatenate'
    nvolmask = 2
    allregiondata = load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all)
    nvox,ts = np.shape(allregiondata)

    region_name_list = [region_coordinate_list[x]['rname'] for x in range(len(region_coordinate_list))]
    for nn, rname in enumerate(region_name_list):
        nvox = region_coordinate_list[nn]['nvox']
        cx = region_coordinate_list[nn]['cx']
        cy = region_coordinate_list[nn]['cy']
        cz = region_coordinate_list[nn]['cz']
        occurrence = region_coordinate_list[nn]['occurrence']
        n1 = region_start[nn]
        n2 = region_end[nn]

        regiondata = allregiondata[n1:n2,:]

        #-----------------check for extreme variance------------
        tsize = int(ts/nruns_total)
        rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order = 'F').copy()
        varcheck2 = np.var(rdtemp, axis = 1)

        if varcheckmethod == 'median':
            typicalvar2 = np.median(varcheck2)
        else:
            typicalvar2 = np.mean(varcheck2)
        varlimit = varcheckthresh * typicalvar2

        cv,cp = np.where(varcheck2 > varlimit)  # voxels with crazy variance
        if len(cv) > 0:
            for vv in range(len(cv)):
                rdtemp[cv[vv],:,cp[vv]] = np.zeros(tsize)
            print('---------------!!!!!----------------------');
            print('Variance check found {} crazy voxels'.format(len(cv)) )
            print('---------------!!!!!----------------------\n');
        else:
            print('Variance check did not find any crazy voxels');
        regiondata = rdtemp.reshape(nvox, ts, order = 'F').copy()
        # ------------done correcting for crazy variance - -------------------

        # now do the clustering for this region
        varcheck = np.var(regiondata, axis = 1)
        # cvox = np.where(varcheck > 0) # exclude voxels with effectively constant values
        cvox = [i for i in range(len(varcheck)) if varcheck[i] > 0]
        print('using {} voxels of {} with non-zero variance for defining clusters'.format(len(cvox), nvox))
        if len(cvox)>0:
            regiondata = regiondata[cvox, :]
            cx = cx[cvox]
            cy = cy[cvox]
            cz = cz[cvox]

        # divide each region into N clusters with similar timecourse properties
        nclusters = ncluster_list2[nn]
        kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(regiondata)
        # IDX = kmeans.labels_
        # cluster_tc = kmeans.cluster_centers_

        # modified clustering method - for roughly equal size clusters
        # Thanks to Eyal Shulman who shared on StackOverflow  https://stackoverflow.com/users/6247548/eyal-shulman
        # method for making clusters approximately equal size
        nvoxels, tsizefull = np.shape(regiondata)
        cluster_size = np.floor(nvoxels/nclusters).astype(int)
        centers = kmeans.cluster_centers_
        centers = centers.reshape(-1, 1, regiondata.shape[-1]).repeat(cluster_size, 1).reshape(-1, regiondata.shape[-1])
        distance_matrix = cdist(regiondata, centers)
        IDX = linear_sum_assignment(distance_matrix)[1] // cluster_size

        tc = np.zeros([nclusters,ts])
        tc_sem = np.zeros([nclusters,ts])
        for aa in range(nclusters):
            cc = [i for i in range(len(IDX)) if IDX[i] == aa]
            nvox = len(cc)
            tc[aa,:] = np.mean(regiondata[cc, :], axis=0)
            tc_sem[aa,:] = np.std(regiondata[cc, :], axis=0)/np.sqrt(nvox)

        clusterdef_entry = {'cx':cx, 'cy':cy, 'cz':cz,'IDX':IDX, 'nclusters':nclusters, 'rname':rname, 'regionindex':regionindex, 'regionnum':regionnum, 'occurrence':occurrence}
        regiondata_entry = {'tc':tc, 'tc_sem':tc_sem, 'nruns_per_person':nruns_per_person, 'tsize':tsize, 'rname':rname, 'DBname':DBname, 'DBnum':DBnum, 'prefix':prefix, 'occurrence':occurrence}
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

                        cx = np.concatenate((cx,cx2),axis=0)
                        cy = np.concatenate((cy,cy2),axis=0)
                        cz = np.concatenate((cz,cz2),axis=0)
                        IDX = np.concatenate((IDX,IDX2+ncluster_total),axis=0)
                        ncluster_total += nclusters2
                        tc = np.concatenate((tc,tc2),axis=0)
                        tc_sem = np.concatenate((tc,tc_sem2),axis=0)

                clusterdef_entry_temp = {'cx': cx, 'cy': cy, 'cz': cz, 'IDX': IDX, 'nclusters': ncluster_total, 'rname': rname, 'regionindex': regionindex, 'regionnum': regionnum}
                regiondata_entry_temp = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize, 'rname': rname, 'DBname': DBname, 'DBnum': DBnum, 'prefix': prefix}

                cluster_properties2.append(clusterdef_entry_temp)
                region_properties2.append(regiondata_entry_temp)
            else:
                cluster_properties2.append(cluster_properties[cr[0]])
                region_properties2.append(region_properties[cr[0]])

        cluster_properties = cluster_properties2
        region_properties = region_properties2

    print('cluster definition complete.')
    return cluster_properties, region_properties


def load_cluster_data(cluster_properties, DBname, DBnum, prefix, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
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
    network, ncluster_list, sem_region_list = load_network_model(networkmodel, exclude_latent = True)

    # identify the voxels in the regions of interest
    region_properties = []
    for nn, rname in enumerate(sem_region_list):
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
            nvolmask = 2
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
                cc = [i for i in range(len(IDX)) if IDX[i] == aa]
                nvox = len(cc)
                tc[aa, :] = np.mean(regiondata[cc, :], axis=0)
                tc_sem[aa, :] = np.std(regiondata[cc, :], axis=0) / np.sqrt(nvox)

            regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize, 'rname':rname, 'DBname':DBname, 'DBnum':DBnum, 'prefix':prefix}
            region_properties.append(regiondata_entry)

    return region_properties


    # import matplotlib.pyplot as plt
    # aa = 1
    # cx = [i for i in range(len(IDX)) if IDX[i] == aa]
    # tc0 = np.mean(regiondata[cx, :], axis=0)
    # fig = plt.figure(23);  plt.plot(range(138), tc0, 'bo-');  plt.plot(range(138), cluster_tc[aa, :], 'g-')


    # going to need to check these voxel locations
    # import matplotlib.pyplot as plt
    # import copy
    #
    # background = template_img.astype(float) / template_img.max()
    # red = copy.deepcopy(background)
    # green = copy.deepcopy(background)
    # blue = copy.deepcopy(background)
    #
    # red[cx,cy,cz] = 1.0
    # green[cx,cy,cz] = 0.0
    # blue[cx,cy,cz] = 0.0
    #
    # sag_slice = 13
    # tcimg = np.dstack((red[sag_slice, :, :], green[sag_slice, :, :], blue[sag_slice, :, :]))
    # fig = plt.figure(21), plt.imshow(tcimg)
    #
    # ax_slice = 175
    # tcimg = np.dstack((red[:,:,ax_slice], green[:,:,ax_slice], blue[:,:,ax_slice]))
    # fig = plt.figure(22), plt.imshow(tcimg)


def load_data_from_region(filename_list, nvolmask, mode, cx, cy, cz):
    NP = len(filename_list)
    group_divisor = 0
    for pnum in range(NP):
        print('compile_data_sets:  reading participant data ', pnum + 1, ' of ', NP)
        # per_person_level
        list1 = filename_list[pnum]
        divisor = 0
        for runnum, name in enumerate(list1):
            # read in the name, and do something with the data ...
            # if mode is average, then sum the data for now
            # otherwise, concatenate the data
            input_img = nib.load(name)
            input_data = input_img.get_fdata()
            roi_data = input_data[cx,cy,cz,:]   # check the size of this
            print('size of roi_data = {}'.format(np.shape(roi_data)))

            nvox, ts = roi_data.shape
            # mask out the initial volumes, if wanted
            if nvolmask > 0:
                for tt in range(nvolmask): roi_data[:, tt] = roi_data[:, nvolmask]

            # # convert to signal change from the average----------------
            # if data have been cleaned they are already percent signal changes
            mean_data = np.mean(roi_data, axis=1)
            mean_data = np.repeat(mean_data[:, np.newaxis], ts, axis=1)
            roi_data = roi_data - mean_data

            if runnum == 0:
                person_data = roi_data
                divisor += 1
            else:
                if mode == 'average':
                    # average across the person
                    person_data += roi_data
                    divisor += 1
                else:
                    # concatenate across the person
                    person_data = np.concatenate((person_data, roi_data), axis=1)
        person_data = person_data / divisor  # average

        if pnum == 0:
            group_data = person_data
        else:
            group_data = np.concatenate((group_data, person_data), axis=1)

    return group_data


