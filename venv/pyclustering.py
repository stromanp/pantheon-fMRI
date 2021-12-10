#-----------------------------------------------------------------------------------
#  cluster data for predefined ROIs, based on voxel time-course properties
#
# 1) identify data sets to load
# 2) identify regions, and number of clusters per region
# 3) load all voxel data and perform clustering

import numpy as np
import os
import pandas as pd
import nibabel as nib
import pybasissets
import pydatabase
import GLMfit
from sklearn.cluster import KMeans


def load_network_model(networkmodel):
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
    dnclusters = pd.read_excel(xls, 'nclusters')

    nregions = len(dnclusters)
    ntargets, ncols = dnet.shape
    nsources_max = ncols-1

    sem_region_list = []
    ncluster_list = []
    for nn in range(nregions):
        sem_region_list.append(dnclusters.loc[nn,'name'])
        entry = {'name':dnclusters.loc[nn,'name'],'nclusters':dnclusters.loc[nn,'nclusters']}
        ncluster_list.append(entry)

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


def define_clusters_and_load_data(DBname, DBnum, prefix, networkmodel, regionmap_img, anatlabels):
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
    mode = 'concatenate_group'
    # nvolmask is similar to removing the initial volumes, except this is the number of volumes that are replaced
    # by a later volume, so that the effects of the initial volumes are not present, but the total number of volumes
    # has not changed
    nvolmask = 2
    group_data = GLMfit.compile_data_sets(DBname, DBnum, prefix, mode, nvolmask)   # from GLMfit.py
    # group_data is now xs, ys, zs, tsize
    xs,ys,zs,ts = group_data.shape

    # the voxels in the regions of interest need to be extracted
    filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
    nruns_per_person = np.zeros(NP).astype(int)
    for nn in range(NP):
        nruns_per_person[nn] = len(filename_list[nn])
    nruns_total = np.sum(nruns_per_person)

    # load information about the network
    network, ncluster_list, sem_region_list = load_network_model(networkmodel)

    anatnamelist = []
    for name in anatlabels['names']:
        anatnamelist.append(name)

    # identify the voxels in the regions of interest
    region_properties = []
    cluster_properties = []
    for nn, rname in enumerate(sem_region_list):
        regionindex = anatnamelist.index(rname)
        regionnum = anatlabels['numbers'][regionindex]
        cx,cy,cz = np.where(regionmap_img == regionnum)

        regiondata = group_data[cx,cy,cz,:]   # nvox x tsize

        #-----------------check for extreme variance------------
        tsize = int(ts/nruns_total)
        nvox = len(cx)
        rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order = 'F').copy()
        varcheck2 = np.var(rdtemp, axis = 1)
        typicalvar2 = np.mean(varcheck2)
        varlimit = 5.0 * typicalvar2
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
        nclusters = ncluster_list[nn]['nclusters']
        kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(regiondata)
        IDX = kmeans.labels_
        cluster_tc = kmeans.cluster_centers_

        tc = np.zeros([nclusters,ts])
        tc_sem = np.zeros([nclusters,ts])
        for aa in range(nclusters):
            cc = [i for i in range(len(IDX)) if IDX[i] == aa]
            nvox = len(cc)
            tc[aa,:] = np.mean(regiondata[cc, :], axis=0)
            tc_sem[aa,:] = np.std(regiondata[cc, :], axis=0)/np.sqrt(nvox)

        clusterdef_entry = {'cx':cx, 'cy':cy, 'cz':cz,'IDX':IDX, 'nclusters':nclusters, 'rname':rname, 'regionindex':regionindex, 'regionnum':regionnum}
        regiondata_entry = {'tc':tc, 'tc_sem':tc_sem, 'nruns_per_person':nruns_per_person, 'tsize':tsize, 'rname':rname, 'DBname':DBname, 'DBnum':DBnum, 'prefix':prefix}
        region_properties.append(regiondata_entry)
        cluster_properties.append(clusterdef_entry)

    print('cluster definition complete.')
    return cluster_properties, region_properties


def load_cluster_data(cluster_properties, DBname, DBnum, prefix, networkmodel):
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
    mode = 'concatenate_group'
    # nvolmask is similar to removing the initial volumes, except this is the number of volumes that are replaced
    # by a later volume, so that the effects of the initial volumes are not present, but the total number of volumes
    # has not changed
    nvolmask = 2
    print('load_cluster_data:  DBname = ', DBname)
    group_data = GLMfit.compile_data_sets(DBname, DBnum, prefix, mode, nvolmask)  # from GLMfit.py
    # group_data is now xs, ys, zs, tsize
    xs, ys, zs, ts = group_data.shape

    # the voxels in the regions of interest need to be extracted
    filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
    nruns_per_person = np.zeros(NP).astype(int)
    for nn in range(NP):
        nruns_per_person[nn] = len(filename_list[nn])
    nruns_total = np.sum(nruns_per_person)

    # load information about the network
    network, ncluster_list, sem_region_list = load_network_model(networkmodel)

    # anatnamelist = []
    # for name in anatlabels['names']:
    #     anatnamelist.append(name)

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
            regiondata = group_data[cx, cy, cz, :]  # nvox x tsize

            # -----------------check for extreme variance------------
            tsize = int(ts / nruns_total)
            nvox = len(cx)
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




