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


def calc_distance_matrix(data1, data2):
    nvox1, tsize1 = np.shape(data1)
    nvox2, tsize2 = np.shape(data2)

    dist_matrix = np.zeros((nvox1,nvox2))
    for aa in range(nvox1):
        tc1 = data1[aa,:]
        for bb in range(nvox2):
            dist_matrix[aa,bb] = np.sqrt(np.sum( (tc1 - data2[bb,:])**2))

    return dist_matrix


def sort_by_dist(dist_matrix):
    # sort into roughly equal size bins based on distance
    nvox1, nvox2 = np.shape(dist_matrix)

    IDX = -np.ones(nvox1)
    voxcount = nvox1
    clist = list(range(nvox1))
    bin = 0
    while voxcount > 0:
        # assign one at a time, then take it out of the list
        dist = dist_matrix[clist,bin]
        dd = np.argmin(dist)
        IDX[clist[dd]] = bin
        voxcount -= 1
        if voxcount > 0:
            clist.remove(clist[dd])
        bin = (bin+1) % nvox2

    return IDX


def define_clusters_and_load_data(DBname, DBnum, prefix, nvolmask, networkmodel, regionmap_img, anatlabels, varcheckmethod = 'median', varcheckthresh = 3.0):
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
        regionnum = copy.deepcopy(anatlabels['numbers'][regionindex])
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
    # load all the data from all regions, all data sets
    mode = 'concatenate'
    allregiondata = load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all, definingclusters = True)
    nvox,ts = np.shape(allregiondata)

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

        regiondata = allregiondata[n1:n2,:]

        #-----------------check for high variance------------
        tsize = int(ts/nruns_total)
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
        high_var_record = {'cv':cv, 'cp':cp}
        if len(cv) > 0:
            for vv in range(len(cv)):
                rdtemp[cv[vv],cp[vv],:] = np.zeros(tsize)   # replace with zeros so the high variance does not mess up clustering
            for vv in range(len(cv)):
                meanrun = np.mean(rdtemp[cv[vv], :, :],axis=0)
                rdtemp[cv[vv], cp[vv], :] = copy.deepcopy(meanrun)  # replace with the average now, with the high variance runs zeroed out
            print('---------------------------------!!!!!--------------------------------------');
            print('Variance check found {} voxels with high variance ({:.1f} percent of total)'.format(len(cv), 100.*len(cv)/nvarvalstotal) )
            print('---------------------------------!!!!!--------------------------------------\n');
        else:
            print('Variance check did not find any voxels with high variance');
        # regiondata = rdtemp.reshape(nvox, ts, order = 'F').copy()
        regiondata = np.reshape(rdtemp, (nvox, ts))
        # ------------done correcting for crazy variance - -------------------

        # now do the clustering for this region
        #-----------------remove this part--------------------------
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
            cluster_size = np.floor(nvoxels/nclusters).astype(int)
            nvox_trunc = cluster_size * nclusters
            centers = kmeans.cluster_centers_
            centers = centers.reshape(-1, 1, regiondata.shape[-1]).repeat(cluster_size, 1).reshape(-1, regiondata.shape[-1])
            distance_matrix = cdist(regiondata[:nvox_trunc,:], centers)
            val = linear_sum_assignment(distance_matrix)
            IDX = val[1] // cluster_size

            # add in remaining voxels to the nearest clusters
            nresidual = nvoxels - nvox_trunc
            IDXresidual = []
            for xx in range(nresidual):
                tc = regiondata[-xx,:]
                dist = [np.sqrt(np.sum( tc - kmeans.cluster_centers_[dd,:])**2) for dd in range(nclusters)]
                IDXresidual += [np.argmin(dist)]
            IDX = np.concatenate((IDX, np.array(IDXresidual[::-1])),axis=0)

        tc = np.zeros([nclusters,ts])
        tc_sem = np.zeros([nclusters,ts])
        for aa in range(nclusters):
            # cc = [i for i in range(len(IDX)) if IDX[i] == aa]
            cc = np.where(IDX == aa)[0]
            nvox = len(cc)
            tc[aa,:] = np.mean(regiondata[cc, :], axis=0)
            tc_sem[aa,:] = np.std(regiondata[cc, :], axis=0)/np.sqrt(nvox)
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

                cc2 = [xx for xx in range(len(IDX)) if (varcheck2[xx, bb] <= varlimit) and (IDX[xx] == aa)]
                nvox2 = len(cc2)
                if nvox2 > 0:
                    tcr[aa, bb, :] = np.mean(rdtemp[cc2, bb, :], axis=0)
                    tcr_sem[aa, :] = np.std(rdtemp[cc2, bb, :], axis=0) / np.sqrt(nvox2)

        # check for bad data sets
        check = np.sum(tcr**2, axis = 2)
        cbad, rbad = np.where(check == 0)
        if len(cbad) > 0:
            print('rbad = {}'.format(rbad))
            print('size of DBnum = {}'.format(np.shape(DBnum)))
            print('\n-----------------------------------------------------------')
            print('pyclustering, define_clusters_and_load_data:')
            print('possible bad data sets in database numbers {}'.format(DBnum[rbad]))
            print('    ... values are all zeros')
            print('-----------------------------------------------------------\n')

        tc = np.reshape(tcr, (nclusters, ts))
        tc_sem = np.reshape(tcr_sem, (nclusters, ts))

        # for aa in range(nclusters):
        #     # cc = [i for i in range(len(IDX)) if IDX[i] == aa]
        #     cc = np.where(IDX == aa)[0]
        #     for bb in range(nruns_total):
        #         rcheck = np.where(cp2 == bb)[0]   # find the entries for this run
        #         vcheck = cv2[rcheck]    # find the good voxels for this run
        #         cc2 = [i for i in cc if i in vcheck]
        #         nvox2 = len(cc2)
        #         if nvox2 > 0:
        #             tcr[aa,bb,:] = np.mean(rdtemp[cc2, bb, :], axis=0)
        #             tcr_sem[aa,:] = np.std(rdtemp[cc2, bb, :], axis=0)/np.sqrt(nvox2)
        # tc = np.reshape(tcr,(nclusters,ts))
        # tc_sem = np.reshape(tcr_sem,(nclusters,ts))

        clusterdef_entry = {'cx':cx, 'cy':cy, 'cz':cz,'IDX':IDX, 'nclusters':nclusters, 'rname':rname, 'regionindex':regionindex, 'regionnum':regionnum, 'occurrence':occurrence}
        # Oct 8 2025 - removed 'DBname':DBname, 'DBnum':DBnum from regiondata_entry
        regiondata_entry = {'tc':tc, 'tc_sem':tc_sem, 'tc_original':tc_original, 'tc_sem_original':tc_sem_original, 'nruns_per_person':nruns_per_person, 'tsize':tsize, 'rname':rname, 'prefix':prefix, 'occurrence':occurrence}
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
                        # DBname = region_properties[cc]['DBname']
                        # DBnum = region_properties[cc]['DBnum']
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
                # Oct 8 2025 - removed 'DBname':DBname, 'DBnum':DBnum from regiondata_entry_temp
                regiondata_entry_temp = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize, 'rname': rname, 'prefix': prefix}

                cluster_properties2.append(clusterdef_entry_temp)
                region_properties2.append(regiondata_entry_temp)
            else:
                cluster_properties2.append(cluster_properties[cr[0]])
                region_properties2.append(region_properties[cr[0]])

        cluster_properties = cluster_properties2
        region_properties = region_properties2

    print('cluster definition complete.')
    return cluster_properties, region_properties




def update_oneregion_and_load_data(DBname, DBnum, prefix, nvolmask, cluster_properties, region_properties, regionname, nclusters, regionmap_img, anatlabels, varcheckmethod = 'median', varcheckthresh = 3.0):
    '''
    Function to load data from a group, and define clusters for one specific region based on voxel time-course properties
    If the region already exists in the cluster definition then it is updated/overwritten
    update_oneregion_and_load_data in pyclustering.py
    cluster_properties, region_properties = define_clusters_and_load_data(DBname, DBnum, prefix, nvolmask,
                region_properties, regionname, regionmap_img, anatlabels)
    :param DBname:  name of the database file (probably an excel file)
    :param DBnum:   list of the database entry numbers to use
    :param prefix:   prefix of the nifti format image files to read (indicates the preprocessing
                        steps that have been applied)
    :param nvolmask:  number of initial volumes to mask out, to avoid non-steady-state T1-weighting in the fMRI data
    :param cluster_properties: the existing cluster definition that is to be updated or revised
    :param region_properties:  the existing region properties that are to be updated or revised
    :param regionname:  name of the region to be added or updated
    :param nclusters:  number of clusters in the new region/cluster definition
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

    # the voxels in the regions of interest need to be extracted
    filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
    nruns_per_person = np.zeros(NP).astype(int)
    for nn in range(NP):
        nruns_per_person[nn] = len(filename_list[nn])
    nruns_total = np.sum(nruns_per_person)

    anatnamelist = []
    for name in anatlabels['names']:
        anatnamelist.append(name)

    # identify the voxels in the regions of interest
    new_region_properties = []
    new_cluster_properties = []
    region_coordinate_list = []
    region_start = []
    region_end = []
    # ncluster_list2 = []
    # vox_count = 0

    # load the region information
    # regionindex = anatnamelist.index(rname)
    regionindex = [x for x, name in enumerate(anatnamelist) if name == regionname]
    regionnum = copy.deepcopy(anatlabels['numbers'][regionindex])
    print('searching for region {} {}'.format(rname,regionnum))
    if len(regionnum) > 1:
        # if the number of clusters divides evenly into the number of regions
        # then split the clusters amongst the regions (to maintain R/L divisions for example)
        # otherwise, put all the regions together
        clusters_per_region = nclusters/len(regionnum)
        if clusters_per_region == np.floor(clusters_per_region):
            for aa,rr in enumerate(regionnum):
                cx,cy,cz = np.where(regionmap_img == rr)
                region_start += [vox_count]
                region_end += [vox_count + len(cx)]
                vox_count += len(cx)
                region_coordinate_list.append({'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence':aa})
                if (aa == 0):
                    ncluster_list2 = [np.ceil(nclusters/len(regionnum)).astype(int)]
                    cx_all = cx
                    cy_all = cy
                    cz_all = cz
                else:
                    ncluster_list2 += [np.ceil(nclusters/len(regionnum)).astype(int)]
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

            cx = copy.deepcopy(cx_full)
            cy = copy.deepcopy(cy_full)
            cz = copy.deepcopy(cz_full)

            region_start += [vox_count]
            region_end += [vox_count + len(cx)]
            vox_count += len(cx)
            region_coordinate_list.append(
                {'rname': rname, 'nvox': len(cx), 'cx': cx, 'cy': cy, 'cz': cz, 'occurrence': 0})

            # ncluster_list2 = [ncluster_list[nn]['nclusters']]
            cx_all = cx
            cy_all = cy
            cz_all = cz
    else:
        cx,cy,cz = np.where(regionmap_img == regionnum.values[0])

        region_start += [vox_count]
        region_end += [vox_count+len(cx)]
        vox_count += len(cx)
        region_coordinate_list.append({'rname':rname, 'nvox':len(cx), 'cx':cx, 'cy':cy, 'cz':cz, 'occurrence':0})

        cx_all = cx
        cy_all = cy
        cz_all = cz


    # regiondata = group_data[cx,cy,cz,:]   # nvox x tsize
    # load all the data from all regions, all data sets
    mode = 'concatenate'
    allregiondata = load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all, definingclusters = True)
    nvox,ts = np.shape(allregiondata)

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

        regiondata = allregiondata[n1:n2,:]

        #-----------------check for high variance------------
        tsize = int(ts/nruns_total)
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
        high_var_record = {'cv':cv, 'cp':cp}
        if len(cv) > 0:
            for vv in range(len(cv)):
                rdtemp[cv[vv],cp[vv],:] = np.zeros(tsize)   # replace with zeros so the high variance does not mess up clustering
            for vv in range(len(cv)):
                meanrun = np.mean(rdtemp[cv[vv], :, :],axis=0)
                rdtemp[cv[vv], cp[vv], :] = copy.deepcopy(meanrun)  # replace with the average now, with the high variance runs zeroed out
            print('---------------------------------!!!!!--------------------------------------');
            print('Variance check found {} voxels with high variance ({:.1f} percent of total)'.format(len(cv), 100.*len(cv)/nvarvalstotal) )
            print('---------------------------------!!!!!--------------------------------------\n');
        else:
            print('Variance check did not find any voxels with high variance');
        # regiondata = rdtemp.reshape(nvox, ts, order = 'F').copy()
        regiondata = np.reshape(rdtemp, (nvox, ts))
        # ------------done correcting for crazy variance - -------------------

        # now do the clustering for this region
        # divide each region into N clusters with similar timecourse properties
        # nclusters = ncluster_list2[nn]
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
            cluster_size = np.floor(nvoxels/nclusters).astype(int)
            nvox_trunc = cluster_size * nclusters
            centers = kmeans.cluster_centers_
            centers = centers.reshape(-1, 1, regiondata.shape[-1]).repeat(cluster_size, 1).reshape(-1, regiondata.shape[-1])
            distance_matrix = cdist(regiondata[:nvox_trunc,:], centers)
            val = linear_sum_assignment(distance_matrix)
            IDX = val[1] // cluster_size

            # add in remaining voxels to the nearest clusters
            nresidual = nvoxels - nvox_trunc
            IDXresidual = []
            for xx in range(nresidual):
                tc = regiondata[-xx,:]
                dist = [np.sqrt(np.sum( tc - kmeans.cluster_centers_[dd,:])**2) for dd in range(nclusters)]
                IDXresidual += [np.argmin(dist)]
            IDX = np.concatenate((IDX, np.array(IDXresidual[::-1])),axis=0)

        tc = np.zeros([nclusters,ts])
        tc_sem = np.zeros([nclusters,ts])
        for aa in range(nclusters):
            # cc = [i for i in range(len(IDX)) if IDX[i] == aa]
            cc = np.where(IDX == aa)[0]
            nvox = len(cc)
            tc[aa,:] = np.mean(regiondata[cc, :], axis=0)
            tc_sem[aa,:] = np.std(regiondata[cc, :], axis=0)/np.sqrt(nvox)
        tc_original = copy.deepcopy(tc)
        tc_sem_original = copy.deepcopy(tc_sem)

        # handle high variance voxels differently
        # rdtemp = np.reshape(regiondata, (nvox, nruns_total, tsize))
        cv2,cp2 = np.where(varcheck2 <= varlimit)  # voxels without high variance
        tcr = np.zeros([nclusters,nruns_total,tsize])
        tcr_sem = np.zeros([nclusters,nruns_total,tsize])

        for aa in range(nclusters):
            for bb in range(nruns_total):
                cc2 = [xx for xx in range(len(IDX)) if (varcheck2[xx, bb] <= varlimit) and (IDX[xx] == aa)]
                nvox2 = len(cc2)
                if nvox2 > 0:
                    tcr[aa, bb, :] = np.mean(rdtemp[cc2, bb, :], axis=0)
                    tcr_sem[aa, :] = np.std(rdtemp[cc2, bb, :], axis=0) / np.sqrt(nvox2)

        # check for bad data sets
        check = np.sum(tcr**2, axis = 2)
        cbad, rbad = np.where(check == 0)
        if len(cbad) > 0:
            print('rbad = {}'.format(rbad))
            print('size of DBnum = {}'.format(np.shape(DBnum)))
            print('\n-----------------------------------------------------------')
            print('pyclustering, define_clusters_and_load_data:')
            print('possible bad data sets in database numbers {}'.format(DBnum[rbad]))
            print('    ... values are all zeros')
            print('-----------------------------------------------------------\n')

        tc = np.reshape(tcr, (nclusters, ts))
        tc_sem = np.reshape(tcr_sem, (nclusters, ts))

        clusterdef_entry = {'cx':cx, 'cy':cy, 'cz':cz,'IDX':IDX, 'nclusters':nclusters, 'rname':rname, 'regionindex':regionindex, 'regionnum':regionnum, 'occurrence':occurrence}
        # Oct 8 2025 - removed 'DBname':DBname, 'DBnum':DBnum from regiondata_entry
        regiondata_entry = {'tc':tc, 'tc_sem':tc_sem, 'tc_original':tc_original, 'tc_sem_original':tc_sem_original, 'nruns_per_person':nruns_per_person, 'tsize':tsize, 'rname':rname, 'prefix':prefix, 'occurrence':occurrence}

        new_region_properties.append(regiondata_entry)
        new_cluster_properties.append(clusterdef_entry)

    # combine repeated occurrences, if they occur
    occurrences = [new_cluster_properties[x]['occurrence'] for x in range(len(new_cluster_properties))]
    if (np.array(occurrences) > 0).any():
        rnamelist = [new_cluster_properties[x]['rname'] for x in range(len(new_cluster_properties))]
        new_cluster_properties2 = []
        new_region_properties2 = []
        # for nn, rname in enumerate(sem_region_list):

        cr = [x for x in range(len(rnamelist)) if rnamelist[x] == regionname]
        if len(cr) > 1:
            ncluster_total = 0
            for aa, cc in enumerate(cr):
                if aa == 0:
                    cx = new_cluster_properties[cc]['cx']
                    cy = new_cluster_properties[cc]['cy']
                    cz = new_cluster_properties[cc]['cz']
                    IDX = new_cluster_properties[cc]['IDX']
                    nclusters = new_cluster_properties[cc]['nclusters']
                    rname = new_cluster_properties[cc]['rname']
                    regionindex = new_cluster_properties[cc]['regionindex']
                    regionnum = new_cluster_properties[cc]['regionnum']
                    tc = new_region_properties[cc]['tc']
                    tc_sem = new_region_properties[cc]['tc_sem']
                    nruns_per_person = new_region_properties[cc]['nruns_per_person']
                    tsize = new_region_properties[cc]['tsize']
                    # DBname = new_region_properties[cc]['DBname']
                    # DBnum = new_region_properties[cc]['DBnum']
                    prefix = new_region_properties[cc]['prefix']
                    ncluster_total += nclusters
                else:
                    cx2 = new_cluster_properties[cc]['cx']
                    cy2 = new_cluster_properties[cc]['cy']
                    cz2 = new_cluster_properties[cc]['cz']
                    IDX2 = new_cluster_properties[cc]['IDX']
                    nclusters2 = new_cluster_properties[cc]['nclusters']
                    tc2 = new_region_properties[cc]['tc']
                    tc_sem2 = new_region_properties[cc]['tc_sem']

                    cx = np.concatenate((cx,cx2),axis=0)
                    cy = np.concatenate((cy,cy2),axis=0)
                    cz = np.concatenate((cz,cz2),axis=0)
                    IDX = np.concatenate((IDX,IDX2+ncluster_total),axis=0)
                    ncluster_total += nclusters2
                    tc = np.concatenate((tc,tc2),axis=0)
                    tc_sem = np.concatenate((tc,tc_sem2),axis=0)

            clusterdef_entry_temp = {'cx': cx, 'cy': cy, 'cz': cz, 'IDX': IDX, 'nclusters': ncluster_total, 'rname': rname, 'regionindex': regionindex, 'regionnum': regionnum}
            # Oct 8 2025 - removed 'DBname':DBname, 'DBnum':DBnum from regiondata_entry_temp
            regiondata_entry_temp = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize, 'rname': rname, 'prefix': prefix}

            new_cluster_properties2.append(clusterdef_entry_temp)
            new_region_properties2.append(regiondata_entry_temp)
        else:
            new_cluster_properties2.append(new_cluster_properties[cr[0]])
            new_region_properties2.append(new_region_properties[cr[0]])

        new_cluster_properties = new_cluster_properties2
        new_region_properties = new_region_properties2

    # update region_properties
    rnamelist = [region_properties[xx]['rname'] for xx in range(len(region_properties))]
    if regionname in rnamelist:
        c = np.where(rnamelist == regionname)[0]
        region_properties[c] = copy.deepcopy(new_region_properties[0])
        cluster_properties[c] = copy.deepcopy(new_cluster_properties[0])
    else:
        region_properties.append(new_region_properties[0])
        cluster_properties.append(new_cluster_properties[0])

    print('cluster definition complete.')
    return cluster_properties, region_properties


# def load_cluster_data     (cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
def load_cluster_data(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
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
    network, ncluster_list, sem_region_list = load_network_model(networkmodel, exclude_latent = True)

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
    allregiondata = load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all)
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
        IDX = copy.deepcopy(region_coordinate_list[nn]['IDX'])
        n1 = copy.deepcopy(region_start[nn])
        n2 = copy.deepcopy(region_end[nn])

        print('{} IDX values, {} voxels:  span  x: {} to {}, y {} to {}, z {} to {}'.format(len(IDX), nvox, np.min(cx), np.max(cx),
                                        np.min(cy), np.max(cy), np.min(cz), np.max(cz)))

        regiondata = copy.deepcopy(allregiondata[n1:n2,:])

        #-----------------check for high variance------------
        tsize = int(ts/nruns_total)
        print('ts = {}  nruns_total = {}   tsize = {}  nvox = {}'.format(ts,nruns_total, ts/nruns_total, nvox))
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

        # #------------replace in define_cluster_and_load_data function---------------
        # IDX = copy.deepcopy(region_coordinate_list[nn]['IDX'])
        # #------------end of replace in define_cluster_and_load_data function--------

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

        # check for bad data sets
        check = np.sum(tcr**2, axis = 2)
        cbad, rbad = np.where(check == 0)
        if len(cbad) > 0:
            print('rbad = {}'.format(rbad))
            print('size of DBnum = {}'.format(np.shape(DBnum)))
            print('\n-----------------------------------------------------------')
            print('pyclustering, load_cluster_data:')
            print('possible bad data sets in database numbers {}'.format(DBnum[rbad]))
            print('    ... values are all zeros')
            print('-----------------------------------------------------------\n')

        clusterdef_entry = {'cx':cx, 'cy':cy, 'cz':cz,'IDX':IDX, 'nclusters':nclusters, 'rname':rname, 'regionindex':regionindex, 'regionnum':regionnum, 'occurrence':occurrence}
        # Oct 8 2025 - removed 'DBname':DBname, 'DBnum':DBnum from regiondata_entry
        regiondata_entry = {'tc':tc, 'tc_sem':tc_sem, 'tc_original':tc_original, 'tc_sem_original':tc_sem_original, 'nruns_per_person':nruns_per_person, 'tsize':tsize, 'rname':rname, 'prefix':prefix, 'occurrence':occurrence}
        region_properties.append(regiondata_entry)
        cluster_properties.append(clusterdef_entry)

    print('loading cluster data complete.')
    return region_properties



# update_oneregion_and_load_data
# def load_cluster_data     (cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
def load_oneregion_cluster_data(cluster_properties, region_properties, DBname, DBnum, prefix, nvolmask, regionname, varcheckmethod = 'median', varcheckthresh = 3.0):
    '''
    Function to load data from a group, based on previous cluster definitions.
    define_clusters_and_load_data in pyclustering.py
    region_properties = load_cluster_data(DBname, DBnum, prefix, networkmodel)
    :param cluster_properties:  the cluster definition file
    :param region_properties:  the existing cluster data to be updated
    :param DBname:  name of the database file (probably an excel file)
    :param DBnum:   list of the database entry numbers to use
    :param prefix:   prefix of the nifti format image files to read (indicates the preprocessing
                        steps that have been applied)
    :param regionname:  the region to be load/updated
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
    # network, ncluster_list, sem_region_list = load_network_model(networkmodel, exclude_latent = True)

    #-------------------replace section of cluster definition function-----------------------
    # identify the voxels in the regions of interest
    new_region_properties = []
    region_coordinate_list = []
    region_start = []
    region_end = []
    ncluster_list2 = []
    vox_count = 0
    # for nn, rname in enumerate(sem_region_list):

    rnamelist = [cluster_properties[xx]['rname'] for xx in range(len(cluster_properties))]
    nn = np.where(rnamelist == regionname)[0]
    print('loading data for region {}'.format(regionname))

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

    cx_all = copy.deepcopy(cx)
    cy_all = copy.deepcopy(cy)
    cz_all = copy.deepcopy(cz)

    # -------------------end of part that replaced cluster definition function---------------------

    mode = 'concatenate'
    allregiondata = load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all)
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
        IDX = copy.deepcopy(region_coordinate_list[nn]['IDX'])
        n1 = copy.deepcopy(region_start[nn])
        n2 = copy.deepcopy(region_end[nn])

        print('{} IDX values, {} voxels:  span  x: {} to {}, y {} to {}, z {} to {}'.format(len(IDX), nvox, np.min(cx), np.max(cx),
                                        np.min(cy), np.max(cy), np.min(cz), np.max(cz)))

        regiondata = copy.deepcopy(allregiondata[n1:n2,:])

        #-----------------check for high variance------------
        tsize = int(ts/nruns_total)
        print('ts = {}  nruns_total = {}   tsize = {}  nvox = {}'.format(ts,nruns_total, ts/nruns_total, nvox))
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

        # #------------replace in define_cluster_and_load_data function---------------
        # IDX = copy.deepcopy(region_coordinate_list[nn]['IDX'])
        # #------------end of replace in define_cluster_and_load_data function--------

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
            for bb in range(nruns_total):
                cc2 = [xx for xx in range(len(IDX)) if (varcheck2[xx,bb] <= varlimit) and (IDX[xx] == aa)]
                nvox2 = len(cc2)
                if nvox2 > 0:
                    tcr[aa,bb,:] = np.mean(rdtemp[cc2, bb, :], axis=0)
                    tcr_sem[aa,:] = np.std(rdtemp[cc2, bb, :], axis=0)/np.sqrt(nvox2)
        tc = np.reshape(tcr,(nclusters,ts))
        tc_sem = np.reshape(tcr_sem,(nclusters,ts))

        # check for bad data sets
        check = np.sum(tcr**2, axis = 2)
        cbad, rbad = np.where(check == 0)
        if len(cbad) > 0:
            print('rbad = {}'.format(rbad))
            print('size of DBnum = {}'.format(np.shape(DBnum)))
            print('\n-----------------------------------------------------------')
            print('pyclustering, load_cluster_data:')
            print('possible bad data sets in database numbers {}'.format(DBnum[rbad]))
            print('    ... values are all zeros')
            print('-----------------------------------------------------------\n')

        # clusterdef_entry = {'cx':cx, 'cy':cy, 'cz':cz,'IDX':IDX, 'nclusters':nclusters, 'rname':rname, 'regionindex':regionindex, 'regionnum':regionnum, 'occurrence':occurrence}
        # Oct 8 2025 - removed 'DBname':DBname, 'DBnum':DBnum from regiondata_entry
        regiondata_entry = {'tc':tc, 'tc_sem':tc_sem, 'tc_original':tc_original, 'tc_sem_original':tc_sem_original, 'nruns_per_person':nruns_per_person, 'tsize':tsize, 'rname':rname, 'prefix':prefix, 'occurrence':occurrence}
        new_region_properties.append(regiondata_entry)

    # update region_properties
    rnamelist = [region_properties[xx]['rname'] for xx in range(len(region_properties))]
    if regionname in rnamelist:
        c = np.where(rnamelist == regionname)[0]
        region_properties[c] = copy.deepcopy(new_region_properties[0])
    else:
        region_properties.append(new_region_properties[0])

    print('loading cluster data complete.')
    return region_properties



def load_cluster_data_original(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod = 'median', varcheckthresh = 3.0):
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

            # Oct 8 2025 - removed 'DBname':DBname, 'DBnum':DBnum from regiondata_entry
            regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize, 'rname':rname, 'prefix':prefix}
            region_properties.append(regiondata_entry)

    return region_properties

#
# def load_cluster_data_newmethod(cluster_properties, DBname, DBnum, prefix, nvolmask, networkmodel, varcheckmethod='median',
#                       varcheckthresh=3.0):
#     '''
#     Function to load data from a group, using a predefined cluster definition
#     load_cluster_data in pyclustering.py
#     region_properties = load_cluster_data(cluster_properties, DBname, DBnum, prefix, networkmodel, template_img, ...
#                                         regionmap_img, anatlabels)
#     :param cluster_properties:  cluster definition data, created in define_clusters_and_load_data
#     :param DBname:  name of the database file (probably an excel file)
#     :param DBnum:   list of the database entry numbers to use
#     :param prefix:   prefix of the nifti format image files to read (indicates the preprocessing
#                         steps that have been applied)
#     :param networkmodel:  the network definition file name (probably an excel file)
#     :return:  output is cluster_properties, region_properties
#             cluster_properties is an array of dictionaries (one entry per cluster),
#                 with keys cx, cy, cz, IDX, nclusters, rname, regionindex, regionnum
#                 cx, cy, cz are the 3D coordinates of the voxels in each cluster
#                 IDX is the list of cluster labels for each voxel
#                 nclusters is the number of clusters for the region
#                 rname is the region name, regionindex in the index in the anat template definition, and
#                     regionnum is the number index used to indicate voxels in the anat template
#             region_properties is an array of dictionaries (one entry per cluster),
#                         with keys: tc, tc_sem, nruns_per_person, and tsize
#                         tc and tc_sem are the average and standard error of the time-course for each cluster
#                         nruns_per_person lists the number of data sets that are concatenated per person
#                         tsize is the number of time points in each individual fMRI run
#
#     Modified Dec 2023 to run faster, by collecting data from all voxels at a time from each set of data
#     and sorting the data into regions afterward.
#     '''
#
#     #  data will be concatenated across the entire group, with all runs in each person:  tsize  = N x ts
#     # need to save data on how much runs are loaded per person
#     # mode = 'concatenate_group'
#     # # nvolmask is similar to removing the initial volumes, except this is the number of volumes that are replaced
#     # # by a later volume, so that the effects of the initial volumes are not present, but the total number of volumes
#     # # has not changed
#     # nvolmask = 2
#     # print('load_cluster_data:  DBname = ', DBname)
#     # group_data = GLMfit.compile_data_sets(DBname, DBnum, prefix, mode, nvolmask)  # from GLMfit.py
#     # # group_data is now xs, ys, zs, tsize
#     # xs, ys, zs, ts = group_data.shape
#
#     # the voxels in the regions of interest need to be extracted
#     filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list',
#                                                                               separate_conditions=True)
#     nruns_per_person = np.zeros(NP).astype(int)
#     for nn in range(NP):
#         nruns_per_person[nn] = len(filename_list[nn])
#     nruns_total = np.sum(nruns_per_person)
#
#     # load information about the network
#     network, ncluster_list, sem_region_list = load_network_model(networkmodel, exclude_latent=True)
#
#     # identify the voxels in the regions of interest
#     cx_all, cy_all, cz_all, IDX_all, nclusters_all = [], [], [], [], []
#     for nn, rname in enumerate(sem_region_list):
#         rname_check = cluster_properties[nn]['rname']
#         regionindex = cluster_properties[nn]['regionindex']
#         regionnum = cluster_properties[nn]['regionnum']
#
#         if nn == 0:
#             cx_all = copy.deepcopy(cluster_properties[nn]['cx'])
#             cy_all = copy.deepcopy(cluster_properties[nn]['cy'])
#             cz_all = copy.deepcopy(cluster_properties[nn]['cz'])
#             IDX_all = copy.deepcopy(cluster_properties[nn]['IDX'])
#         else:
#             cx_all = np.concatenate((cx_all,cluster_properties[nn]['cx']),axis=0)
#             cy_all = np.concatenate((cy_all,cluster_properties[nn]['cy']),axis=0)
#             cz_all = np.concatenate((cz_all,cluster_properties[nn]['cz']),axis=0)
#             IDX_all = np.concatenate((IDX_all,cluster_properties[nn]['IDX']),axis=0)
#
#     cx_all = np.array(cx_all)
#     cy_all = np.array(cy_all)
#     cz_all = np.array(cz_all)
#     IDX_all = np.array(IDX_all)
#
#     print('shape of cx_all = {}'.format(np.shape(cx_all)))
#
#     print('loading data for {} voxels from {} data sets'.format(len(cx_all), len(filename_list)))
#     mode = 'concatenate'
#     regiondata_all = load_data_from_region(filename_list, nvolmask, mode, cx_all, cy_all, cz_all)
#     nvox_all, ts = np.shape(regiondata_all)
#     print('finished loading data ...')
#
#     region_properties = []
#     voxel_count = 0
#     for nn, rname in enumerate(sem_region_list):
#         rname_check = cluster_properties[nn]['rname']
#         regionindex = cluster_properties[nn]['regionindex']
#         regionnum = cluster_properties[nn]['regionnum']
#         cx = cluster_properties[nn]['cx']
#         cy = cluster_properties[nn]['cy']
#         cz = cluster_properties[nn]['cz']
#         IDX = cluster_properties[nn]['IDX']
#         nclusters = cluster_properties[nn]['nclusters']
#         v1, v2 = voxel_count, voxel_count + len(cx)
#
#         if rname_check != rname:
#             print('Problem with inconsistent cluster and network definitions!')
#             return region_properties  # to this point region_properties will be incomplete
#         else:
#             regiondata = copy.deepcopy(regiondata_all[v1:v2,:])
#             nvox, ts = np.shape(regiondata)
#
#             # -----------------check for extreme variance------------
#             tsize = int(ts / nruns_total)
#             nvox = len(cx)
#             rdtemp = regiondata.reshape(nvox, tsize, nruns_total, order='F').copy()
#             varcheck2 = np.var(rdtemp, axis=1)
#
#             if varcheckmethod == 'median':
#                 typicalvar2 = np.median(varcheck2)
#             else:
#                 typicalvar2 = np.mean(varcheck2)
#             varlimit = varcheckthresh * typicalvar2
#
#             cv, cp = np.where(varcheck2 > varlimit)  # voxels with crazy variance
#             if len(cv) > 0:
#                 for vv in range(len(cv)):
#                     rdtemp[cv[vv], :, cp[vv]] = np.zeros(tsize)
#                 print('---------------!!!!!----------------------');
#                 print('Variance check found {} crazy voxels'.format(len(cv)))
#                 print('---------------!!!!!----------------------\n');
#             else:
#                 print('Variance check did not find any crazy voxels');
#             regiondata = rdtemp.reshape(nvox, ts, order='F').copy()
#             # ------------done correcting for crazy variance - -------------------
#
#             tc = np.zeros([nclusters, ts])
#             tc_sem = np.zeros([nclusters, ts])
#             for aa in range(nclusters):
#                 cc = [i for i in range(len(IDX)) if IDX[i] == aa]
#                 nvox = len(cc)
#                 tc[aa, :] = np.mean(regiondata[cc, :], axis=0)
#                 tc_sem[aa, :] = np.std(regiondata[cc, :], axis=0) / np.sqrt(nvox)
#
#             regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,
#                                 'rname': rname, 'DBname': DBname, 'DBnum': DBnum, 'prefix': prefix}
#             region_properties.append(regiondata_entry)
#
#     return region_properties


def load_data_from_region(filename_list, nvolmask, mode, cx, cy, cz, definingclusters = False):
    # timespantoload = 30

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
            # if definingclusters:
            #     roi_data = input_data[cx,cy,cz,:timespantoload]   # check the size of this
            # else:
            #     roi_data = input_data[cx,cy,cz,:]   # check the size of this

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


def load_all_voxel_data(DBname, DBnum, prefix, normtemplatename, nametag):
    # October 2025
    # This function is not used in Pantheon, it is a special case
    # This function is to load all of the data from every voxel within a region mask
    # in order to check for voxel-to-region correlations later. The purpose is to
    # check anatomical region maps
    #
    outputdir, f = os.path.split(DBname)
    outputfname = '{}_allvoxel_data.npy'.format(nametag)
    outputname = os.path.join(outputdir, outputfname)

    # the voxels in the regions of interest need to be extracted
    filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, prefix, mode='list')
    nruns_per_person = np.zeros(NP).astype(int)
    for nn in range(NP):
        nruns_per_person[nn] = len(filename_list[nn])
    nruns_total = np.sum(nruns_per_person)

    # load templates and region masks
    resolution = 1
    template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_map = \
        load_templates.load_template_and_masks(normtemplatename, resolution)

    if normtemplatename.lower() == 'brain':
        # for brain data, need to match the template, region map, etc., to the data size/position
        dbhome = df1.loc[self.DBnum[0], 'datadir']
        fname = df1.loc[self.DBnum[0], 'niftiname']
        niiname = os.path.join(dbhome, fname)
        fullpath, filename = os.path.split(niiname)
        prefix_niiname = os.path.join(fullpath, self.CLprefix + filename)
        temp_data = nib.load(prefix_niiname)
        img_data_affine = temp_data.affine
        hdr = temp_data.header
        template_img = i3d.convert_affine_matrices_nearest(template_img, template_affine, img_data_affine,
                                                           hdr['dim'][1:4])
        regionmap_img = i3d.convert_affine_matrices_nearest(regionmap_img, template_affine, img_data_affine,
                                                            hdr['dim'][1:4])

    # now load all the voxel data
    cx, cy, cz = np.where(regionmap_img > 0)
    nvolmask = 2

    mode = 'concatenate'
    allregiondata = load_data_from_region(filename_list, nvolmask, mode, cx, cy, cz)
    nvox,ts = np.shape(allregiondata)
    print('nvox = {}   ts = {}'.format(nvox,ts))


    # outputname = r'Y:\BigAnatomicalAnalysis\allpain_allvoxel_data.npy'
    np.save(outputname, {'voxeldata':allregiondata, 'DBname':DBname, 'DBnum':DBnum,
        'cx':cx, 'cy':cy, 'cz':cz, 'regionmap_img':regionmap_img, 'template_img':template_img})

    print('loading voxel data complete.')
    return outputname

