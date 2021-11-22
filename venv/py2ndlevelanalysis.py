# set of programs for 2nd-level analysis of BOLD responses and SEM results, in
# relation to participant characteristics

# 1) need to indicate type of data to use:  a) BOLD responses, b) 1- and 2-source SEM, c) network SEM
# 2) need to indicate which personal characteristics to use, such as painratings, temperatures, age, etc. from DB
# 3) need to indicate type of test; significant group average, correlation, regression, group comparisons, ANOVA, ANCOVA
# 4) need to indicate statistical thresholds
# 5) need to save/display results

# need DBname and DBnum as inputs
# need database entry names for personal characteristics

# import sys
# sys.path.append('C:\\Users\\Stroman\\PycharmProjects\\pyspinalfmri3\\venv')

import numpy as np
from scipy import stats
import pyclustering
import pysem
import copy
import pydisplay
import os
import time
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings

# 2-source SEM results
# save the results somehow
# results = {'type': '2source', 'CCrecord': CCrecord, 'beta2': beta2, 'beta1': beta1, 'Zgrid2': Zgrid2,
#            'Zgrid1_1': Zgrid1_1, 'Zgrid1_2': Zgrid1_2, 'DBname': self.DBname, 'DBnum': self.DBnum,
#            'cluster_properties': cluster_properties}
#
# resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_2source_record_' + self.SEMsavetag + '.npy')
# np.save(resultsrecordname, results)
#
#
# # network SEM results
# # save the results somehow
# results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
#            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
#            'DBnum': self.DBnum}
# resultsrecordname = os.path.join(self.SEMresultsdir, 'SEMresults_network_record_' + self.SEMsavetag + '.npy')
# np.save(resultsrecordname, results)
#
#
# # BOLD response data
# region_data = np.load(self.SEMregionname, allow_pickle=True).flat[0]
# region_properties = region_data['region_properties']
# # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize'])
#
# cluster_data = np.load(self.SEMclustername, allow_pickle=True).flat[0]
# cluster_properties = cluster_data['cluster_properties']
# # dict_keys(['cx', 'cy', 'cz', 'IDX', 'nclusters', 'rname', 'regionindex', 'regionnum'])


def get_cluster_info(namelist, nclusterlist, number):
    regionend = np.cumsum(nclusterlist)
    cc = np.where(regionend > number)[0]
    regionnum = cc[0]
    if regionnum == 0:
        clusternum = number
    else:
        clusternum = (number - regionend[regionnum-1]).astype(int)
    regionname = namelist[regionnum]
    return regionname, clusternum, regionnum


def get_cluster_position_details(cluster_properties):
    nregions = len(cluster_properties)
    cluster_info = []
    rname_list = []
    ncluster_list = np.zeros(nregions)
    for ii in range(nregions):
        cx = cluster_properties[ii]['cx']
        cy = cluster_properties[ii]['cy']
        cz = cluster_properties[ii]['cz']
        IDX = cluster_properties[ii]['IDX']
        nclusters = cluster_properties[ii]['nclusters']
        ncluster_list[ii] = nclusters
        rname = cluster_properties[ii]['rname']
        rname_list.append(rname)
        regionnum = cluster_properties[ii]['regionnum']
        regionindex = cluster_properties[ii]['regionindex']
        regionlimits = [np.min(cx), np.max(cx), np.min(cy), np.max(cy), np.min(cz), np.max(cz)]
        cluster_coords = np.zeros((nclusters,3))
        for nn in range(nclusters):
            cc = np.where(IDX == nn)[0]
            if len(cc) > 0:
                x0 = np.mean(cx[cc])
                y0 = np.mean(cy[cc])
                z0 = np.mean(cz[cc])
            else:
                x0 = 0
                y0 = 0
                z0 = 0
            cluster_coords[nn,:] = x0,y0,z0
        entry = {'rname':rname, 'nclusters':nclusters, 'regionnum':regionnum, 'regionlimits':regionlimits, 'cluster_coords':cluster_coords}
        cluster_info.append(entry)
    return cluster_info, rname_list, ncluster_list


def remove_reps_and_sort(id_list, value_list, data):
    # eliminate redundant values, for repeats keep the one with the largest value
    uid, indices = np.unique(id_list, return_index=True)
    keep_indices = []
    for cc in uid:
        indices2 = np.where(id_list == cc)[0]
        vindex = np.argmax(np.abs(value_list[indices2]))
        keep_indices.append(indices2[vindex])
    data2 = []
    value_list2 = []
    for cc in keep_indices:
        data2.append(data[cc])
        value_list2.append(value_list[cc])
    data = copy.deepcopy(data2)
    value_list = value_list2
    data2 = []
    value_list2 = []

    # sort by significance
    dorder = np.argsort(np.abs(value_list))
    data2 = copy.deepcopy(data)
    value_list2 = np.zeros(len(dorder))
    for ii, dd in enumerate(dorder):
        data2[ii] = copy.deepcopy(data[dd])
        value_list2[ii] = copy.deepcopy(value_list[dd])

    return data2, value_list2

def generate_output_name(commontag_input, filename1, filename2, tag, extension):
    # prep output name
    p1, f1 = os.path.split(filename1)
    f1b, e1 = os.path.splitext(f1)
    if len(filename2) == 0:
        outputname = os.path.join(p1, f1b + tag + extension)
    else:
        p2, f2 = os.path.split(filename2)
        f2b, e2 = os.path.splitext(f2)

        maxlength = np.max([len(f1b), len(f2b)])
        runsearch = True
        pos = 0
        while runsearch and pos < maxlength:
            if f1b[pos] == f2b[pos]:
                pos += 1
            else:
                runsearch = False
        commontag = f1b[:pos]
        utag1 = f1b[pos:]
        utag2 = f2b[pos:]

        if len(commontag_input) > 0:
            commontag = commontag_input
        outputname = os.path.join(p1, commontag + utag1 + '_' + utag2 + tag + extension)
    return outputname


def GLMregression(data, covariates, axis):
    # function to do GLM regression w.r.t. covariates
    # the covariates will be set to each have a mean value of zero
    # the data will also be set to have a mean value of zero, along the axis of interest
    # and there must be one covariate value for each data point along the axis of interest
    # for example, if the data size is a x b x c, and axis = 1, then
    # the covariates must have size n x b, where n is the number of different types of covariates
    ndim = np.ndim(data)
    data2 = np.moveaxis(data,axis,-1)   # move axis of interest to the end
    dsize = np.shape(data2)

    mdata2 = np.mean(data2,axis=-1)
    data2a = data2 - np.repeat(np.expand_dims(mdata2,axis=-1),dsize[-1],axis =-1)

    varmask = np.var(data2a,axis = -1) > 0   # flag data with constant values

    # set the covariates to have average values = 0
    nc,NP = np.shape(covariates)
    mcov = np.mean(covariates,axis = -1)
    covariates2 = covariates - np.repeat(np.expand_dims(mcov,axis=-1),NP,axis =-1)

    # data2a = b @ G    where data2 has size a x b x c,  G has size n x c,  b has size a x b x n

    # need to add a constant term to G to allow for non-zero average values in data2
    G = np.ones((nc+1,NP))
    G[:nc,:] = covariates2

    iGG = np.linalg.inv(G @ G.T)
    b = data2 @ G.T @ iGG
    fit = b @ G

    ssq = np.sum(data2**2,axis = -1)
    ssqa = np.sum(data2a**2,axis = -1)
    residual_ssq = np.sum((data2-fit)**2,axis = -1)

    # bsem = sqrt(abs(contrast'*iGG*contrast)*err2);
    err2 = residual_ssq/NP
    iGGdiags = np.abs(np.diagonal(iGG))
    for nn in range(nc+1):
        bsem1 = np.sqrt(iGGdiags[nn]*err2)
        bsem1 = np.expand_dims(bsem1,axis=-1)
        if nn == 0:
            bsem = bsem1
        else:
            bsem = np.concatenate((bsem,bsem1),axis = -1)

    tol = 1.0e-10
    R2 = 1.0 - residual_ssq/(ssqa + tol)   # determine how much variance is explained, not including the average offset
    R2 = R2*varmask
    R2[R2 >= 1.0] = 1.0-tol
    R2[R2 <= -1.0] = -1.0+tol
    Z = np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(NP-3)

    # correlation
    for nn in range(nc):
        cv = covariates2[nn,:]
        cv = np.reshape(cv, ([1]*(ndim-1) + [NP]))
        cv2 = np.tile(cv,(np.concatenate((dsize[:-1],[1]))))

        ssq_dc = np.sum(data2a*cv2, axis=-1)
        ssq_c = np.sum(cv2**2, axis=-1)
        CC = ssq_dc/(np.sqrt(ssqa*ssq_c) + tol)

        if nc > 1:
            CC2 = np.reshape(CC, (np.concatenate((np.shape(CC),[1]))))
            if nn == 0:
                R = CC2
            else:
                R = np.concatenate((R,CC2),axis=-1)
        else:
            R = CC

    R[R >=1.0] = 1.0-tol
    R[R <= -1.0] = -1.0+tol
    Rcorrelation = R
    Zcorrelation = np.arctanh(R) * np.sqrt(NP - 3)

    return b, bsem, R2, Z, Rcorrelation, Zcorrelation


#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# look for significant group-average beta-value differences from zero
def group_significance(filename, pthreshold, statstype = 'average', covariates = 'none', covnames = 'none'):
    # test fMRI results for 1) significant differences from zero (statstype = 'average')
    # or regression (statstype = 'regression') or correlations (statstype = 'correlation') with covariates
    #
    print('running py2ndlevelanalysis:  group_significance:  started at ',time.ctime())
    data = np.load(filename, allow_pickle=True).flat[0]

    # setup output name
    tag = ''
    if covnames != 'none':
        tag = ''
        for tname in covnames:  tag += tname+' '
        tag = '_' + tag[:-1]  # take off the trailing space
    excelfilename = generate_output_name('Group_',filename, '', tag, '.xlsx')

    datafiletype = 0
    try:
        keylist = list(data.keys())
        if 'type' in keylist: datafiletype = 1   # SEM results
        if 'region_properties' in keylist: datafiletype = 2    # BOLD time-course data
        print('group_significance:  datafiletype = ', datafiletype)
    except:
        print('group_significance:  input file does not appear to have the correct data structure')
        return 0

    if datafiletype == 1:
        semtype = data['type']
        print('SEM results loaded:  type ',semtype)

        if semtype == '2source':
            cluster_properties = data['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            beta1 = data['beta1']
            beta2 = data['beta2']

            ntclusters, ns1sclusters, ns2clusters, ntimepoints, NP, nbeta = np.shape(beta1)

            # stats based on group average ----------
            if statstype == 'average':
                stattitle = 'Tvalue'
                # stats based on group average - sig diff from zero?
                mean_beta1 = np.mean(beta1,axis = 4)
                se_beta1 = np.std(beta1,axis = 4)/np.sqrt(NP)
                Tbeta1 = mean_beta1/(se_beta1 + 1.0e-10)

                mean_beta2 = np.mean(beta2,axis = 4)
                se_beta2 = np.std(beta2,axis = 4)/np.sqrt(NP)
                Tbeta2 = mean_beta2/(se_beta2 + 1.0e-10)

                Tthresh = stats.t.ppf(1-pthreshold,NP-1)

                beta1_sig = np.abs(Tbeta1) > Tthresh
                beta2_sig = np.abs(Tbeta2) > Tthresh
                stat_of_interest1 = Tbeta1
                stat_of_interest2 = Tbeta2

            # stats based on regression with covariates - --------
            if statstype == 'regression':
                stattitle = 'Zregression'
                Zthresh = stats.norm.ppf(1-pthreshold)

                terms, NPt = np.shape(covariates)  # need one term per person, for each covariate
                b1, b1sem, R21, Z1, Rcorrelation1, Zcorrelation1 = GLMregression(beta1, covariates, 4)
                b2, b2sem, R22,Z2, Rcorrelation2, Zcorrelation2 = GLMregression(beta2, covariates, 4)

                beta1_sig = np.abs(Z1) > Zthresh
                beta2_sig = np.abs(Z2) > Zthresh
                stat_of_interest1 = Z1
                stat_of_interest2 = Z2

            #---------------------------------------------------
            if statstype == 'correlation':
                stattitle = 'Zcorr'
                Zthresh = stats.norm.ppf(1-pthreshold)

                terms, NPt = np.shape(covariates)  # need one term per person, for each covariate
                b1, b1sem, R21, Z1, Rcorrelation1, Zcorrelation1 = GLMregression(beta1, covariates, 4)
                b2, b2sem, R22,Z2, Rcorrelation2, Zcorrelation2 = GLMregression(beta2, covariates, 4)

                beta1_sig = np.abs(Zcorrelation1) > Zthresh
                beta2_sig = np.abs(Zcorrelation2) > Zthresh
                stat_of_interest1 = Zcorrelation1
                stat_of_interest2 = Zcorrelation2
            #---------------------------------------------------

            keys = ['tname', 'tcluster', 'sname', 'scluster', stattitle, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2', 'tlimy1',
                    'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2', 'slimz1', 'slimz2',
                    't','s1','s2','tt','nb']

            # write out significant results, based on beta1------------------------------
            # if np.ndim(beta1_sig) < 6:  # allow for different forms of results (some have multiple stats terms)
            #     beta1_sig = np.expand_dims(beta1_sig, axis=-1)
            #     stat_of_interest1 = np.expand_dims(stat_of_interest1, axis=-1)

            for tt in range(ntimepoints):
                results = []
                sig_temp = beta1_sig[:,:,:,tt,:]
                t,s1,s2,nb = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    Svalue_list[ii] = stat_of_interest1[t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']
                    connection_info = [t[ii],s1[ii],s2[ii],tt,nb[ii]]

                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits),connection_info))
                    entry = dict(zip(keys, values))
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                if len(results) > 0:
                    print('removing redundant values ...')
                    results2, Svalue_list2 = remove_reps_and_sort(connid_list, Svalue_list, results)

                    excelsheetname = '2S beta1 ' + statstype + ' ' + str(tt)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
                else:
                    results2 = []
                    print('no significant results found at p < {}'.format(pthreshold))

            results_beta1 = results2

            # now, write out significant results, based on beta2-------------------------
            # if np.ndim(beta2_sig) < 6:  # allow for different forms of results (some have multiple stats terms)
            #     beta2_sig = np.expand_dims(beta2_sig, axis=-1)
            #     stat_of_interest2 = np.expand_dims(stat_of_interest2, axis=-1)

            for tt in range(ntimepoints):
                results = []
                sig_temp = beta2_sig[:,:,:,tt,:]
                t,s1,s2,nb = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    Svalue_list[ii] = stat_of_interest2[t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    connection_info = [t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits),connection_info))
                    entry = dict(zip(keys, values))

                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                if len(results) > 0:
                    print('removing redundant values ...')
                    results2, Svalue_list2 = remove_reps_and_sort(connid_list, Svalue_list, results)

                    excelsheetname = '2S beta2 ' + statstype + ' ' + str(tt)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
                else:
                    results2 = []
                    print('no significant results found at p < {}'.format(pthreshold))

            results_beta2 = results2

            return excelfilename


        if semtype == 'network':
            # results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
            #            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
            #            'DBnum': self.DBnum}

            resultsnames = data['resultsnames']
            clustername = data['clustername']
            regionname = data['regionname']
            networkmodel = data['network']
            network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)
            nclusterlist = np.array([ncluster_list[i]['nclusters'] for i in range(len(ncluster_list))])

            cluster_data = np.load(clustername, allow_pickle=True).flat[0]
            cluster_properties = cluster_data['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            results = []
            Svalue_list = []
            connid_list = []   # identify connections - to be able to remove redundant ones later
            for networkcomponent, fname in enumerate(resultsnames):
                print('analyzing network component: {}'.format(fname))
                semresults = np.load(fname, allow_pickle=True).flat[0]
                sem_one_target = semresults['sem_one_target_results']
                ntclusters = len(sem_one_target)

                target = network[networkcomponent]['target']
                sources = network[networkcomponent]['sources']
                targetnum = network[networkcomponent]['targetnum']
                sourcenums = network[networkcomponent]['sourcenums']
                targetname = cluster_info[targetnum]['rname']
                targetlimits = cluster_info[targetnum]['regionlimits']

                for tt in range(ntclusters):
                    targetcoords = cluster_info[targetnum]['cluster_coords'][tt, :]
                    beta = sem_one_target[tt]['b']
                    ncombinations, ntimepoints, NP, nsources = np.shape(beta)

                    # stats based on group average - --------------------
                    if statstype == 'average':
                        stattitle = 'Tvalue'
                        mean_beta = np.mean(beta, axis=2)
                        se_beta = np.std(beta, axis=2) / np.sqrt(NP)
                        Tbeta = mean_beta / (se_beta + 1.0e-10)
                        Tthresh = stats.t.ppf(1 - pthreshold, NP - 1)
                        beta_sig = np.abs(Tbeta) > Tthresh    # size is ncombinations x ntimepoints x nsources
                        stat_of_interest = Tbeta

                    # stats based on regression with covariates - --------
                    if statstype == 'regression':
                        stattitle = 'Zregression'
                        Zthresh = stats.norm.ppf(1 - pthreshold)
                        terms, NPt = np.shape(covariates)  # need one term per person, for each covariate
                        b, bsem, R2, Z, Rcorrelation, Zcorrelation = GLMregression(beta, covariates, 2)
                        beta_sig = np.abs(Z) > Zthresh
                        stat_of_interest = Z

                    # stats based on regression with covariates - --------
                    if statstype == 'correlation':
                        stattitle = 'Zcorr'
                        Zthresh = stats.norm.ppf(1 - pthreshold)
                        terms, NPt = np.shape(covariates)  # need one term per person, for each covariate
                        b, bsem, R2, Z, Rcorrelation, Zcorrelation = GLMregression(beta, covariates, 2)
                        beta_sig = np.abs(Zcorrelation) > Zthresh
                        stat_of_interest = Zcorrelation
                    #---------------------------------------------------

                    keys = ['tname', 'tcluster', 'sname', 'scluster', stattitle, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2',
                            'tlimy1',
                            'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2',
                            'slimz1', 'slimz2', 'networkcomponent', 'tt', 'combo','timepoint','ss']

                    # organize significant results
                    # if np.ndim(beta_sig) < 4:  # allow for different forms of results (some have multiple stats terms)
                    #     stat_of_interest = np.expand_dims(stat_of_interest, axis=-1)
                    #     beta_sig = np.expand_dims(beta_sig, axis=-1)
                    print('group_significance:  size of beta_sig is ',np.shape(beta_sig))
                    combo, nt, ss = np.where(beta_sig)   # significant connections during this time period

                    # cc = 0   # this value is always zero
                    for ii in range(len(combo)):
                        # get region names, cluster numbers, etc.
                        Svalue = stat_of_interest[combo[ii], nt[ii], ss[ii]]
                        timepoint = nt[ii]
                        sourcename = cluster_info[sourcenums[ss[ii]]]['rname']
                        mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums],combo[ii]).astype(int)   # cluster number for each source
                        sourcecluster = mlist[ss[ii]]
                        sourcecoords = cluster_info[sourcenums[ss[ii]]]['cluster_coords'][sourcecluster, :]
                        sourcelimits = cluster_info[sourcenums[ss[ii]]]['regionlimits']

                        connid = nt[ii]*1e7 + targetnum*1e5 + tt*1e3 + sourcenums[ss[ii]]*10 + sourcecluster

                        connection_info = [networkcomponent, tt, combo[ii], nt[ii], ss[ii]]
                        values = np.concatenate(([targetname, tt, sourcename, sourcecluster, Svalue],
                             list(targetcoords), list(targetlimits), list(sourcecoords), list(sourcelimits), connection_info))

                        entry = dict(zip(keys, values))

                        results.append(entry)
                        Svalue_list.append(Svalue)
                        connid_list.append(connid)

            # eliminate redundant values, for repeats keep the one with the largest Tvalue
            if len(results) > 0:
                print('removing redundant values ...')
                results2, Svalue_list2 = remove_reps_and_sort(np.array(connid_list), np.array(Svalue_list), results)

                # separate by timepoints
                timepoint_list = [int(results2[ii]['timepoint']) for ii in range(len(results2))]
                times = np.unique(timepoint_list)
                print('time point values: ',times)

                # still need to split the data according to timepoints
                print('separating values from different time periods...')
                for timepoint in times:
                    indices = np.where(timepoint_list == timepoint)[0]
                    print('timepoint: {}, found {} entries'.format(timepoint,len(indices)))
                    results1 = []
                    for ii in indices:
                        results1.append(results2[ii])

                    excelsheetname = 'network ' + statstype + ' ' + str(timepoint)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results1, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
            else:
                print('no significant results found with p < {} '.format(pthreshold))

            return excelfilename

    if datafiletype == 2:
        # analyzing BOLD responses
        region_properties = data['region_properties']
        # regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,'rname': rname}

        nregions = len(region_properties)
        for rr in range(nregions):
            tc = region_properties[rr]['tc']          # nclusters x tsize_total
            tc_sem = region_properties[rr]['tc_sem']
            tsize = region_properties[rr]['tsize']
            nruns_per_person = region_properties[rr]['nruns_per_person']
            rname = region_properties[rr]['rname']
            NP = len(nruns_per_person)
            nclusters, tsize_total = np.shape(tc)

            # change shape of timecourse data array - prep data
            tc_per_person = np.zeros((nclusters,tsize,NP))
            tc_per_person_sem = np.zeros((nclusters,tsize,NP))
            for nn in range(NP):
                nruns = nruns_per_person[nn]
                t1 = np.sum(nruns_per_person[:nn])*tsize
                t2 = np.sum(nruns_per_person[:(nn+1)])*tsize
                tp = list(range(t1,t2))
                tc1 = np.mean(np.reshape(tc[:,tp],(nclusters,tsize,nruns)),axis = 2)
                tc1_sem = np.mean(np.reshape(tc_sem[:,tp],(nclusters,tsize,nruns)),axis = 2)
                tc_per_person[:,:,nn] = tc1
                tc_per_person_sem[:,:,nn] = tc1_sem

            # stats based on group average ---------------------------
            if statstype == 'average':
                mean_tc = np.mean(tc_per_person, axis = 2)
                sem_tc = np.std(tc_per_person, axis = 2)/np.sqrt(NP)
                T = mean_tc/(sem_tc + 1.0e-10)
                Tthresh = stats.t.ppf(1 - pthreshold, NP - 1)
                sig = np.abs(T) > Tthresh

                # check significance and write out results
                keys = []
                for cc in range(nclusters):
                    keys = keys + ['avg ' + str(cc), 'sem ' + str(cc), 'T ' + str(cc), 'sig ' + str(cc)]

                outputdata = []
                for tt in range(tsize):
                    values = []
                    for cc in range(nclusters):
                        values = values + [mean_tc[cc, tt], sem_tc[cc, tt], T[cc, tt], sig[cc, tt]]
                    entry = dict(zip(keys, values))
                    outputdata.append(entry)


            # stats based on regression with covariates ----------
            if statstype == 'regression':
                Zthresh = stats.norm.ppf(1 - pthreshold)
                b, bsem, R2, Z, Rcorrelation, Zcorrelation = GLMregression(tc_per_person, covariates, 2)
                sig = np.abs(Z) > Zthresh

                # check significance and write out results
                ncov,NP = np.shape(covariates)
                covnames = []
                for tn in range(ncov): covnames += ['cov'+str(tn+1)]
                covnames += ['intercept']

                keys = []
                for cc in range(nclusters):
                    for tn in range(ncov+1):
                        keys = keys + ['b_'+covnames[tn] + ' ' + str(cc), 'bsem_'+covnames[tn] + ' ' + str(cc), 'R2_'+covnames[tn] + ' ' + str(cc), 'sig_'+covnames[tn] + ' ' + str(cc)]

                outputdata = []
                for tt in range(tsize):
                    values = []
                    for cc in range(nclusters):
                        for tn in range(ncov+1):
                            values = values + [b[cc, tt, tn], bsem[cc, tt, tn], R2[cc, tt, tn], sig[cc, tt, tn]]
                    entry = dict(zip(keys, values))
                    outputdata.append(entry)


            # stats based on regression with covariates ----------
            if statstype == 'correlation':
                Zthresh = stats.norm.ppf(1 - pthreshold)
                R = np.corrcoef()
                b, bsem, R2, Z, Rcorrelation, Zcorrelation = GLMregression(tc_per_person, covariates, 2)
                sig = np.abs(Zcorrelation) > Zthresh

                # check significance and write out results
                ncov,NP = np.shape(covariates)
                covnames = []
                for tn in range(ncov): covnames += ['cov'+str(tn+1)]
                covnames += ['intercept']

                keys = []
                for cc in range(nclusters):
                    for tn in range(ncov):
                        keys = keys + ['R_'+covnames[tn] + ' ' + str(cc), 'Z_'+covnames[tn] + ' ' + str(cc), 'sig_'+covnames[tn] + ' ' + str(cc)]

                outputdata = []
                for tt in range(tsize):
                    values = []
                    for cc in range(nclusters):
                        for tn in range(ncov):
                            values = values + [Rcorrelation[cc, tt, tn], Zcorrelation[cc, tt, tn], sig[cc, tt, tn]]
                    entry = dict(zip(keys, values))
                    outputdata.append(entry)
            #---------------------------------------------------
            if len(outputdata) > 0:
                excelsheetname = rname
                print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                pydisplay.pywriteexcel(outputdata, excelfilename, excelsheetname, 'append', '%.3f')
                print('finished writing results to ',excelfilename)
            else:
                print('no significant results found at p < {}'.format(pthreshold))
            return excelfilename

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# look for significant group-average beta-value differences from zero
def group_difference_significance(filename1, filename2, pthreshold, mode = 'unpaired', statstype = 'average', covariates = 'none', covnames = 'none'):
    # test fMRI results for 1) significant differences from zero (statstype = 'average')
    # or regression (statstype = 'regression') or correlations (statstype = 'correlation') with covariates
    #
    print('running py2ndlevelanalysis:  group_difference_significance:  started at ',time.ctime())
    data1 = np.load(filename1, allow_pickle=True).flat[0]
    data2 = np.load(filename2, allow_pickle=True).flat[0]

    # setup output name
    tag = ''
    if covnames != 'none':
        tag = ''
        for tname in covnames:  tag += tname+' '
        tag = tag[:-1]  # take off the trailing space
    excelfilename = generate_output_name('Group_',filename1, filename2, tag, '.xlsx')

    datafiletype = 0
    try:
        keylist = list(data1.keys())
        if 'type' in keylist: datafiletype = 1   # SEM results
        if 'region_properties' in keylist: datafiletype = 2    # BOLD time-course data
        print('group_significance:  datafiletype = ', datafiletype)
    except:
        print('group_significance:  input file does not appear to have the correct data structure')
        return 0

    if datafiletype == 1:
        semtype = data1['type']
        print('SEM results loaded:  type ',semtype)

        if semtype == '2source':
            cluster_properties = data1['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            beta1_1 = data1['beta1']
            beta2_1 = data1['beta2']
            beta1_2 = data2['beta1']
            beta2_2 = data2['beta2']

            ntclusters, ns1sclusters, ns2clusters, ntimepoints, NP2, nbeta = np.shape(beta1_2)
            ntclusters, ns1sclusters, ns2clusters, ntimepoints, NP1, nbeta = np.shape(beta1_1)
            print('size of beta1_1 is {}'.format(np.shape(beta1_1)))
            print('size of beta1_2 is {}'.format(np.shape(beta1_2)))

            # stats based on group average ----------
            if statstype == 'average':
                if mode == 'unpaired':
                    stattitle = 'Tvalue unpaired'
                    # stats based on group average - sig diff between groups?
                    mean_beta1_1 = np.mean(beta1_1,axis = 4)
                    var_beta1_1 = np.var(beta1_1,axis = 4)
                    mean_beta1_2 = np.mean(beta1_2,axis = 4)
                    var_beta1_2 = np.var(beta1_2,axis = 4)
                    # pooled standard deviation:
                    sp = np.sqrt( ((NP1-1)*var_beta1_1 + (NP2-1)*var_beta1_2)/(NP1+NP2-2) )

                    Tbeta1 = (mean_beta1_1 - mean_beta1_2)/(sp*np.sqrt(1/NP1 + 1/NP2) + 1.0e-20)

                    # stats based on group average - sig diff between groups?
                    mean_beta2_1 = np.mean(beta2_1,axis = 4)
                    var_beta2_1 = np.var(beta2_1,axis = 4)
                    mean_beta2_2 = np.mean(beta2_2,axis = 4)
                    var_beta2_2 = np.var(beta2_2,axis = 4)
                    # pooled standard deviation:
                    sp = np.sqrt( ((NP1-1)*var_beta2_1 + (NP2-1)*var_beta2_2)/(NP1+NP2-2) )

                    Tbeta2 = (mean_beta2_1 - mean_beta2_2)/(sp*np.sqrt(1/NP1 + 1/NP2) + 1.0e-20)

                    Tthresh = stats.t.ppf(1-pthreshold,NP1+NP2-1)

                    beta1_sig = np.abs(Tbeta1) > Tthresh
                    beta2_sig = np.abs(Tbeta2) > Tthresh
                    stat_of_interest1 = Tbeta1
                    stat_of_interest2 = Tbeta2
                else:
                    stattitle = 'Tvalue paired'
                    # stats based on group average - sig diff between groups?
                    beta_diff = beta1_1-beta1_2
                    mean_beta1_diff = np.mean(beta_diff,axis = 4)
                    sem_beta1_diff = np.std(beta_diff)/np.sqrt(NP1)
                    Tbeta1 = mean_beta1_diff/(sem_beta1_diff + 1.0e-10)

                    # stats based on group average - sig diff between groups?
                    beta_diff = beta2_1-beta2_2
                    mean_beta2_diff = np.mean(beta_diff,axis = 4)
                    sem_beta2_diff = np.std(beta_diff)/np.sqrt(NP1)
                    Tbeta2 = mean_beta2_diff/(sem_beta2_diff + 1.0e-10)

                    Tthresh = stats.t.ppf(1-pthreshold,NP1-1)

                    beta1_sig = np.abs(Tbeta1) > Tthresh
                    beta2_sig = np.abs(Tbeta2) > Tthresh
                    stat_of_interest1 = Tbeta1
                    stat_of_interest2 = Tbeta2

            keys = ['tname', 'tcluster', 'sname', 'scluster', stattitle, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2', 'tlimy1',
                    'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2', 'slimz1', 'slimz2',
                    't','s1','s2','tt','nb']

            # write out significant results, based on beta1------------------------------
            # if np.ndim(beta1_sig) < 6:  # allow for different forms of results (some have multiple stats terms)
            #     beta1_sig = np.expand_dims(beta1_sig, axis=-1)
            #     stat_of_interest1 = np.expand_dims(stat_of_interest1, axis=-1)

            for tt in range(ntimepoints):
                results = []
                sig_temp = beta1_sig[:,:,:,tt,:]
                t,s1,s2,nb = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    Svalue_list[ii] = stat_of_interest1[t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']
                    connection_info = [t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits),connection_info))
                    entry = dict(zip(keys, values))
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                if len(results) > 0:
                    print('removing redundant values ...')
                    results2, Svalue_list2 = remove_reps_and_sort(connid_list, Svalue_list, results)

                    excelsheetname = '2S beta2 ' + statstype + ' ' + str(tt)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
                else:
                    results2 = []
                    print('no significant results found at p < {}'.format(pthreshold))

            results_beta1 = results2

            # now, write out significant results, based on beta2-------------------------
            # if np.ndim(beta2_sig) < 6:  # allow for different forms of results (some have multiple stats terms)
            #     beta2_sig = np.expand_dims(beta2_sig, axis=-1)
            #     stat_of_interest2 = np.expand_dims(stat_of_interest2, axis=-1)

            for tt in range(ntimepoints):
                results = []
                sig_temp = beta2_sig[:,:,:,tt,:]
                t,s1,s2,nb = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    Svalue_list[ii] = stat_of_interest2[t[ii],s1[ii],s2[ii],tt,nb[ii]]
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']
                    connection_info = [t[ii],s1[ii],s2[ii],tt,nb[ii]]

                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits),connection_info))
                    entry = dict(zip(keys, values))
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                if len(results) > 0:
                    print('removing redundant values ...')
                    results2, Svalue_list2 = remove_reps_and_sort(connid_list, Svalue_list, results)

                    excelsheetname = '2S beta2 ' + statstype + ' ' + str(tt)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
                else:
                    results2 = []
                    print('no significant results found at p < {}'.format(pthreshold))

            results_beta2 = results2

            return excelfilename

        if semtype == 'network':
            # results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
            #            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
            #            'DBnum': self.DBnum}

            resultsnames1 = data1['resultsnames']
            resultsnames2 = data2['resultsnames']
            clustername = data1['clustername']
            regionname = data1['regionname']
            networkmodel = data1['network']
            network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)
            nclusterlist = np.array([ncluster_list[i]['nclusters'] for i in range(len(ncluster_list))])

            cluster_data = np.load(clustername, allow_pickle=True).flat[0]
            cluster_properties = cluster_data['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            results = []
            Svalue_list = []
            connid_list = []   # identify connections - to be able to remove redundant ones later
            for networkcomponent, fname1 in enumerate(resultsnames1):
                fname2 = resultsnames2[networkcomponent]
                print('analyzing network component: \n{}\n{}\n'.format(fname1,fname2))
                semresults1 = np.load(fname1, allow_pickle=True).flat[0]
                sem_one_target1 = semresults1['sem_one_target_results']
                semresults2= np.load(fname2, allow_pickle=True).flat[0]
                sem_one_target2 = semresults2['sem_one_target_results']
                ntclusters = len(sem_one_target1)

                target = network[networkcomponent]['target']
                sources = network[networkcomponent]['sources']
                targetnum = network[networkcomponent]['targetnum']
                sourcenums = network[networkcomponent]['sourcenums']
                targetname = cluster_info[targetnum]['rname']
                targetlimits = cluster_info[targetnum]['regionlimits']

                for tt in range(ntclusters):
                    targetcoords = cluster_info[targetnum]['cluster_coords'][tt, :]
                    beta1 = sem_one_target1[tt]['b']
                    beta2 = sem_one_target2[tt]['b']
                    ncombinations, ntimepoints, NP2, nsources = np.shape(beta2)
                    ncombinations, ntimepoints, NP1, nsources = np.shape(beta1)

                    # stats based on group average - --------------------
                    if statstype == 'average':
                        if mode == 'unpaired':
                            stattitle = 'Tvalue unpaired'
                            # stats based on group average - sig diff between groups?
                            mean_beta1 = np.mean(beta1, axis=2)
                            var_beta1 = np.var(beta1, axis=2)
                            mean_beta2 = np.mean(beta2, axis=2)
                            var_beta2 = np.var(beta2, axis=2)
                            # pooled standard deviation:
                            sp = np.sqrt(((NP1 - 1) * var_beta1 + (NP2 - 1) * var_beta2) / (NP1 + NP2 - 2))

                            Tbeta = (mean_beta1 - mean_beta2) / (sp * np.sqrt(1 / NP1 + 1 / NP2))
                            Tthresh = stats.t.ppf(1 - pthreshold, NP1 - 1)
                            beta_sig = np.abs(Tbeta) > Tthresh    # size is ncombinations x ntimepoints x nsources
                            stat_of_interest = Tbeta
                        else:
                            stattitle = 'Tvalue paired'
                            # stats based on group average - sig diff between groups?
                            beta_diff = beta1-beta2
                            mean_beta_diff = np.mean(beta_diff,axis = 2)
                            sem_beta_diff = np.std(beta_diff)/np.sqrt(NP1)
                            Tbeta = mean_beta_diff/(sem_beta_diff + 1.0e-10)

                            Tthresh = stats.t.ppf(1-pthreshold,NP-1)
                            beta_sig = np.abs(Tbeta) > Tthresh
                            stat_of_interest = Tbeta

                    # # stats based on regression with covariates - --------
                    # if statstype == 'regression':
                    #     stattitle = 'Zregression'
                    #     Zthresh = stats.norm.ppf(1 - pthreshold)
                    #     terms, NPt = np.shape(covariates)  # need one term per person, for each covariate
                    #     b, bsem, R2, Z, Rcorrelation, Zcorrelation = GLMregression(beta, covariates, 2)
                    #     beta_sig = np.abs(Z) > Zthresh
                    #     stat_of_interest = Z
                    #
                    # # stats based on regression with covariates - --------
                    # if statstype == 'correlation':
                    #     stattitle = 'Zcorr'
                    #     Zthresh = stats.norm.ppf(1 - pthreshold)
                    #     terms, NPt = np.shape(covariates)  # need one term per person, for each covariate
                    #     b, bsem, R2, Z, Rcorrelation, Zcorrelation = GLMregression(beta, covariates, 2)
                    #     beta_sig = np.abs(Zcorrelation) > Zthresh
                    #     stat_of_interest = Zcorrelation
                    # #---------------------------------------------------

                    keys = ['tname', 'tcluster', 'sname', 'scluster', stattitle, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2',
                            'tlimy1',
                            'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2',
                            'slimz1', 'slimz2', 'networkcomponent','tcluster', 'combo','nt','ss']

                    # organize significant results
                    # if np.ndim(beta_sig) < 4:  # allow for different forms of results (some have multiple stats terms)
                    #     stat_of_interest = np.expand_dims(stat_of_interest, axis=-1)
                    #     beta_sig = np.expand_dims(beta_sig, axis=-1)
                    combo, nt, ss = np.where(beta_sig)   # significant connections during this time period

                    # cc = 0   # what about regression with two or more terms?
                    for ii in range(len(combo)):
                        # get region names, cluster numbers, etc.
                        Svalue = stat_of_interest[combo[ii], nt[ii], ss[ii]]
                        timepoint = nt[ii]
                        sourcename = cluster_info[sourcenums[ss[ii]]]['rname']
                        mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums],combo[ii]).astype(int)   # cluster number for each source
                        sourcecluster = mlist[ss[ii]]
                        sourcecoords = cluster_info[sourcenums[ss[ii]]]['cluster_coords'][sourcecluster, :]
                        sourcelimits = cluster_info[sourcenums[ss[ii]]]['regionlimits']

                        connid = nt[ii]*1e7 + targetnum*1e5 + tt*1e3 + sourcenums[ss[ii]]*10 + sourcecluster

                        connection_info = [networkcomponent, tt, combo[ii], nt[ii], ss[ii]]
                        values = np.concatenate(([targetname, tt, sourcename, sourcecluster, Svalue],
                             list(targetcoords), list(targetlimits), list(sourcecoords), list(sourcelimits), connection_info))
                        entry = dict(zip(keys, values))

                        results.append(entry)
                        Svalue_list.append(Svalue)
                        connid_list.append(connid)

            # eliminate redundant values, for repeats keep the one with the largest Tvalue
            if len(results) > 0:
                print('removing redundant values ...')
                results2, Svalue_list2 = remove_reps_and_sort(np.array(connid_list), np.array(Svalue_list), results)

                # separate by timepoints
                timepoint_list = [int(results2[ii]['nt']) for ii in range(len(results2))]
                times = np.unique(timepoint_list)
                print('time point values: ',times)

                # still need to split the data according to timepoints
                print('separating values from different time periods...')
                for timepoint in times:
                    indices = np.where(timepoint_list == timepoint)[0]
                    print('timepoint: {}, found {} entries'.format(timepoint,len(indices)))
                    results1 = []
                    for ii in indices:
                        results1.append(results2[ii])

                    excelsheetname = 'network ' + statstype + ' ' + str(timepoint)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results1, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
            else:
                print('no significant results found with p < {} '.format(pthreshold))

            return excelfilename

    if datafiletype == 2:
        # analyzing BOLD responses
        region_properties1 = data1['region_properties']
        region_properties2 = data2['region_properties']
        # regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,'rname': rname}

        nregions = len(region_properties1)
        for rr in range(nregions):
            tc1 = region_properties1[rr]['tc']          # nclusters x tsize_total
            tc_sem1 = region_properties1[rr]['tc_sem']
            tsize1 = region_properties1[rr]['tsize']
            nruns_per_person1 = region_properties1[rr]['nruns_per_person']
            rname1 = region_properties1[rr]['rname']
            NP1 = len(nruns_per_person1)

            tc2 = region_properties2[rr]['tc']          # nclusters x tsize_total
            tc_sem2 = region_properties2[rr]['tc_sem']
            tsize2 = region_properties2[rr]['tsize']
            nruns_per_person2 = region_properties2[rr]['nruns_per_person']
            rname2 = region_properties2[rr]['rname']
            NP2 = len(nruns_per_person2)

            nclusters, tsize_total1 = np.shape(tc1)   # nclusters need to be the same for the two data sets, for comparisons
            nclusters, tsize_total2 = np.shape(tc2)   # nclusters need to be the same for the two data sets, for comparisons

            # change shape of timecourse data array - prep data - group 1
            tc_per_person1 = np.zeros((nclusters,tsize1,NP1))
            tc_per_person_sem1 = np.zeros((nclusters,tsize1,NP1))
            for nn in range(NP1):
                nruns = nruns_per_person1[nn]
                t1 = np.sum(nruns_per_person1[:nn])*tsize1
                t2 = np.sum(nruns_per_person1[:(nn+1)])*tsize1
                tp = list(range(t1,t2))
                tcsingle = np.mean(np.reshape(tc1[:,tp],(nclusters,tsize1,nruns)),axis = 2)
                tc_semsingle = np.mean(np.reshape(tc_sem1[:,tp],(nclusters,tsize1,nruns)),axis = 2)
                tc_per_person1[:,:,nn] = tcsingle
                tc_per_person_sem1[:,:,nn] = tc_semsingle

            # change shape of timecourse data array - prep data - group 2
            tc_per_person2 = np.zeros((nclusters,tsize2,NP2))
            tc_per_person_sem2 = np.zeros((nclusters,tsize2,NP2))
            for nn in range(NP2):
                nruns = nruns_per_person2[nn]
                t1 = np.sum(nruns_per_person2[:nn])*tsize2
                t2 = np.sum(nruns_per_person2[:(nn+1)])*tsize2
                tp = list(range(t1,t2))
                tcsingle = np.mean(np.reshape(tc2[:,tp],(nclusters,tsize2,nruns)),axis = 2)
                tc_semsingle = np.mean(np.reshape(tc_sem2[:,tp],(nclusters,tsize2,nruns)),axis = 2)
                tc_per_person2[:,:,nn] = tcsingle
                tc_per_person_sem2[:,:,nn] = tc_semsingle

            # stats based on group average ---------------------------
            if statstype == 'average':
                if mode == 'unpaired':
                    # stats based on group average - sig diff between groups?
                    mean_tc1 = np.mean(tc_per_person1, axis=2)
                    var_tc1 = np.var(tc_per_person1, axis=2)
                    mean_tc2 = np.mean(tc_per_person2, axis=2)
                    var_tc2 = np.var(tc_per_person2, axis=2)
                    # pooled standard deviation:
                    sp = np.sqrt(((NP1 - 1) * var_tc1 + (NP2 - 1) * var_tc2) / (NP1 + NP2 - 2))
                    T = (mean_tc1 - mean_tc2) / (sp * np.sqrt(1 / NP1 + 1 / NP2))
                    Tthresh = stats.t.ppf(1 - pthreshold, NP1 - 1)
                    sig = np.abs(T) > Tthresh

                    tc_diff = mean_tc1-mean_tc2
                    tc_diff_sem =  (sp * np.sqrt(1 / NP1 + 1 / NP2))
                else:
                    diff = tc_per_person1-tc_per_person2
                    mean_diff = np.mean(diff,axis = 2)
                    sem_diff = np.std(diff)/np.sqrt(NP1)
                    T = mean_diff/(sem_diff + 1.0e-10)
                    Tthresh = stats.t.ppf(1 - pthreshold, NP1 - 1)
                    sig = np.abs(T) > Tthresh

                    tc_diff = mean_diff
                    tc_diff_sem =  sem_diff

                # check significance and write out results
                keys = []
                for cc in range(nclusters):
                    keys = keys + ['avg ' + str(cc), 'sem ' + str(cc), 'T ' + str(cc), 'sig ' + str(cc)]

                # write out timecourse values?
                outputdata = []
                for tt in range(tsize):
                    values = []
                    for cc in range(nclusters):
                        values = values + [tc_diff[cc, tt], tc_diff_sem[cc, tt], T[cc, tt], sig[cc, tt]]
                    entry = dict(zip(keys, values))
                    outputdata.append(entry)

            if len(outputdata) > 0:
                excelsheetname = rname
                print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                pydisplay.pywriteexcel(outputdata, excelfilename, excelsheetname, 'append', '%.3f')
                print('finished writing results to ',excelfilename)
            else:
                print('no significant results found at p < {}'.format(pthreshold))
            return excelfilename


# -------------group_comparison_ANOVA--------------------------------------
def group_comparison_ANOVA(filename1, filename2, covariates1, covariates2, pthreshold, mode = 'ANOVA', covariate_name = 'cov1'):
    # test fMRI results for group-level differences, using an ANOVA
    #
    # if covariates1 and covariates2 contain more than one set of values, only the first one is used
    # "mode" can be ANOVA or ANCOVA
    # for ANOVA:  the covariate will be treated as a categorical variable
    # for ANCOVA:  the covariate will be treated as a continuous variable
    #
    print('running py2ndlevelanalysis:  group_comparison_ANOVA:  started at ',time.ctime())
    data1 = np.load(filename1, allow_pickle=True).flat[0]
    data2 = np.load(filename2, allow_pickle=True).flat[0]

    # setup output name
    tag = '_' + covariate_name
    excelfilename = generate_output_name(mode+'_',filename1, filename2, tag, '.xlsx')

    if np.ndim(covariates1) > 1:
        ncov1, NP1 = np.shape(covariates1)
        cov1 = covariates1[0,:]
    else:
        cov1 = covariates1

    if np.ndim(covariates2) > 1:
        ncov2, NP2 = np.shape(covariates2)
        cov2 = covariates2[0, :]
    else:
        cov2 = covariates2

    # check the data types
    if mode == 'ANCOVA':
        cov1 = np.array(cov1).astype(float)
        cov2 = np.array(cov2).astype(float)

    print('cov1 = {}'.format(cov1))
    print('cov2 = {}'.format(cov2))

    datafiletype = 0
    try:
        keylist = list(data1.keys())
        if 'type' in keylist: datafiletype = 1   # SEM results
        if 'region_properties' in keylist: datafiletype = 2    # BOLD time-course data
        print('group_comparison_ANOVA:  datafiletype = ', datafiletype)
    except:
        print('group_comparison_ANOVA:  input file does not appear to have the correct data structure')
        return 0

    if datafiletype == 1:
        semtype = data1['type']
        print('SEM results loaded:  type ',semtype)

        if semtype == '2source':
            cluster_properties = data1['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            beta1_1 = data1['beta1']
            beta2_1 = data1['beta2']
            beta1_2 = data2['beta1']
            beta2_2 = data2['beta2']

            ntclusters, ns1clusters, ns2clusters, ntimepoints, NP2, nbeta = np.shape(beta1_2)
            ntclusters, ns1clusters, ns2clusters, ntimepoints, NP1, nbeta = np.shape(beta1_1)

            anova_p_beta1 = np.ones((ntclusters,ns1clusters, ns2clusters, ntimepoints, nbeta,3))
            anova_p_beta2 = np.ones((ntclusters,ns1clusters, ns2clusters, ntimepoints, nbeta,3))

            # ---------do the ANOVA here-----------------------
            if mode == 'ANOVA':
                statstype = 'ANOVA'
                formula_key1 = 'C(Group)'
                formula_key2 = 'C(' + covariate_name + ')'
                formula_key3 = 'C(Group):C('+covariate_name+')'
                atype = 2
            else:
                statstype = 'ANCOVA'
                formula_key1 = 'C(Group)'
                formula_key2 = covariate_name
                formula_key3 = 'C(Group):'+covariate_name
                atype = 2

            for t in range(ntclusters):
                print('{} percent complete {}'.format(100.*t/ntclusters,time.ctime()))
                for s1 in range(ns1clusters):
                    for s2 in range(ns2clusters):
                        for tp in range(ntimepoints):
                            for nb in range(nbeta):
                                # one-source
                                b1 = beta1_1[t, s1, s2, tp, :, nb]
                                b2 = beta1_2[t, s1, s2, tp, :, nb]
                                covname = covariate_name
                                if np.var(b1) > 0 and np.var(b2) > 0:
                                    anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2, covname, formula_key1, formula_key2, formula_key3, atype)
                                    anova_p_beta1[t,s1,s2,tp,nb,:] = np.array([p_MeoG, p_MeoC, p_intGC])

                                # two-source
                                b1 = beta2_1[t, s1, s2, tp, :, nb]
                                b2 = beta2_2[t, s1, s2, tp, :, nb]
                                if np.var(b1) > 0 and np.var(b2) > 0:
                                    # anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA2(beta1, beta2, cov1, cov2, covname='cov1', mode='ANOVA')
                                    anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2, covname, formula_key1, formula_key2, formula_key3, atype)
                                    anova_p_beta2[t,s1,s2,tp,nb,:] = np.array([p_MeoG, p_MeoC, p_intGC])

            print('100 percent complete {}'.format(time.ctime()))
            beta1_sig = anova_p_beta1 < pthreshold
            beta2_sig = anova_p_beta2 < pthreshold
            stat_of_interest1 = anova_p_beta1
            stat_of_interest2 = anova_p_beta2

            # -----------sorting and writing results---------------------------------

            keys = ['tname', 'tcluster', 'sname', 'scluster', statstype, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2', 'tlimy1',
                    'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2', 'slimz1', 'slimz2',
                    't','s1','s2','tt','nb','nc']

            # write out significant results, based on beta1------------------------------
            for tt in range(ntimepoints):
                sig_temp = beta1_sig[:,:,:,tt,:,:]
                t,s1,s2,nb,nc = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                results = []
                for ii in range(len(t)):
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = nc[ii]*1e6+t[ii]*1000 + s   # a unique identifier for the connection
                    Svalue_list[ii] = stat_of_interest1[t[ii], s1[ii], s2[ii], tt, nb[ii], nc[ii]]
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    connection_info = [t[ii],s1[ii],s2[ii],tt,nb[ii],nc[ii]]
                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits), connection_info))
                    entry = dict(zip(keys, values))
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                taglist = ['MeoG','MeoC','Interaction']
                for ncval in range(3):
                    aa = np.where(nc == ncval)[0]
                    tagname = taglist[ncval]

                    results1c = []
                    for aval in aa: results1c.append(results[aval])
                    Svalue_list1c = Svalue_list[aa]
                    connid_list1c = connid_list[aa]

                    if len(results1c) > 0:
                        print('removing redundant values ...')
                        results2, Svalue_list2 = remove_reps_and_sort(connid_list1c, Svalue_list1c, results1c)
                        excelsheetname = '1S' + statstype + ' ' + tagname + ' ' + str(tt)
                        print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                        pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    else:
                        print('no significant results found at p < {}'.format(pthreshold))

            print('finished writing results to excel.')
            # results_beta1 = results2

            # now, write out significant results, based on beta2-------------------------
            for tt in range(ntimepoints):
                sig_temp = beta2_sig[:,:,:,tt,:,:]
                t,s1,s2,nb,nc = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                results = []
                for ii in range(len(t)):
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    connid_list[ii] = nc[ii]*1e6+t[ii]*1000 + s   # a unique identifier for the connection
                    Svalue_list[ii] = stat_of_interest2[t[ii], s1[ii], s2[ii], tt, nb[ii], nc[ii]]
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    connection_info = [t[ii], s1[ii], s2[ii], tt, nb[ii], nc[ii]]
                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits), connection_info))
                    entry = dict(zip(keys, values))
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                taglist = ['MeoG','MeoC','Interaction']
                for ncval in range(3):
                    aa = np.where(nc == ncval)[0]
                    tagname = taglist[ncval]

                    results1c = []
                    for aval in aa: results1c.append(results[aval])
                    Svalue_list1c = Svalue_list[aa]
                    connid_list1c = connid_list[aa]

                    if len(results1c) > 0:
                        print('removing redundant values ...')
                        results2, Svalue_list2 = remove_reps_and_sort(connid_list1c, Svalue_list1c, results1c)
                        excelsheetname = '2S ' + statstype + ' ' + tagname + ' ' + str(tt)
                        print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                        pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    else:
                        print('no significant results found at p < {}'.format(pthreshold))

            print('finished writing results to excel.')
            # results_beta2 = results2

            return excelfilename

        if semtype == 'network':
            # results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
            #            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
            #            'DBnum': self.DBnum}

            resultsnames1 = data1['resultsnames']
            resultsnames2 = data2['resultsnames']
            clustername = data1['clustername']
            regionname = data1['regionname']
            networkmodel = data1['network']
            network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)
            nclusterlist = np.array([ncluster_list[i]['nclusters'] for i in range(len(ncluster_list))])

            cluster_data = np.load(clustername, allow_pickle=True).flat[0]
            cluster_properties = cluster_data['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            results = []
            Svalue_list = []
            connid_list = []   # identify connections - to be able to remove redundant ones later
            nc_list = []
            time_list = []
            for networkcomponent, fname1 in enumerate(resultsnames1):
            # for networkcomponent in range(2,5):  # this is just for testing
            #     fname1 = resultsnames1[networkcomponent]
                fname2 = resultsnames2[networkcomponent]
                print('analyzing network component: \n{}\n{}\n'.format(fname1,fname2))
                semresults1 = np.load(fname1, allow_pickle=True).flat[0]
                sem_one_target1 = semresults1['sem_one_target_results']
                semresults2= np.load(fname2, allow_pickle=True).flat[0]
                sem_one_target2 = semresults2['sem_one_target_results']
                ntclusters = len(sem_one_target1)
                target = network[networkcomponent]['target']
                sources = network[networkcomponent]['sources']
                targetnum = network[networkcomponent]['targetnum']
                sourcenums = network[networkcomponent]['sourcenums']
                targetname = cluster_info[targetnum]['rname']
                targetlimits = cluster_info[targetnum]['regionlimits']

                ncombinations, ntimepoints, NP1, nsources = np.shape(sem_one_target1[0]['b'])
                ncombinations, ntimepoints, NP2, nsources = np.shape(sem_one_target2[0]['b'])
                # initialize for saving results for each network component
                anova_p = np.ones((ntclusters, ncombinations, ntimepoints,nsources,3))

                for tt in range(ntclusters):
                    print('{} percent complete {}'.format(100.*tt/ntclusters,time.ctime()))
                    targetcoords = cluster_info[targetnum]['cluster_coords'][tt, :]
                    beta1 = sem_one_target1[tt]['b']
                    beta2 = sem_one_target2[tt]['b']

                    # -------------do the ANOVA here-----------------------
                    if mode == 'ANOVA':
                        statstype = 'ANOVA'
                        formula_key1 = 'C(Group)'
                        formula_key2 = 'C(' + covariate_name + ')'
                        formula_key3 = 'C(Group):C('+covariate_name+')'
                        atype = 2
                    else:
                        statstype = 'ANCOVA'
                        formula_key1 = 'C(Group)'
                        formula_key2 = covariate_name
                        formula_key3 = 'C(Group):'+covariate_name
                        atype = 2

                    covname = covariate_name
                    for nc in range(ncombinations):
                        for nt in range(ntimepoints):
                            for ns in range(nsources):
                                b1 = beta1[nc,nt,:,ns]
                                b2 = beta2[nc,nt,:,ns]
                                if np.var(b1) > 0 and np.var(b2) > 0:
                                    anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA2(b1, b2, cov1, cov2, covname, formula_key1, formula_key2, formula_key3, atype)
                                    anova_p[tt,nc,nt,ns,:] = np.array([p_MeoG, p_MeoC, p_intGC])

                print('100 percent complete {}'.format(time.ctime()))
                beta_sig = anova_p  < pthreshold
                stat_of_interest = anova_p

                #--------sort and write out the results--------------------------
                keys = ['tname', 'tcluster', 'sname', 'scluster', statstype, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2',
                        'tlimy1',
                        'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2',
                        'slimz1', 'slimz2', 'networkcomponent', 'tt','ncombo','timepoint','ss','nc']

                for timepoint in range(ntimepoints):
                    # organize significant results
                    tt, ncombo, ss, nc = np.where(beta_sig[:,:,timepoint,:,:])   # significant connections for this time period

                    for ii in range(len(tt)):
                        # get region names, cluster numbers, etc.
                        Svalue = stat_of_interest[tt[ii],ncombo[ii], timepoint, ss[ii],nc[ii]]
                        sourcename = cluster_info[sourcenums[ss[ii]]]['rname']
                        mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums],ncombo[ii]).astype(int)   # cluster number for each source
                        sourcecluster = mlist[ss[ii]]
                        sourcecoords = cluster_info[sourcenums[ss[ii]]]['cluster_coords'][sourcecluster, :]
                        sourcelimits = cluster_info[sourcenums[ss[ii]]]['regionlimits']

                        connid = timepoint*1e7 + targetnum*1e5 + tt[ii]*1e3 + sourcenums[ss[ii]]*10 + sourcecluster

                        connection_info = [networkcomponent, tt[ii],ncombo[ii], timepoint, ss[ii],nc[ii]]
                        values = np.concatenate(([targetname, tt[ii], sourcename, sourcecluster, Svalue],
                             list(targetcoords), list(targetlimits), list(sourcecoords), list(sourcelimits), connection_info))
                        entry = dict(zip(keys, values))

                        results.append(entry)
                        Svalue_list.append(Svalue)
                        connid_list.append(connid)
                        nc_list.append(nc[ii])
                        time_list.append(timepoint)

            # save intermediate results for testing
            ptemp,ftemp = os.path.split(excelfilename)
            intermediate_name = os.path.join(ptemp,'intermediate_results.npy')
            intermediate_data = {'results':results, 'Svalue_list':Svalue_list, 'connid_list':connid_list, 'nc_list':nc_list, 'time_list':time_list}
            np.save(intermediate_name, intermediate_data)

            # eliminate redundant values, for repeats keep the one with the largest Tvalue
            tagnamelist = ['MeoG','MeoC','Interaction']
            if len(results) > 0:
                # separate by time and effect type
                for tt in range(ntimepoints):
                    for nc in range(3):
                        # aa = np.where( (np.array(time_list) == tt) and (np.array(nc_list) == nc))[0]
                        aa = [a for a in range(len(time_list)) if ((time_list[a] == tt) and (nc_list[a] == nc))]
                        if len(aa) > 0:
                            tagname = tagnamelist[nc]

                            results1 = [results[aval] for aval in aa]
                            connid1 = [connid_list[aval] for aval in aa]
                            Svalue1 = [Svalue_list[aval] for aval in aa]

                            print('removing redundant values ... time {}  effect {}'.format(tt,nc))
                            results2, Svalue_list2 = remove_reps_and_sort(np.array(connid1), np.array(Svalue1), results1)

                            excelsheetname = 'network ' + statstype + ' ' + tagname + ' ' + str(tt)
                            print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                            pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')

                print('finished writing results to ',excelfilename)
            else:
                print('no significant results found with p < {} '.format(pthreshold))

            return excelfilename

    if datafiletype == 2:
        # analyzing BOLD responses
        region_properties1 = data1['region_properties']
        region_properties2 = data2['region_properties']
        # regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,'rname': rname}

        nregions = len(region_properties1)
        for rr in range(nregions):
            tc1 = region_properties1[rr]['tc']          # nclusters x tsize_total
            tc_sem1 = region_properties1[rr]['tc_sem']
            tsize1 = region_properties1[rr]['tsize']
            nruns_per_person1 = region_properties1[rr]['nruns_per_person']
            rname1 = region_properties1[rr]['rname']
            NP1 = len(nruns_per_person1)

            tc2 = region_properties2[rr]['tc']          # nclusters x tsize_total
            tc_sem2 = region_properties2[rr]['tc_sem']
            tsize2 = region_properties2[rr]['tsize']
            nruns_per_person2 = region_properties2[rr]['nruns_per_person']
            rname2 = region_properties2[rr]['rname']
            NP2 = len(nruns_per_person2)

            nclusters, tsize_total1 = np.shape(tc1)   # nclusters need to be the same for the two data sets, for comparisons
            nclusters, tsize_total2 = np.shape(tc2)   # nclusters need to be the same for the two data sets, for comparisons

            # change shape of timecourse data array - prep data - group 1
            tc_per_person1 = np.zeros((nclusters,tsize1,NP1))
            tc_per_person_sem1 = np.zeros((nclusters,tsize1,NP1))
            for nn in range(NP1):
                nruns = nruns_per_person1[nn]
                t1 = np.sum(nruns_per_person1[:nn])*tsize1
                t2 = np.sum(nruns_per_person1[:(nn+1)])*tsize1
                tp = list(range(t1,t2))
                tcsingle = np.mean(np.reshape(tc1[:,tp],(nclusters,tsize1,nruns)),axis = 2)
                tc_semsingle = np.mean(np.reshape(tc_sem1[:,tp],(nclusters,tsize1,nruns)),axis = 2)
                tc_per_person1[:,:,nn] = tcsingle
                tc_per_person_sem1[:,:,nn] = tc_semsingle

            # change shape of timecourse data array - prep data - group 2
            tc_per_person2 = np.zeros((nclusters,tsize2,NP2))
            tc_per_person_sem2 = np.zeros((nclusters,tsize2,NP2))
            for nn in range(NP2):
                nruns = nruns_per_person2[nn]
                t1 = np.sum(nruns_per_person2[:nn])*tsize2
                t2 = np.sum(nruns_per_person2[:(nn+1)])*tsize2
                tp = list(range(t1,t2))
                tcsingle = np.mean(np.reshape(tc2[:,tp],(nclusters,tsize2,nruns)),axis = 2)
                tc_semsingle = np.mean(np.reshape(tc_sem2[:,tp],(nclusters,tsize2,nruns)),axis = 2)
                tc_per_person2[:,:,nn] = tcsingle
                tc_per_person_sem2[:,:,nn] = tc_semsingle

            #-----------do the ANOVA here--------------------------------
            anova_p = np.ones(nclusters,tsize1,3)
            if mode == 'ANOVA':
                statstype = 'ANOVA'
                formula_key1 = 'C(Group)'
                formula_key2 = 'C(' + covariate_name + ')'
                formula_key3 = 'C(Group):C(' + covariate_name + ')'
                atype = 2
            else:
                statstype = 'ANCOVA'
                formula_key1 = 'C(Group)'
                formula_key2 = covariate_name
                formula_key3 = 'C(Group):' + covariate_name
                atype = 2

            covname = covariate_name
            for nc in range(nclusters):
                for ts in range(tsize1):
                    tc1 = tc_per_person1[nc,ts,:]
                    tc2 = tc_per_person2[nc,ts,:]
                    if np.var(tc1) > 0 and np.var(tc2) > 0:
                        anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA2(tc1, tc2, cov1, cov2, covname,
                                                                                    formula_key1, formula_key2,
                                                                                    formula_key3, atype)
                        anova_p[nc, ts, :] = np.array([p_MeoG, p_MeoC, p_intGC])

            beta_sig = anova_p < pthreshold

            # --------sort and write out the results---------------------------
            keys = ['tname', 'tcluster', 'sname', 'scluster', 'anova_p', 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2',
                    'tlimy1',
                    'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2',
                    'slimz1', 'slimz2', 'combo','nt','ss']

            # organize significant results
            # if np.ndim(beta_sig) < 4:  # allow for different forms of results (some have multiple stats terms)
            #     # stat_of_interest = np.expand_dims(stat_of_interest, axis=-1)
            #     beta_sig = np.expand_dims(beta_sig, axis=-1)
            combo, nt, ss = np.where(beta_sig)  # significant connections during this time period

            # cc = 0  # what about regression with two or more terms?
            for ii in range(len(combo)):
                # get region names, cluster numbers, etc.
                Svalue = beta_sig[combo[ii], nt[ii], ss[ii]]
                timepoint = nt[ii]
                sourcename = cluster_info[sourcenums[ss[ii]]]['rname']
                mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums], combo[ii]).astype(
                    int)  # cluster number for each source
                sourcecluster = mlist[ss[ii]]
                sourcecoords = cluster_info[sourcenums[ss[ii]]]['cluster_coords'][sourcecluster, :]
                sourcelimits = cluster_info[sourcenums[ss[ii]]]['regionlimits']

                # connid = nt[ii] * 1e7 + targetnum * 1e5 + tt * 1e3 + sourcenums[ss[ii]] * 10 + sourcecluster

                connection_info = [combo[ii], nt[ii], ss[ii]]
                values = np.concatenate(([targetname, tt, sourcename, sourcecluster, Svalue],
                                         list(targetcoords), list(targetlimits), list(sourcecoords), list(sourcelimits),
                                         connection_info))
                entry = dict(zip(keys, values))

                results.append(entry)
                Svalue_list.append(Svalue)
                # connid_list.append(connid)

            #-------------sort and write out the results?-------------------------
            if len(results) > 0:
                excelsheetname = rname
                print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                pydisplay.pywriteexcel(results, excelfilename, excelsheetname, 'append', '%.3f')
                print('finished writing results to ',excelfilename)
            else:
                print('no significant results found at p < {}'.format(pthreshold))
            return excelfilename



#---------------single group ANOVA--------------------------------------
#------------------------------------------------------------------------

# -------------group_comparison_ANOVA--------------------------------------
def single_group_ANOVA(filename1, covariates1, pthreshold, mode = 'ANOVA', covariate_names = ['cov1','cov2']):
    # test fMRI results for differences based on covariates, using an ANOVA or ANCOVA
    #
    # if covariates1 contains more than two sets of values, only the first two are used
    # "mode" can be ANOVA or ANCOVA
    # for ANOVA:  the 2nd covariate will be treated as a categorical variable
    # for ANCOVA:  the 2nd covariate will be treated as a continuous variable
    #
    print('running py2ndlevelanalysis:  single_group_ANOVA:  started at ',time.ctime())
    data1 = np.load(filename1, allow_pickle=True).flat[0]

    # setup output name
    tag = ''
    if covnames != 'none':
        tag = ''
        for tname in covnames:  tag += tname+'_'
        tag = tag[:-1]  # take off the trailing character

    excelfilename = generate_output_name(mode+'_',filename1, '', tag, '.xlsx')

    if np.ndim(covariates1) > 2:
        ncov1, NP1 = np.shape(covariates1)
        cov1 = covariates1[0,:]
        cov2 = covariates1[1,:]
        covname1 = covariate_names[0]
        covname2 = covariate_names[1]
    else:
        print('error:  not enough covariates provided for single_group_ANOVA')

    # check data types
    if mode == 'ANCOVA':
        cov2 = cov2.astype(float)   # make sure data are not strings, and interpreted as categorical values

    datafiletype = 0
    try:
        keylist = list(data1.keys())
        if 'type' in keylist: datafiletype = 1   # SEM results
        if 'region_properties' in keylist: datafiletype = 2    # BOLD time-course data
        print('group_comparison_ANOVA:  datafiletype = ', datafiletype)
    except:
        print('group_comparison_ANOVA:  input file does not appear to have the correct data structure')
        return 0

    if datafiletype == 1:
        semtype = data1['type']
        print('SEM results loaded:  type ',semtype)

        if semtype == '2source':
            cluster_properties = data1['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            beta1 = data1['beta1']
            beta2 = data1['beta2']

            ntclusters, ns1clusters, ns2clusters, ntimepoints, NP, nbeta = np.shape(beta1)

            anova_p_beta1 = np.ones((ntclusters,ns1clusters, ns2clusters, ntimepoints, nbeta,3))
            anova_p_beta2 = np.ones((ntclusters,ns1clusters, ns2clusters, ntimepoints, nbeta,3))

            # ---------do the ANOVA here-----------------------
            if mode == 'ANOVA':
                formula_key1 = 'C(' + covname1 + ')'
                formula_key2 = 'C(' + covname2 + ')'
                formula_key3 = 'C('+covname1+'):C('+covname2+')'
                atype = 2
            else:
                formula_key1 = 'C(' + covname1 + ')'
                formula_key2 = covname2
                formula_key3 = 'C('+covname1+'):'+covname2
                atype = 2

            for t in range(ntclusters):
                print('{} percent complete {}'.format(100.*t/ntclusters,time.ctime()))
                for s1 in range(ns1clusters):
                    for s2 in range(ns2clusters):
                        for tp in range(ntimepoints):
                            for nb in range(nbeta):

                                # one-source
                                b1 = beta1[t, s1, s2, tp, :, nb]
                                if np.var(b1) > 0:
                                    anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA1(b1, cov1, cov2, covname1, covname2, formula_key1, formula_key2, formula_key3, atype)
                                    anova_p_beta1[t,s1,s2,tp,nb,:] = np.array([p_MeoG, p_MeoC, p_intGC])

                                # two-source
                                b1 = beta2[t, s1, s2, tp, :, nb]
                                if np.var(b1) > 0:
                                    anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA1(b1, cov1, cov2, covname1, covname2, formula_key1, formula_key2, formula_key3, atype)
                                    anova_p_beta2[t,s1,s2,tp,nb,:] = np.array([p_MeoG, p_MeoC, p_intGC])

            print('100 percent complete {}'.format(time.ctime()))
            beta1_sig = anova_p_beta1 < pthreshold
            beta2_sig = anova_p_beta2 < pthreshold
            stat_of_interest1 = anova_p_beta1
            stat_of_interest2 = anova_p_beta2

            # -----------sorting and writing results---------------------------------
            statstype = 'anova_p'
            keys = ['tname', 'tcluster', 'sname', 'scluster', statstype, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2', 'tlimy1',
                    'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2', 'slimz1', 'slimz2',
                    't','s1','s2','tt','nb','nc']

            # write out significant results, based on beta1------------------------------
            # if np.ndim(beta1_sig) < 6:  # allow for different forms of results (some have multiple stats terms)
            #     beta1_sig = np.expand_dims(beta1_sig, axis=-1)
            #     stat_of_interest1 = np.expand_dims(stat_of_interest1, axis=-1)

            for tt in range(ntimepoints):
                results = []
                sig_temp = beta1_sig[:,:,:,tt,:,:]
                t,s1,s2,nb,nc = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    Svalue_list[ii] = stat_of_interest1[t[ii],s1[ii],s2[ii],tt,nb[ii],nc[ii]]
                    connid_list[ii] = nc[ii]*1e6+t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    connection_info = [t[ii],s1[ii],s2[ii],tt,nb[ii],nc[ii]]
                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits),connection_info))
                    entry = dict(zip(keys, values))
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                if len(results) > 0:
                    print('removing redundant values ...')
                    results2, Svalue_list2 = remove_reps_and_sort(connid_list, Svalue_list, results)

                    excelsheetname = '2S beta2 ' + statstype + ' ' + str(tt)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
                else:
                    print('no significant results found at p < {}'.format(pthreshold))

            results_beta1 = results2

            # now, write out significant results, based on beta2-------------------------
            # if np.ndim(beta2_sig) < 6:  # allow for different forms of results (some have multiple stats terms)
            #     beta2_sig = np.expand_dims(beta2_sig, axis=-1)
            #     stat_of_interest2 = np.expand_dims(stat_of_interest2, axis=-1)

            for tt in range(ntimepoints):
                results = []
                sig_temp = beta2_sig[:,:,:,tt,:,:]
                t,s1,s2,nb,nc = np.where(sig_temp)    # significant connections during this time period

                Svalue_list = np.zeros(len(t))
                connid_list = np.zeros(len(t))   # identify connections - to be able to remove redundant ones later
                for ii in range(len(t)):
                    if nb[ii] == 0:
                        s = s1[ii]
                    else:
                        s = s2[ii]
                    # get region names, cluster numbers, etc.
                    Svalue_list[ii] = stat_of_interest2[t[ii],s1[ii],s2[ii],tt,nb[ii],nc[ii]]
                    connid_list[ii] = t[ii]*1000 + s   # a unique identifier for the connection
                    targetname, targetcluster, targetnumber = get_cluster_info(rname_list, ncluster_list, t[ii])
                    sourcename, sourcecluster, sourcenumber = get_cluster_info(rname_list, ncluster_list, s)
                    targetcoords = cluster_info[targetnumber]['cluster_coords'][targetcluster,:]
                    targetlimits = cluster_info[targetnumber]['regionlimits']
                    sourcecoords = cluster_info[sourcenumber]['cluster_coords'][sourcecluster,:]
                    sourcelimits = cluster_info[sourcenumber]['regionlimits']

                    connection_info = [t[ii], s1[ii], s2[ii], tt, nb[ii], nc[ii]]
                    values = np.concatenate(([targetname, targetcluster, sourcename, sourcecluster, Svalue_list[ii]],
                                             list(targetcoords),list(targetlimits), list(sourcecoords),list(sourcelimits),connection_info))
                    entry = dict(zip(keys, values))
                    results.append(entry)

                # eliminate redundant values, for repeats keep the one with the largest Tvalue
                if len(results) > 0:
                    print('removing redundant values ...')
                    results2, Svalue_list2 = remove_reps_and_sort(connid_list, Svalue_list, results)

                    excelsheetname = '2S beta2 ' + statstype + ' ' + str(tt)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results2, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
                else:
                    print('no significant results found at p < {}'.format(pthreshold))

            results_beta2 = results2

            return excelfilename


        if semtype == 'network':
            # results = {'type': 'network', 'resultsnames': outputnamelist, 'network': self.networkmodel,
            #            'regionname': self.SEMregionname, 'clustername': self.SEMclustername, 'DBname': self.DBname,
            #            'DBnum': self.DBnum}

            resultsnames1 = data1['resultsnames']
            clustername = data1['clustername']
            regionname = data1['regionname']
            networkmodel = data1['network']
            network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkmodel)
            nclusterlist = np.array([ncluster_list[i]['nclusters'] for i in range(len(ncluster_list))])

            cluster_data = np.load(clustername, allow_pickle=True).flat[0]
            cluster_properties = cluster_data['cluster_properties']
            cluster_info, rname_list, ncluster_list = get_cluster_position_details(cluster_properties)

            results = []
            Svalue_list = []
            connid_list = []   # identify connections - to be able to remove redundant ones later
            for networkcomponent, fname1 in enumerate(resultsnames1):
                semresults1 = np.load(fname1, allow_pickle=True).flat[0]
                sem_one_target1 = semresults1['sem_one_target_results']
                ntclusters = len(sem_one_target1)

                target = network[networkcomponent]['target']
                sources = network[networkcomponent]['sources']
                targetnum = network[networkcomponent]['targetnum']
                sourcenums = network[networkcomponent]['sourcenums']
                targetname = cluster_info[targetnum]['rname']
                targetlimits = cluster_info[targetnum]['regionlimits']

                ncombinations, ntimepoints, NP1, nsources = np.shape(sem_one_target1[0]['b'])
                # initialize for saving results for each network component
                anova_p = np.ones((ntclusters, ncombinations, ntimepoints,nsources,3))

                for tt in range(ntclusters):
                    print('{} percent complete {}'.format(100.*t/ntclusters,time.ctime()))
                    targetcoords = cluster_info[targetnum]['cluster_coords'][tt, :]
                    beta1 = sem_one_target1[tt]['b']

                    # -------------do the ANOVA here-----------------------
                    if mode == 'ANOVA':
                        formula_key1 = 'C(' + covname1 + ')'
                        formula_key2 = 'C(' + covname2 + ')'
                        formula_key3 = 'C('+covname1+'):C('+covname2+')'
                        atype = 2
                    else:
                        formula_key1 = 'C(' + covname1 + ')'
                        formula_key2 = covname2
                        formula_key3 = 'C('+covname1+'):'+covname2
                        atype = 2

                    for nc in range(ncombinations):
                        for nt in range(ntimepoints):
                            for ns in range(nsources):
                                b1 = beta1[nc,nt,:,ns]
                                if np.var(b1) > 0 and np.var(b2) > 0:
                                    anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA1(b1, cov1, cov2, covname1, covname2, formula_key1, formula_key2, formula_key3, atype)
                                    anova_p[tt,nc,nt,ns,:] = np.array([p_MeoG, p_MeoC, p_intGC])

                print('100 percent complete {}'.format(time.ctime()))
                beta_sig = anova_p  < pthreshold

                #--------sort and write out the results---------------------------
                statstype = 'anova_p'
                keys = ['tname', 'tcluster', 'sname', 'scluster', statstype, 'tx', 'ty', 'tz', 'tlimx1', 'tlimx2',
                        'tlimy1',
                        'tlimy2', 'tlimz1', 'tlimz2', 'sx', 'sy', 'sz', 'slimx1', 'slimx2', 'slimy1', 'slimy2',
                        'slimz1', 'slimz2', 'tt','combo','nt','ss','nc']

                # organize significant results
                # if np.ndim(beta_sig) < 4:  # allow for different forms of results (some have multiple stats terms)
                #     stat_of_interest = np.expand_dims(stat_of_interest, axis=-1)
                #     beta_sig = np.expand_dims(beta_sig, axis=-1)
                tt, combo, nt, ss, nc = np.where(beta_sig)   # significant connections during this time period

                # cc = 0   # what about regression with two or more terms?
                for ii in range(len(combo)):
                    # get region names, cluster numbers, etc.
                    Svalue = stat_of_interest[tt[ii],combo[ii], nt[ii], ss[ii],nc[ii]]
                    timepoint = nt[ii]
                    sourcename = cluster_info[sourcenums[ss[ii]]]['rname']
                    mlist = pysem.ind2sub_ndims(nclusterlist[sourcenums],combo[ii]).astype(int)   # cluster number for each source
                    sourcecluster = mlist[ss[ii]]
                    sourcecoords = cluster_info[sourcenums[ss[ii]]]['cluster_coords'][sourcecluster, :]
                    sourcelimits = cluster_info[sourcenums[ss[ii]]]['regionlimits']

                    connid = nt[ii]*1e7 + targetnum*1e5 + tt[ii]*1e3 + sourcenums[ss[ii]]*10 + sourcecluster

                    connection_info = [tt[ii],combo[ii], nt[ii], ss[ii],nc[ii]]
                    values = np.concatenate(([targetname, tt[ii], sourcename, sourcecluster, Svalue],
                         list(targetcoords), list(targetlimits), list(sourcecoords), list(sourcelimits), connection_info))
                    entry = dict(zip(keys, values))

                    results.append(entry)
                    Svalue_list.append(Svalue)
                    connid_list.append(connid)

            # eliminate redundant values, for repeats keep the one with the largest Tvalue
            if len(results) > 0:
                print('removing redundant values ...')
                results2, Svalue_list2 = remove_reps_and_sort(np.array(connid_list), np.array(Svalue_list), results)

                # separate by timepoints
                timepoint_list = [int(results2[ii]['timepoint']) for ii in range(len(results2))]
                times = np.unique(timepoint_list)
                print('time point values: ',times)

                # still need to split the data according to timepoints
                print('separating values from different time periods...')
                for timepoint in times:
                    indices = np.where(timepoint_list == timepoint)[0]
                    print('timepoint: {}, found {} entries'.format(timepoint,len(indices)))
                    results1 = []
                    for ii in indices:
                        results1.append(results2[ii])

                    excelsheetname = 'network ' + statstype + ' ' + str(timepoint)
                    print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                    pydisplay.pywriteexcel(results1, excelfilename, excelsheetname, 'append')
                    print('finished writing results to ',excelfilename)
            else:
                print('no significant results found with p < {} '.format(pthreshold))

            return excelfilename


    if datafiletype == 2:
        # analyzing BOLD responses
        region_properties1 = data1['region_properties']
        # regiondata_entry = {'tc': tc, 'tc_sem': tc_sem, 'nruns_per_person': nruns_per_person, 'tsize': tsize,'rname': rname}

        nregions = len(region_properties1)
        for rr in range(nregions):
            tc1 = region_properties1[rr]['tc']          # nclusters x tsize_total
            tc_sem1 = region_properties1[rr]['tc_sem']
            tsize1 = region_properties1[rr]['tsize']
            nruns_per_person1 = region_properties1[rr]['nruns_per_person']
            rname1 = region_properties1[rr]['rname']
            NP1 = len(nruns_per_person1)

            nclusters, tsize_total1 = np.shape(tc1)   # nclusters need to be the same for the two data sets, for comparisons
            nclusters, tsize_total2 = np.shape(tc2)   # nclusters need to be the same for the two data sets, for comparisons

            # change shape of timecourse data array - prep data - group 1
            tc_per_person1 = np.zeros((nclusters,tsize1,NP1))
            tc_per_person_sem1 = np.zeros((nclusters,tsize1,NP1))
            for nn in range(NP1):
                nruns = nruns_per_person1[nn]
                t1 = np.sum(nruns_per_person1[:nn])*tsize1
                t2 = np.sum(nruns_per_person1[:(nn+1)])*tsize1
                tp = list(range(t1,t2))
                tcsingle = np.mean(np.reshape(tc1[:,tp],(nclusters,tsize1,nruns)),axis = 2)
                tc_semsingle = np.mean(np.reshape(tc_sem1[:,tp],(nclusters,tsize1,nruns)),axis = 2)
                tc_per_person1[:,:,nn] = tcsingle
                tc_per_person_sem1[:,:,nn] = tc_semsingle

            #-----------do the ANOVA here--------------------------------
                if mode == 'ANOVA':
                    formula_key1 = 'C(' + covname1 + ')'
                    formula_key2 = 'C(' + covname2 + ')'
                    formula_key3 = 'C('+covname1+'):C('+covname2+')'
                    atype = 2
                else:
                    formula_key1 = 'C(' + covname1 + ')'
                    formula_key2 = covname2
                    formula_key3 = 'C('+covname1+'):'+covname2
                    atype = 2

            anova_p = np.ones((nclusters,tsize1,3))
            for nc in range(nclusters):
                for ts in range(tsize1):
                    tc1 = tc_per_person1[nc, ts, :]
                    if np.var(tc1) > 0 and np.var(tc2) > 0:
                        anova_table, p_MeoG, p_MeoC, p_intGC = run_ANOVA_or_ANCOVA1(tc1, cov1, cov2, covname1, covname2,
                                                                                    formula_key1, formula_key2,
                                                                                    formula_key3, atype)
                        anova_p[nc, ts, :] = np.array([p_MeoG, p_MeoC, p_intGC])

            beta_sig = anova_p < pthreshold

            #-------------sort and write out the results-------------------------
            if len(outputdata) > 0:
                excelsheetname = rname
                print('writing results to {}, sheet {}'.format(excelfilename, excelsheetname))
                pydisplay.pywriteexcel(outputdata, excelfilename, excelsheetname, 'append', '%.3f')
                print('finished writing results to ',excelfilename)
            else:
                print('no significant results found at p < {}'.format(pthreshold))
            return excelfilename
        
        
        

# def run_ANOVA_or_ANCOVA2(beta1, beta2, cov1, cov2, covname = 'cov1', mode = 'ANOVA'):
def run_ANOVA_or_ANCOVA2(beta1, beta2, cov1, cov2, covname, formula_key1, formula_key2, formula_key3, atype):
    # make up test values
    NP1 = len(beta1)
    NP2 = len(beta2)

    g1 = ['group1']
    g2 = ['group2']
    group = g1 * NP1 + g2 * NP2
    beta = list(beta1) + list(beta2)
    cov = list(cov1) + list(cov2)

    d = {'beta': beta, 'Group': group, covname:cov}
    # print('size of beta is {}'.format(np.shape(beta)))
    # print('size of group is {}'.format(np.shape(group)))
    # print('size of cov is {}'.format(np.shape(cov)))
    # print('d = {}'.format(d))

    df = pd.DataFrame(data=d)

    formula = 'beta ~ ' + formula_key1 + ' + ' + formula_key2 + ' + ' + formula_key3

    try:
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=atype)

        p_MeoG = anova_table['PR(>F)'][formula_key1]
        p_MeoC = anova_table['PR(>F)'][formula_key2]
        p_intGC = anova_table['PR(>F)'][formula_key3]
    except:
        anova_table = []
        p_MeoG = 1.0
        p_MeoC = 1.0
        p_intGC = 1.0

    return anova_table, p_MeoG, p_MeoC, p_intGC



# def run_ANOVA_or_ANCOVA2(beta1, beta2, cov1, cov2, covname = 'cov1', mode = 'ANOVA'):
def run_ANOVA_or_ANCOVA1(beta1, cov1, cov2, covname1, covname2, formula_key1, formula_key2, formula_key3, atype):
    # make up test values
    NP = len(beta1)

    d = {'beta':beta1, covname1:cov1, covname2:cov2}
    df = pd.DataFrame(data=d)

    formula = 'beta ~ ' + formula_key1 + ' + ' + formula_key2 + ' + ' + formula_key3

    try:
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=atype)

        p_MeoG = anova_table['PR(>F)'][formula_key1]
        p_MeoC = anova_table['PR(>F)'][formula_key2]
        p_intGC = anova_table['PR(>F)'][formula_key3]
    except:
        anova_table = []
        p_MeoG = 1.0
        p_MeoC = 1.0
        p_intGC = 1.0

    return anova_table, p_MeoG, p_MeoC, p_intGC


def move_network_data(network_filename, newfolder):
    # dict_keys(['type', 'resultsnames', 'network', 'regionname', 'clustername', 'DBname', 'DBnum'])
    pname,fname = os.path.split(network_filename)
    data1 = np.load(network_filename, allow_pickle=True).flat[0]

    if pname == newfolder:   # datafile has already been moved, need to update contents
        newfilename = network_filename
    else:
        newfilename = os.path.join(newfolder,fname)

    # resultsnames
    newresultsnames = copy.deepcopy(data1['resultsnames'])
    for nn in range(len(newresultsnames)):
        p,f = os.path.split(newresultsnames[nn])
        newresultsnames[nn] = os.path.join(newfolder,f)

    # clustername
    newclustername = copy.deepcopy(data1['clustername'])
    p,f = os.path.split(newclustername)
    newclustername = os.path.join(newfolder,f)

    # regionname
    newregionname = copy.deepcopy(data1['regionname'])
    p,f = os.path.split(newregionname)
    newregionname = os.path.join(newfolder,f)

    newdata = copy.deepcopy(data1)
    newdata['resultsnames'] = newresultsnames
    newdata['clustername'] = newclustername
    newdata['regionname'] = newregionname

    # write the result
    np.save(newfilename,newdata)

