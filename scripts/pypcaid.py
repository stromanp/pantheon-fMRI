# sys.path.extend([r'C:\Users\Stroman\PycharmProjects\pantheon\scripts'])

"""
pypcaid.py    ....python,  principal components analysis (PCA) selection (id) of subregions

This set of functions is for identifyingn the best subregion combinations to
use for Structural and Physiological Modeling (SAPM), as developed by P. Stroman

Evidence of a persistent altered neural state in people with fibromyalgia syndrome during functional MRI studies and its relationship with pain and anxiety
P. W. Stroman; R. Staud; C. F. Pukall
PLOS One (accepted Dec 16, 2024)

Structural and Physiological Modeling (SAPM) for the analysis of functional MRI data applied to a study of human nociceptive processing
P. W. Stroman, M. Umraw, B. Keast, H. Algitami, S. Hassanpour, J. Merletti
Brain Sci. 2023, 13, 1568. https://doi.org/10.3390/brainsci13111568

Proof-of-concept of a novel structural equation modelling approach for the analysis of functional MRI data applied to investigate individual differences in human pain responses
P. W. Stroman, J. M. Powers, G. Ioachim
Human Brain Mapping, 44:2523–2542 (2023). https://doi.org/10.1002/hbm.26228

Based on a predefined anatomical network model, the BOLD signal variations in
each region are explained by modeling the input and output signaling from each region
with variations driven by sources external to the network (latent inputs)

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
import copy
import pysapm
import time
import sklearn
import os
import pandas as pd


# testing or running from command line
def testfunction():
    savename = r'E:\EVrecord_10region_test_V2.npy'
    leaveout = []   # include all regions
    np_step = 5

    regiondataname = r'E:/allstim_regiondata_Oct2024.npy'
    clusterdataname = r'E:/allstim_equalsize_cluster_def_Oct2024.npy'
    SAPMparametersname = r'E:\test_search_10region_params.npy'
    networkfile = r'E:\network_model_June2023_SAPM.xlsx'

    covariatefile = r'E:\test_4440212041_results_covariates.npy'

    timepoint = 'all'
    epoch = 'all'

    covdata = np.load(covariatefile, allow_pickle=True).flat[0]
    GRPcharacteristicslist = copy.deepcopy(covdata['GRPcharacteristicslist'])
    GRPcharacteristicsvalues = copy.deepcopy(covdata['GRPcharacteristicsvalues'])
    GRPcharacteristicscount = copy.deepcopy(covdata['GRPcharacteristicscount'])

    covvals = copy.deepcopy(GRPcharacteristicsvalues[0,:])

    search_by_pca(savename, regiondataname, clusterdataname, SAPMparametersname, networkfile,
                  np_step = np_step, covariate = covvals)


    # check one
    cnums = [4, 0, 4, 1, 1, 1, 1, 0, 4, 3]
    cnums = [0, 3, 4, 1, 0, 2, 3, 3, 4, 3]
    cnums = [4, 3, 4, 1, 1, 1, 1, 1, 4, 3]
    EVrecord, EVrecord_total, EVcorr_cov_record, covariate = check_one_combo_by_pca(cnums, savename, regiondataname,
                        clusterdataname, SAPMparametersname, networkfile, leaveout = [], np_step = 1, covariate = covvals)


def search_by_pca(savename, regiondataname, clusterdataname, SAPMparametersname, networkfile,
                  leaveout = [], np_step = 1, timepoint = 'all', epoch = 'all', covariate = [], resume = False):

    p,f = os.path.split(savename)
    bigsavename = os.path.join(p, 'big'+f)
    if np_step < 1:  np_step = 1

    # load some data, setup some parameters...
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = pysapm.load_network_model_w_intrinsics(networkfile)
    nclusterlist = np.array([nclusterdict[x]['nclusters'] for x in range(len(nclusterdict))])
    cnums = [{'cnums':list(range(nclusterlist[x]))} for x in range(len(nclusterlist))]

    n_components = vintrinsic_count + 1   # add one more for flexibility

    # if not os.path.isfile(SAPMparametersname):
    pysapm.prep_data_sem_physio_model_SO_FC(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                     cnums=cnums, run_whole_group=False, normalizevar=True, filter_tcdata=False)
    #------------------------------
    # determine which cnums to use for the pruned network
    #  ... base this on the R2 of the fit to the original data
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    rnamelist = copy.deepcopy(SAPMparams['rnamelist'])
    nclusterlist = copy.deepcopy(SAPMparams['nclusterlist'])
    nclusterstotal = np.sum(nclusterlist)
    tplist_full = copy.deepcopy(SAPMparams['tplist_full'])
    nruns_per_person = copy.deepcopy(SAPMparams['nruns_per_person'])
    NP = len(nruns_per_person)
    tsize = copy.deepcopy(SAPMparams['tsize'])

    nr = copy.deepcopy(SAPMparams['nregions'])
    fintrinsic_count = copy.deepcopy(SAPMparams['fintrinsic_count'])
    vintrinsic_count = copy.deepcopy(SAPMparams['vintrinsic_count'])
    nlatent = fintrinsic_count + vintrinsic_count

    tcdata_centered = copy.deepcopy(SAPMparams['tcdata_centered'])
    tcdata_centered_original = copy.deepcopy(SAPMparams['tcdata_centered_original'])

    # Rrecord = np.zeros((NP, nclusterstotal))

    if len(covariate) == NP:
        correlate_with_covariate = True
        print('search_by_pca:   results will be tested for correlation with a covariate')
    else:
        correlate_with_covariate = False
        print('search_by_pca:   results will NOT be tested for correlation with a covariate')

    Sinput_data = []
    for nperson in range(NP):
        tp = tplist_full[0][nperson]['tp']
        nruns = nruns_per_person[nperson]
        tsize_total = len(tp)

        if fintrinsic_count > 0:
            fintrinsic_full = np.repeat(fintrinsic_base, nruns, axis = 1)

        Sinput = []
        for nc in range(nclusterstotal):  # use all clusters
            tc1 = tcdata_centered[nc, tp]
            Sinput.append(tc1)
        Sinput = np.array(Sinput)

        Sinput_original = []
        for nc in range(nclusterstotal):  # use all clusters
            tc1 = tcdata_centered_original[nc, tp]
            Sinput_original.append(tc1)
        Sinput_original = np.array(Sinput_original)

        if fintrinsic_count > 0:
            Sinput_wo_flatent, fixed_var_fraction = remove_component(Sinput, fintrinsic_full)
            Sinput_original_wo_flatent, fixed_var_fraction_original = remove_component(Sinput_original, fintrinsic_full)
            data_sample = {'Sinput':Sinput_wo_flatent, 'Sinput_original':Sinput_original_wo_flatent,
                           'fixed_var_fraction':fixed_var_fraction, 'fixed_var_fraction_original':fixed_var_fraction_original}
        else:
            data_sample = {'Sinput':Sinput, 'Sinput_original':Sinput_original}

        Sinput_data.append(data_sample)


    # find how much variance is accounted for by first few principal components
    fulllist = list(range(len(nclusterlist)))
    smalllist = [x for x in range(len(nclusterlist)) if x not in leaveout]
    ncombos = np.prod(np.array(nclusterlist)[smalllist])
    print('ncombinations avialable to run in total: {}'.format(ncombos))
    # ncombos = 1000
    print('ncombinations to be run: {}'.format(ncombos))


    # leave out people----------------------------------------
    pplist = list(range(0,NP,np_step))
    NP2 = len(pplist)

    EVrecord = np.zeros((ncombos,n_components, NP2))
    fixedlatent_EV = np.zeros((ncombos, NP2))
    starttime = time.time()

    clusteroffset = np.array([0] + list(np.cumsum(nclusterlist)))[:-1]
    reportinterval = np.floor(ncombos/20).astype(int)

    nclusterlist_small = np.array(nclusterlist)[smalllist]
    clusteroffset_small = clusteroffset[smalllist]

    if resume:   # check and see if bigsavename exists already, and see how complete it is
        if os.path.isfile(bigsavename):
            try:
                EVrecord = np.load(bigsavename, allow_pickle = True)
                ncombocheck, ncompcheck, npcheck = np.shape(EVrecord)
                if (ncombocheck == ncombos) & (ncompcheck == n_components) & (npcheck == NP2):
                    check = np.mean(EVrecord,axis=2)
                    check = np.mean(check,axis=1)
                    zc = np.where(check == 0.0)[0]
                    ncombostart = zc[0]
                    print('RESUMING PREVIOUS SEARCH:  starting from {} of {}'.format(ncombostart,ncombos))
                else:
                    EVrecord = np.zeros((ncombos, n_components, NP2))
                    ncombostart = 0
                    print('ATTEMPTED TO RESUME PREVIOUS SEARCH:\n   ...previous data size does not match current search settings')
            except:
                EVrecord = np.zeros((ncombos, n_components, NP2))
                ncombostart = 0
        else:
            ncombostart = 0
    else:
        ncombostart = 0

    pca = sklearn.decomposition.PCA(n_components=n_components)
    for combonumber in range(ncombostart,ncombos):
        cnums1 = ind2sub_ndims(nclusterlist_small,combonumber)
        cc = np.array(cnums1) + clusteroffset_small

        for nn, pp in enumerate(pplist):
            # Sin_sample = copy.deepcopy(SAPMresults_previous[nn]['Sinput'][cc,:])
            Sin_sample = copy.deepcopy(Sinput_data[pp]['Sinput'][cc,:])
            pca.fit(Sin_sample)
            EVrecord[combonumber,:,nn] = pca.explained_variance_ratio_

            if fintrinsic_count > 0:
                fixed_var_fraction = copy.deepcopy(Sinput_data[pp]['fixed_var_fraction'][cc])
                fixedlatent_EV[combonumber,nn] = np.mean(fixed_var_fraction)
            # else:
            #     fixed_var_fraction = np.zeros(len(cc))

        if (combonumber > 0) and (combonumber % reportinterval == 0):
            print('{:.1f} percent done {}'.format(100.0*combonumber/ncombos, time.ctime()))
            EVrecord_avg = np.mean(EVrecord, axis = 2)
            if fintrinsic_count > 0:
                fixedlatent_EVrecord_avg = np.mean(fixedlatent_EV, axis = 1)
                data = {'EVrecord_avg': EVrecord_avg, 'fixedlatent_EVrecord_avg':fixedlatent_EVrecord_avg}
            else:
                data = {'EVrecord_avg': EVrecord_avg}
            # bigdata = {'EVrecord': EVrecord}

            np.save(savename, data)
            np.save(bigsavename, EVrecord)

    endtime = time.time()
    print('{:.1f} seconds to check {} combinations'.format(endtime-starttime, ncombos))

    # correlate values per person with covariate
    # EVrecord_avg has size (ncombos,n_components, NP2)
    if correlate_with_covariate:
        cov2 = covariate[pplist]
        EVcorr_cov_record = np.zeros(ncombos)
        EVrecord_sum = np.sum(EVrecord[:,:vintrinsic_count,:],axis=1)
        for combonumber in range(ncombos):
            R = np.corrcoef(EVrecord_sum[combonumber,:],cov2)
            EVcorr_cov_record[combonumber] = R[0,1]
    else:
        EVcorr_cov_record = np.array([0.0])

    print('100 percent done {}'.format(time.ctime()))
    EVrecord_avg = np.mean(EVrecord, axis=2)
    if fintrinsic_count > 0:
        fixedlatent_EVrecord_avg = np.mean(fixedlatent_EV, axis = 1)
        data = {'EVrecord_avg': EVrecord_avg, 'fixedlatent_EVrecord_avg':fixedlatent_EVrecord_avg,
                'pplist':pplist, 'nclusterlist':nclusterlist, 'leaveout':leaveout,
                'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
                'EVcorr_cov_record':EVcorr_cov_record}
    else:
        data = {'EVrecord_avg': EVrecord_avg, 'pplist':pplist, 'nclusterlist':nclusterlist,
                'leaveout':leaveout, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
                'EVcorr_cov_record':EVcorr_cov_record}
    np.save(savename, data)

    p,f1 = os.path.split(savename)
    f,e = os.path.splitext(f1)
    excelsavename = os.path.join(p,f+'.xlsx')
    print('finished PCA analysis of clusters .... analyzing results and summarizing ...')
    bestclusters = analyze_pca_results(savename, excelsavename)
    print('best clusters appear to be {}'.format(bestclusters))

    bestclusters_out = [{'cnums':[x]} for x in bestclusters]

    return bestclusters_out


def check_one_combo_by_pca(cnums_input, savename, regiondataname, clusterdataname, SAPMparametersname, networkfile,
                  leaveout = [], np_step = 1, timepoint = 'all', epoch = 'all', covariate = []):

    if np_step < 1:  np_step = 1

    # load some data, setup some parameters...
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = pysapm.load_network_model_w_intrinsics(networkfile)
    nclusterlist = np.array([nclusterdict[x]['nclusters'] for x in range(len(nclusterdict))])
    cnums = [{'cnums':list(range(nclusterlist[x]))} for x in range(len(nclusterlist))]

    n_components = vintrinsic_count + 1   # add one more for flexibility

    if not os.path.isfile(SAPMparametersname):
        pysapm.prep_data_sem_physio_model_SO_FC(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                         cnums=cnums, run_whole_group=False, normalizevar=True, filter_tcdata=False)
    #------------------------------
    # determine which cnums to use for the pruned network
    #  ... base this on the R2 of the fit to the original data
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    rnamelist = copy.deepcopy(SAPMparams['rnamelist'])
    nclusterlist = copy.deepcopy(SAPMparams['nclusterlist'])
    nclusterstotal = np.sum(nclusterlist)
    tplist_full = copy.deepcopy(SAPMparams['tplist_full'])
    nruns_per_person = copy.deepcopy(SAPMparams['nruns_per_person'])
    NP = len(nruns_per_person)
    tsize = copy.deepcopy(SAPMparams['tsize'])

    nr = copy.deepcopy(SAPMparams['nregions'])
    fintrinsic_count = copy.deepcopy(SAPMparams['fintrinsic_count'])
    vintrinsic_count = copy.deepcopy(SAPMparams['vintrinsic_count'])
    nlatent = fintrinsic_count + vintrinsic_count

    tcdata_centered = copy.deepcopy(SAPMparams['tcdata_centered'])
    tcdata_centered_original = copy.deepcopy(SAPMparams['tcdata_centered_original'])

    # Rrecord = np.zeros((NP, nclusterstotal))

    if len(covariate) == NP:
        correlate_with_covariate = True
    else:
        correlate_with_covariate = False

    Sinput_data = []
    for nperson in range(NP):
        tp = tplist_full[0][nperson]['tp']
        nruns = nruns_per_person[nperson]
        tsize_total = len(tp)

        if fintrinsic_count > 0:
            fintrinsic_full = np.repeat(fintrinsic_base, nruns, axis = 1)

        Sinput = []
        for nc in range(nclusterstotal):  # use all clusters
            tc1 = tcdata_centered[nc, tp]
            Sinput.append(tc1)
        Sinput = np.array(Sinput)

        Sinput_original = []
        for nc in range(nclusterstotal):  # use all clusters
            tc1 = tcdata_centered_original[nc, tp]
            Sinput_original.append(tc1)
        Sinput_original = np.array(Sinput_original)

        if fintrinsic_count > 0:
            Sinput_wo_flatent, fixed_var_fraction = remove_component(Sinput, fintrinsic_full)
            Sinput_original_wo_flatent, fixed_var_fraction_original = remove_component(Sinput_original, fintrinsic_full)
            data_sample = {'Sinput':Sinput_wo_flatent, 'Sinput_original':Sinput_original_wo_flatent,
                           'fixed_var_fraction':fixed_var_fraction, 'fixed_var_fraction_original':fixed_var_fraction_original}
        else:
            data_sample = {'Sinput':Sinput, 'Sinput_original':Sinput_original}

        Sinput_data.append(data_sample)

    # find how much variance is accounted for by first few principal components
    fulllist = list(range(len(nclusterlist)))
    smalllist = [x for x in range(len(nclusterlist)) if x not in leaveout]

    # leave out people----------------------------------------
    pplist = list(range(0,NP,np_step))
    NP2 = len(pplist)

    EVrecord = np.zeros((n_components, NP2))
    fixedlatent_EV = np.zeros(NP2)
    starttime = time.time()

    clusteroffset = np.array([0] + list(np.cumsum(nclusterlist)))[:-1]

    nclusterlist_small = np.array(nclusterlist)[smalllist]
    clusteroffset_small = clusteroffset[smalllist]

    pca = sklearn.decomposition.PCA(n_components=n_components)

    # check only one combo
    combo = sub2ind_ndims(nclusterlist_small,cnums_input)
    cc = np.array(cnums_input) + clusteroffset_small
    for nn, pp in enumerate(pplist):
        Sin_sample = copy.deepcopy(Sinput_data[pp]['Sinput'][cc,:])
        pca.fit(Sin_sample)
        EVrecord[:,nn] = pca.explained_variance_ratio_

        if fintrinsic_count > 0:
            fixed_var_fraction = copy.deepcopy(Sinput_data[pp]['fixed_var_fraction'][cc])
            fixedlatent_EV[nn] = np.mean(fixed_var_fraction)

    EVrecord_total = np.sum(EVrecord[:vintrinsic_count,:],axis=0)

    # correlate values per person with covariate
    # EVrecord_avg has size (n_components, NP2)
    if correlate_with_covariate:
        cov2 = covariate[pplist]
        EVrecord_sum = np.sum(EVrecord[:vintrinsic_count,:],axis=0)
        R = np.corrcoef(EVrecord_sum,cov2)
        EVcorr_cov_record = R[0,1]
    else:
        EVcorr_cov_record = 0.0

    return EVrecord, EVrecord_total, EVcorr_cov_record, covariate


def analyze_pca_results(savename, excelsavename = ''):
    # test settings...
    # savename = r'E:\EVrecord_10region_test_V2.npy'

    data = np.load(savename, allow_pickle=True).flat[0]
    keylist = data.keys()

    # data = {'EVrecord_avg': EVrecord_avg, 'fixedlatent_EVrecord_avg': fixedlatent_EVrecord_avg,
    #         'pplist': pplist, 'nclusterlist': nclusterlist, 'leaveout': leaveout,
    #         'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count}

    EVrecord_avg = copy.deepcopy(data['EVrecord_avg'])
    EVcorr_cov_record = copy.deepcopy(data['EVcorr_cov_record'])
    pplist = copy.deepcopy(data['pplist'])
    nclusterlist = copy.deepcopy(data['nclusterlist'])
    leaveout = copy.deepcopy(data['leaveout'])
    fintrinsic_count = copy.deepcopy(data['fintrinsic_count'])
    vintrinsic_count = copy.deepcopy(data['vintrinsic_count'])

    clusteroffset = np.array([0] + list(np.cumsum(nclusterlist)))[:-1]
    smalllist = [x for x in range(len(nclusterlist)) if x not in leaveout]
    nclusterlist_small = np.array(nclusterlist)[smalllist]
    clusteroffset_small = clusteroffset[smalllist]

    EVtotal = np.sum(EVrecord_avg[:,:vintrinsic_count],axis=1)
    if fintrinsic_count > 0:
        fixedlatent_EVrecord_avg = copy.deepcopy(data['fixedlatent_EVrecord_avg'])
        EVtotal += fixedlatent_EVrecord_avg

    # find interesting combinations
    combo_max = np.argmax(EVtotal)
    combo_min = np.argmin(EVtotal)
    combo_sort = np.argsort(EVtotal)
    combo_sort = combo_sort[::-1]

    cnumsx = ind2sub_ndims(nclusterlist_small, combo_max)
    cnumsn = ind2sub_ndims(nclusterlist_small, combo_min)

    found = False
    index = 0
    while not found:
        cnums = ind2sub_ndims(nclusterlist_small, combo_sort[index])
        cdiff = np.abs(cnums-cnumsx)
        zcheck = np.where(cdiff == 0)[0]
        if len(zcheck) <= 1:
            found = True
            cnums_2nd = ind2sub_ndims(nclusterlist_small, combo_sort[index])
            combo_2nd = copy.deepcopy(index)
        else:
            index += 1

    # show results
    EV_max = EVrecord_avg[combo_max,:]
    EV_min = EVrecord_avg[combo_min,:]
    EV_2nd = EVrecord_avg[combo_2nd,:]

    text_max = 'max: cnums: {}  EV  '.format(cnumsx)
    text_min = 'min: cnums: {}  EV  '.format(cnumsn)
    text_2nd = '2nd: cnums: {}  EV  '.format(cnums_2nd)
    for vv in range(vintrinsic_count  + 1):
        text_max += '{:.4f}  '.format(EV_max[vv])
        text_min += '{:.4f}  '.format(EV_min[vv])
        text_2nd += '{:.4f}  '.format(EV_2nd[vv])

    if fintrinsic_count > 0:
        EVf_max = fixedlatent_EVrecord_avg[combo_max]
        EVf_min = fixedlatent_EVrecord_avg[combo_min]
        EVf_2nd = fixedlatent_EVrecord_avg[combo_2nd]

        text_max += '   EVf {:.4f} '.format(EVf_max)
        text_min += '   EVf {:.4f} '.format(EVf_min)
        text_2nd += '   EVf {:.4f} '.format(EVf_2nd)

    print(text_max)
    print(text_2nd)
    print(text_min)

    # look at top 10 cluster combinations
    for index in range(10):
        cnums_sorted = ind2sub_ndims(nclusterlist_small, combo_sort[index])
        EV_sorted = EVrecord_avg[combo_sort[index], :]

        text = 'number {} cnums: {}  EV  '.format(index, cnums_sorted)
        for vv in range(vintrinsic_count + 1):
            text += '{:.4f}  '.format(EV_sorted[vv])

        if fintrinsic_count > 0:
            EVf_sorted = fixedlatent_EVrecord_avg[combo_sort[index]]
            text += '   EVf {:.4f} '.format(EVf_sorted)

        print(text)

    # write out the top combinations-----------------------------
    # create the headings
    headings = ['rank', 'cnums']
    for vv in range(vintrinsic_count + 1):
        headings += ['component{}'.format(vv)]
    if fintrinsic_count > 0:
        headings += ['fixed component']

    # look at top 100 cluster combinations
    topcombodata = []
    for index in range(100):
        cnums_sorted = ind2sub_ndims(nclusterlist_small, combo_sort[index])
        EV_sorted = EVrecord_avg[combo_sort[index], :]

        ranktext = ['{}'.format(index)]
        cnumstext = ['{}'.format(cnums_sorted)]

        componentstext = []
        for vv in range(vintrinsic_count + 1):
            componentstext += ['{:.5f}  '.format(EV_sorted[vv])]

        if fintrinsic_count > 0:
            EVf_sorted = fixedlatent_EVrecord_avg[combo_sort[index]]
            componentstext += '{:.5f} '.format(EVf_sorted)

        alltext = ranktext + cnumstext + componentstext
        entry = dict(zip(headings,alltext))

        topcombodata.append(entry)

    # look at top combinations for each cluster
    # i.e. find all combinations with cluster 2 in region 3 for example
    # and then find the best one of these combinations
    top_percluster_data = []
    for nr in range(len(nclusterlist_small)):
        for cc in range(nclusterlist_small[nr]):
            print('identifying best clusters with subregion {} in region {}'.format(cc,nr))
            m = indices_for_1fixedcluster(nclusterlist_small, nr, cc)
            combo = np.argmax(EVtotal[m])
            EV_this = EVrecord_avg[m[combo], :]
            cnums = ind2sub_ndims(nclusterlist_small,m[combo])

            rankindex = np.where(combo_sort == m[combo])[0]
            ranktext = ['{}'.format(rankindex[0])]
            cnumstext = ['{}'.format(cnums)]

            componentstext = []
            for vv in range(vintrinsic_count + 1):
                componentstext += ['{:.5f}  '.format(EV_this[vv])]

            if fintrinsic_count > 0:
                EVf_this = fixedlatent_EVrecord_avg[m[combo]]
                componentstext += '{:.5f} '.format(EVf_this)

            alltext = ranktext + cnumstext + componentstext
            entry = dict(zip(headings, alltext))

            top_percluster_data.append(entry)


    # look at clusters with top correlation with covariate
    if len(EVcorr_cov_record) > 1:
        corr_sort = np.argsort(EVcorr_cov_record)
        corr_sort = corr_sort[::-1]
        for index in range(10):
            cnums_corr_sorted = ind2sub_ndims(nclusterlist_small, corr_sort[index])
            EV_corr_sorted = EVrecord_avg[corr_sort[index], :]

            text = 'number {} cnums: {}  EV  '.format(index, cnums_corr_sorted)
            for vv in range(vintrinsic_count + 1):
                text += '{:.4f}  '.format(EV_corr_sorted[vv])

            if fintrinsic_count > 0:
                EVf_corr_sorted = fixedlatent_EVrecord_avg[corr_sort[index]]
                text += '   EVf {:.4f} '.format(EVf_corr_sorted)

            print(text)
    else:
        EV_corr_sorted = 0.0
        EVf_corr_sorted = 0.0

    # write out the top combinations-----------------------------
    # create the headings
    headings = ['rank', 'cnums','correlation']
    for vv in range(vintrinsic_count + 1):
        headings += ['component{}'.format(vv)]
    if fintrinsic_count > 0:
        headings += ['fixed component']

    # look at top 100 cluster combinations
    if len(EVcorr_cov_record) > 100:
        topcorrdata = []
        for index in range(100):
            cnums_corr_sorted = ind2sub_ndims(nclusterlist_small, corr_sort[index])
            EV_corr_sorted = EVrecord_avg[corr_sort[index], :]
            corr_val = EVcorr_cov_record[corr_sort[index]]

            rankindex = np.where(combo_sort == index)[0]
            ranktext = ['{}'.format(rankindex[0])]
            cnumstext = ['{}'.format(cnums_corr_sorted)]
            corrtext = ['{:.5f}'.format(corr_val)]

            componentstext = []
            for vv in range(vintrinsic_count + 1):
                componentstext += ['{:.5f}  '.format(EV_corr_sorted[vv])]

            if fintrinsic_count > 0:
                EVf_corr_sorted = fixedlatent_EVrecord_avg[corr_sort[index]]
                componentstext += '{:.5f} '.format(EVf_corr_sorted)

            alltext = ranktext + cnumstext + corrtext + componentstext
            entry = dict(zip(headings,alltext))

            topcorrdata.append(entry)


    # if a name is provided for writing the results to an excel file ...
    f, e = os.path.splitext(excelsavename)
    if e == '.xlsx':  # write the results out
        # now write it to excel sheet
        df1 = pd.DataFrame(topcombodata)
        df2 = pd.DataFrame(top_percluster_data)
        if len(EVcorr_cov_record) > 100:
            df3 = pd.DataFrame(topcorrdata)

        try:
            with pd.ExcelWriter(excelsavename) as writer:
                df1.to_excel(writer, sheet_name='top clusters')
                df2.to_excel(writer, sheet_name='best per cluster')
                if len(EVcorr_cov_record) > 100:
                    df3.to_excel(writer, sheet_name='best corr')
                print('cluster information written to {}'.format(excelsavename))
        except:   # excel file is probably open
            p,f1 = os.path.split(excelsavename)
            f1,e = os.path.splitext(f1)
            timetext = time.ctime()
            timetext = timetext.replace('  ',' ')
            timetext = timetext.replace(':','')
            timetext = timetext.replace(' ','_')
            excelsavename2 = os.path.join(p,f1+timetext+'xlsx')
            with pd.ExcelWriter(excelsavename2) as writer:
                df1.to_excel(writer, sheet_name='top clusters')
                df2.to_excel(writer, sheet_name='best per cluster')
                if len(EVcorr_cov_record) > 100:
                    df3.to_excel(writer, sheet_name='best corr')
                print('cluster information written to {}'.format(excelsavename2))

    return cnumsx

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def ind2sub_ndims(vsize,index):
    # mlist = ind2sub_ndims(vsize, ind)
    ndims = len(vsize)
    m = np.zeros(ndims).astype(int)
    for nn in range(ndims):
        if nn == 0:
            m[0] = np.mod(index,vsize[0])
        else:
            m[nn] = np.mod( np.floor(index/np.prod(vsize[:nn])), vsize[nn])
    return m



def indices_for_1fixedcluster(vsize, fixedindex, fixedvalue):
    # return all of the indices that have one cluster number fixed
    ndims = len(vsize)
    ncombos = np.prod(vsize)
    index = np.array(list(range(ncombos)))

    scales = []
    for nn in range(ndims):
        scales += [np.prod(vsize[:nn])]
    scales = np.array(scales)

    # remove the higher dimensions
    if fixedindex < ndims:
        dimlist = np.array(list(range((fixedindex+1),ndims)))[::-1]
        for d in dimlist:
            m = np.floor(np.mod(np.floor(index / scales[d]), vsize[d])).astype(int)
            index -= m*scales[d]

    # top digit now
    m = np.floor(np.mod(np.floor(index / scales[fixedindex]), vsize[fixedindex])).astype(int)

    c = np.where(m == fixedvalue)[0]
    return c



def sub2ind_ndims(vsize,cnums):
    ndims = len(vsize)
    index = 0
    for nn in range(ndims):
        scale = np.prod(vsize[:nn])
        index += cnums[nn]*scale
    return np.floor(index).astype(int)


def remove_component(S,Sc):
    # S is the signal, with size nr x ntime
    # Sc is the component to remove with size 1 x ntime
    nr, ntime = np.shape(S)
    G = np.concatenate((np.ones((1,ntime)), Sc), axis = 0)
    # S = b*G + residual
    b = S @ G.T @ np.linalg.inv(G @ G.T)

    fit = b @ G
    residual = S - fit

    total_var = np.var(S,axis=1)
    residual_var = np.var(residual,axis=1)
    fit_var = np.var(fit,axis=1)

    var_fraction = fit_var/total_var

    return residual, var_fraction
