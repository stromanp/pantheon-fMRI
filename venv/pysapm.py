"""
pysapm.py

This set of functions is for Structural and Physiological Modeling (SAPM)
which was developed by P. Stroman

Proof-of-concept of a novel structural equation modelling approach for the analysis of
functional MRI data applied to investigate individual differences in human pain responses
P. W. Stroman, J. M. Powers, G. Ioachim
Human Brain Mapping, 44:2523–2542 (2023). https://doi.org/10.1002/hbm.26228

Based on a predefined anatomical network model, including latent inputs, the input and
output signals from every region in a network are modeled as:

Sinput = Minput @ Sconn
Sconn = Mconn @ Sconn    - This is an eigenvalue problem

  Minput is the mixing matrix of deltavalues for each region input in the network
  Mconn is the mixing matrix of betavalues for each connection in the network

The basic steps to solve this are to calculate eigenvectors and latent inputs
then calculate matrix of beta values from those for every time point, each connection
value is determined by a scaled value of the sum of the eigenvectors
so Sconn =  Meigv @ Slatent

The beta values and delta values are determined by means of a gradient descent method
to optimize the fit of the network to the measured data (i.e. Sinput = BOLD responses)

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
import matplotlib.pyplot as plt
import py2ndlevelanalysis
import copy
import pyclustering
import pydisplay
import time
import pysem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
import scipy.linalg as linalg
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d
import random
import draw_sapm_diagram2 as dsd2
import copy
import multiprocessing as mp
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib
import load_templates
from sklearn.linear_model import LinearRegression
import sklearn
import scipy
from scipy.signal import butter, lfilter
from scipy.signal import freqs

plt.rcParams.update({'font.size': 10})

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def load_network_model_w_intrinsics(networkmodel):
    xls = pd.ExcelFile(networkmodel, engine = 'openpyxl')
    dnet = pd.read_excel(xls, 'connections')
    keylist = dnet.keys()
    for nn in range(len(keylist)):
        if 'Unnamed' in keylist[nn]:
            dnet.pop(keylist[nn])   # remove any blank fields
    dnclusters = pd.read_excel(xls, 'nclusters')

    vintrinsic_count = 0
    fintrinsic_count = 0

    nregions = len(dnclusters)
    ntargets, ncols = dnet.shape
    nsources_max = ncols-1

    sem_region_list = []
    nclusterlist = []
    fintrinsic_base = []
    for nn in range(nregions):
        sem_region_list.append(dnclusters.loc[nn,'name'])
        cname = dnclusters.loc[nn,'name']
        if 'vintrinsic' in cname:  vintrinsic_count += 1
        if 'fintrinsic' in cname:  fintrinsic_count += 1
        entry = {'name':dnclusters.loc[nn,'name'],'nclusters':dnclusters.loc[nn,'nclusters']}
        nclusterlist.append(entry)

        # load paradigm for fixed intrinsic input
        if 'fintrinsic' in cname:
            paradigm_data = pd.read_excel(xls, cname)
            del paradigm_data['Unnamed: 0']  # get rid of the unwanted header column
            # get the names of the columns in this sheet ....
            colnames = paradigm_data.keys()

            time = paradigm_data.loc[:, 'time']
            paradigm_names = []

            count = 0
            for num, basisname in enumerate(colnames):
                if basisname != 'time':
                    count += 1
                    paradigm_names.append(basisname)
                    if count == 1:
                        paradigmdef = np.array(paradigm_data.loc[:, basisname])
                        paradigmdef = paradigmdef[np.newaxis, :]
                    else:
                        nextparadigm = paradigm_data.loc[:, basisname]
                        paradigmdef = np.concatenate((paradigmdef, nextparadigm[np.newaxis, :]), axis=0)

            fintrinsic_base = copy.deepcopy(paradigmdef)

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

    return network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base


def sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag):  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv
    nr, ncomponents_to_fit = np.shape(loadings_fit)

    R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
    R2avg = np.mean(R2list)
    R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

    error = np.mean(np.sum((Sinput - fit) ** 2, axis=1) / np.var(Sinput, axis=1))

    cr = np.where(regular_flag > 0)[0]
    cost = np.mean(np.abs(betavals[cr]))  # L1 regularization, ignoring latents
    cost2 = np.mean(np.abs(deltavals-1.0))  # L1 regularization, ignoring latents

    costfactor = Lweight*(cost + cost2)
    ssqd = error + costfactor
    return ssqd, error, cost, costfactor


def gradients_for_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals, deltavals, ctarget,
                              csource, dtarget, dsource, dval, fintrinsic_count, vintrinsic_count,
                              beta_int1, fintrinsic1, Lweight, regular_flag, ncomponents_to_fit = 0):  # , kappavals, ktarget, ksource

    # calculate change in error term with small changes in betavalues
    # include beta_int1
    nbetavals = len(betavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    ndeltavals = len(deltavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    if ncomponents_to_fit < 1:
        nregion,tsize = np.shape(Sinput)
        ncomponents_to_fit = copy.deepcopy(nregion)

    fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                        Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
    # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv


    # gradients for betavals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        ssqdp, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, b, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv


        b = copy.deepcopy(betavals)
        b[nn] -= dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        ssqdp2, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, b, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        dssq_db[nn] = (ssqdp - ssqdp2) /( dval)

    # gradients for deltavals
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    dssq_dd = np.zeros(ndeltavals)
    for nn in range(ndeltavals):
        d = copy.deepcopy(deltavals)
        d[nn] += dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        ssqdp, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, d, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        d = copy.deepcopy(deltavals)
        d[nn] -= dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        ssqdp2, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, d, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        dssq_dd[nn] = (ssqdp - ssqdp2) /( dval)

    # gradients for beta_int1
    b = copy.deepcopy(beta_int1)
    b += dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1, ncomponents_to_fit)
    # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    ssqdp, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    b = copy.deepcopy(beta_int1)
    b -= dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)

    fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1, ncomponents_to_fit)
    # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    ssqdp2, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    # dssq_dbeta1 = (ssqdp - ssqd) / (dval)
    dssq_dbeta1 = (ssqdp - ssqdp2) / (dval)

    return dssq_db, dssq_dd, ssqd, dssq_dbeta1


def update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals, deltavals, betalimit, ctarget, csource,
                       dtarget, dsource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag,
                       alpha, alphabint, ncomponents_to_fit = 0, latent_flag = []):   #, kappavals, ktarget, ksource

    # calculate change in error term with small changes in betavalues
    # include beta_int1
    nregion,tsize = np.shape(Sinput)
    change_weight_limit = 0.1
    if ncomponents_to_fit < 1:
        ncomponents_to_fit = copy.deepcopy(nregion)

    starting_betavals = copy.deepcopy(betavals)
    starting_deltavals = copy.deepcopy(deltavals)
    starting_beta_int1 = copy.deepcopy(beta_int1)

    if len(latent_flag) < len(betavals): latent_flag = np.zeros(len(betavals))
    nbetavals = len(betavals)
    updatebflag = np.zeros(nbetavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    # gradients in beta vals
    dssq_db, dssq_dd, ssqd, dssq_dbeta1 = gradients_for_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals,
                                        deltavals, ctarget, csource, dtarget, dsource, dval,
                                        fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, ncomponents_to_fit)

    # c = np.where(dsource >= nregion)[0]   # latent inputs
    # dssq_dd[c] = 0.0   # over-ride changing the delta values from 1.0, for latent inputs
    c = np.where(csource >= nregion)[0]   # latent inputs
    dssq_db[c] = 0.0   # over-ride changing the beta values from 1.0, for latent inputs

    change_betavals = alpha * dssq_db
    change_deltavals = 0.2*alpha * dssq_dd

    change_betavals[change_betavals > change_weight_limit] = change_weight_limit
    change_betavals[change_betavals < -change_weight_limit] = -change_weight_limit
    change_deltavals[change_deltavals > change_weight_limit] = change_weight_limit
    change_deltavals[change_deltavals < -change_weight_limit] = -change_weight_limit

    betavals -= change_betavals
    deltavals -= change_deltavals

    betavals[betavals > betalimit] = copy.deepcopy(betalimit)
    betavals[betavals < -betalimit] = copy.deepcopy(-betalimit)
    beta_int1 -= alphabint * dssq_dbeta1

    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                        Mconn, fintrinsic_count, vintrinsic_count, beta_int1,
                                                        fintrinsic1, ncomponents_to_fit)
    # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    ssqd_new, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    return betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd, ssqd_new, alpha, alphabint

def network_eigenvector_method_V3(Sinput, components, loadings, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit = 0):
    # # Sinput[:nregions,:] = Minput @ Meigv @ Mintrinsic   #  [nr x tsize] = [nr x ncon] @ [ncon x Nint] @ [Nint x tsize]
    # # Mintrinsic = np.linalg.inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sin
    # fit based on eigenvectors alone, with intrinsic values calculated
    if ncomponents_to_fit < 1:
        nregion,tsize = np.shape(Sinput)
        ncomponents_to_fit = copy.deepcopy(nregion)

    nregions,tsize_total = np.shape(Sinput)
    Nintrinsic = fintrinsic_count + vintrinsic_count
    e,v = np.linalg.eig(Mconn)    # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
    Meigv = np.real(v[:,-Nintrinsic:])
    # Meigv = (Gram_Schmidt_orthogonalization(Meigv.T)).T  # make them a set of linearly indpendent eigvenvectors
    # scale to make the term corresponding to each intrinsic = 1
    for aa in range(Nintrinsic):
        Meigv[:,aa] = Meigv[:,aa]/Meigv[(-Nintrinsic+aa),aa]

    if fintrinsic_count > 0:
        # separate fintrinsic1 from components and loadings
        # components = ff @ fintrinsic1   # fit
        f1 = fintrinsic1[np.newaxis,:]
        ff = components @ f1.T @ np.linalg.inv(f1 @ f1.T)
        # print('ff =  {}'.format(ff))
        componentsR = components - ff @ f1

        # Sinput = loadings @ components
        loadingsR = Sinput @ componentsR.T @ np.linalg.inv(componentsR @ componentsR.T)
        X = loadingsR[:, :ncomponents_to_fit]  # reduced number of loadings to represent Sinput

        # fit the fixed intrinsic, remove it, and then fit the variable intrinsics to the remainder
        Mintrinsic = np.zeros((Nintrinsic, tsize_total))

        # Mint_variable = np.linalg.inv(M1[:,1:].T @ M1[:,1:]) @ M1[:,1:].T @ Sinput
        Mint_fixed = beta_int1*fintrinsic1[np.newaxis,:]

        partial_fit = (Minput @ Meigv[:,0])[:,np.newaxis] @ Mint_fixed
        residual = Sinput-partial_fit
        M1r = Minput @ Meigv[:,1:]

        Mint_variable = np.linalg.inv(M1r.T @ M1r) @ M1r.T @ residual
        W = np.linalg.inv(M1r.T @ M1r) @ M1r.T @ X   # fitting PC loadings

        # make Mint_variable components linearly independent--------------------------
        # Wlimited = np.zeros(np.shape(W))
        # for ww in range(np.shape(W)[1]):
        #     wtemp = W[:,ww]
        #     # wmax = np.max(np.abs(wtemp))
        #     # scale = np.tanh(np.abs(wtemp)/wmax)/np.tanh(1)   # scale down the components with smaller contributions
        #     # Wlimited[:,ww] = scale*wtemp   # need to scale the contributions so that gradient descent can still find the optimal solution
        #     x = np.argmax(np.abs(wtemp))
        #     Wlimited[x,ww] = wtemp[x]
        # W = copy.deepcopy(Wlimited)
        #------------------------------------------------------------------------------

        # Mint_variable = W @ componentsR[:ncomponents_to_fit,:]

        Mintrinsic[0,:] = copy.deepcopy(Mint_fixed)
        Mintrinsic[1:,:] = copy.deepcopy(Mint_variable)

        fit = Minput @ Meigv @ Mintrinsic
        loadings_fit = Minput @ Meigv[:,1:] @ W

    else:
        M1 = Minput @ Meigv
        # Mintrinsic = np.linalg.inv(M1.T @ M1) @ M1.T @ Sinput

        X = loadings[:, :ncomponents_to_fit]  # reduced number of loadings to represent Sinput
        W = np.linalg.inv(M1.T @ M1) @ M1.T @ X   # fitting PC loadings

        Mint_variable = np.linalg.inv(M1.T @ M1) @ M1.T @ Sinput

        # make Mint_variable components linearly independent--------------------------
        # Wlimited = np.zeros(np.shape(W))
        # for ww in range(np.shape(W)[1]):
        #     wtemp = W[:,ww]
        #     # wmax = np.max(np.abs(wtemp))
        #     # scale = np.tanh(np.abs(wtemp)/wmax)/np.tanh(1)   # scale down the components with smaller contributions
        #     # Wlimited[:,ww] = scale*wtemp   # need to scale the contributions so that gradient descent can still find the optimal solution
        #     x = np.argmax(np.abs(wtemp))
        #     Wlimited[x,ww] = wtemp[x]
        # W = copy.deepcopy(Wlimited)
        #------------------------------------------------------------------------------

        # Mintrinsic = W @ components[:ncomponents_to_fit,:]
        Mintrinsic = copy.deepcopy(Mint_variable)

        fit = Minput @ Meigv @ Mintrinsic
        loadings_fit = Minput @ Meigv @ W

    err = np.sum((Sinput - fit)**2)

    return fit, loadings_fit, W, Mintrinsic, Meigv, err

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


def load_filtered_cluster_properties(clusterdataname, networkfile):
    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = cluster_data['cluster_properties']
    clusterregionlist = [cluster_properties[x]['rname'] for x in range(len(cluster_properties))]

    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    networktargetlist = [network[x]['target'] for x in range(len(network))]

    filtered_cluster_properties = []
    for nn, targetname in enumerate(networktargetlist):
        if targetname in clusterregionlist:
            x = clusterregionlist.index(targetname)
            filtered_cluster_properties.append(cluster_properties[x])

    return filtered_cluster_properties

#---------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint = 'all', epoch = 'all', fullgroup = False, normalizevar = False):

    outputdir, f = os.path.split(SAPMparametersname)
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)

    fintrinsic_region = []
    if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
        for nn in range(len(network)):
            sources = network[nn]['sources']
            if 'fintrinsic1' in sources:
                fintrinsic_region = network[nn]['targetnum']  # only one region should have this input

    region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
    region_properties = region_data1['region_properties']

    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    # cluster_properties = cluster_data['cluster_properties']
    cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)

    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
    nclusterstotal = np.sum(nclusterlist)

    tsize = region_properties[0]['tsize']
    nruns_per_person = region_properties[0]['nruns_per_person']
    nruns_total = np.sum(nruns_per_person)
    NP = len(nruns_per_person)  # number of people in the data set

    tcdata = []
    if normalizevar:
        tcdata_std = np.zeros((nclusterstotal,NP))
    for i in range(nregions):
        tc = region_properties[i]['tc']
        if i == 0:
            tcdata = tc
        else:
            tcdata = np.append(tcdata, tc, axis=0)

    # setup index lists---------------------------------------------------------------------------
    # timepoints for full runs----------------------------------------------
    if timepoint == 'all':
        epoch = tsize
        timepoint = np.floor(tsize/2)

    tplist_full = []
    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        if np.floor(epoch/2).astype(int) == np.ceil(epoch/2).astype(int):   # even numbered epoch
            et1 = (timepoint - np.floor(epoch / 2)).astype(int)
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)
        else:
            et1 = (timepoint - np.floor(epoch / 2)).astype(int)
            et2 = (timepoint + np.floor(epoch / 2)).astype(int) + 1
    if et1 < 0: et1 = 0
    if et2 > tsize: et2 = tsize
    epoch = et2-et1

    dtsize = tsize - 1  # for using deriviation of tc wrt time
    tplist1 = []
    nclusterstotal, tsizetotal = np.shape(tcdata)
    tcdata_centered = copy.deepcopy(tcdata)
    for nn in range(NP):
        r1 = sum(nruns_per_person[:nn])
        r2 = sum(nruns_per_person[:(nn + 1)])
        tp = []  # initialize list
        tpoints = []
        for ee2 in range(r1, r2):
            # tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
            tp = list(range((ee2*tsize+et1),(ee2*tsize+et2)))
            tpoints = tpoints + tp  # concatenate lists
            temp = np.mean(tcdata[:, tp], axis=1)
            temp_mean = np.repeat(temp[:, np.newaxis], epoch, axis=1)
            tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean  # center each epoch, in each person
        tplist1.append({'tp': tpoints})

        if normalizevar:
            # # normalize the data to have the same variance, for each person
            tcdata_std[:,nn] = np.std(tcdata_centered[:,tpoints],axis=1)
            # scale_factor = np.repeat(tcdata_std[:,nn][:,np.newaxis],len(tpoints),axis=1)
            scale_factor = np.ones(np.shape(scale_factor))
            tcdata_centered[:, tpoints] /= scale_factor

    tplist_full.append(tplist1)

    if fullgroup:
        # special case to fit the full group together
        # treat the whole group like one person
        tpgroup_full = []
        tpgroup = []
        tp = []
        for nn in range(NP):
            tp += tplist_full[0][nn]['tp']   # concatenate timepoint lists
        tpgroup.append({'tp': tp})
        tpgroup_full.append(tpgroup)
        tplist_full = copy.deepcopy(tpgroup_full)
        nruns_per_person = [np.sum(nruns_per_person)]

    Nintrinsic = fintrinsic_count + vintrinsic_count
    nregions = len(rnamelist)

    beta_list = []
    nbeta = 0
    targetnumlist = []
    beta_id = []
    sourcelist = []
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        targetnumlist += [target]
        for mm in range(len(sources)):
            source = sources[mm]
            sourcelist += [source]
            betaname = '{}_{}'.format(source, target)
            entry = {'name': betaname, 'number': nbeta, 'pair': [source, target]}
            beta_list.append(entry)
            beta_id += [1000 * source + target]
            nbeta += 1

    ncon = nbeta - Nintrinsic

    # reorder to put intrinsic inputs at the end-------------
    beta_list2 = []
    beta_id2 = []
    x = np.where(np.array(sourcelist) < nregions)[0]
    for xx in x:
        beta_list2.append(beta_list[xx])
        beta_id2 += [beta_id[xx]]
    for sn in range(nregions, nregions + Nintrinsic):
        x = np.where(np.array(sourcelist) == sn)[0]
        for xx in x:
            beta_list2.append(beta_list[xx])
            beta_id2 += [beta_id[xx]]

    for nn in range(len(beta_list2)):
        beta_list2[nn]['number'] = nn

    beta_list = beta_list2
    beta_id = beta_id2

    beta_pair = []
    Mconn = np.zeros((nbeta, nbeta))
    count = 0
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        for mm in range(len(sources)):
            source = sources[mm]
            conn1 = beta_id.index(source * 1000 + target)
            if source >= nregions:  # intrinsic input
                conn2 = conn1
                Mconn[conn1, conn2] = 1  # set the intrinsic beta values
            else:
                x = targetnumlist.index(source)
                source_sources = network[x]['sourcenums']
                for nn in range(len(source_sources)):
                    ss1 = source_sources[nn]
                    conn2 = beta_id.index(ss1 * 1000 + source)
                    beta_pair.append([conn1, conn2])
                    count += 1
                    Mconn[conn1, conn2] = count


    # prep to index Mconn for updating beta values
    beta_pair = np.array(beta_pair)
    ctarget = beta_pair[:, 0]
    csource = beta_pair[:, 1]

    latent_flag = np.zeros(len(ctarget))

    reciprocal_flag = np.zeros(len(ctarget))
    for nn in range(len(ctarget)):
        spair = beta_list[csource[nn]]['pair']
        tpair = beta_list[ctarget[nn]]['pair']
        if spair[0] == tpair[1]:
            reciprocal_flag[nn] = 1

    # setup Minput matrix--------------------------------------------------------------
    # Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
    # Sinput = Minput @ Mconn
    Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
    betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        for mm in range(len(sources)):
            source = sources[mm]
            betaname = '{}_{}'.format(source, target)
            x = betanamelist.index(betaname)
            Minput[target, x] = 1

    # save parameters for looking at results later
    SAPMparams = {'betanamelist': betanamelist, 'beta_list': beta_list, 'nruns_per_person': nruns_per_person,
                 'nclusterstotal': nclusterstotal, 'rnamelist': rnamelist, 'nregions': nregions,
                 'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
                 'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
                 'fintrinsic_region':fintrinsic_region, 'sem_region_list': sem_region_list,
                 'nclusterlist': nclusterlist, 'tsize': tsize, 'tplist_full': tplist_full,
                 'tcdata_centered': tcdata_centered, 'ctarget':ctarget ,'csource':csource,
                 'Mconn':Mconn, 'Minput':Minput, 'timepoint':timepoint, 'epoch':epoch, 'latent_flag':latent_flag,
                  'reciprocal_flag':reciprocal_flag, 'fintrinsic_base':fintrinsic_base}   #, 'tcdata_std':tcdata_std
    if normalizevar:
        SAPMparams['tcdata_std'] = tcdata_std
    print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
    np.save(SAPMparametersname, SAPMparams)

# prep data for single output model
#---------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint = 'all',
                                  epoch = 'all', fullgroup = False, normalizevar = False, filter_tcdata = False):
# model each region as having a single output that is common to all regions it projects to
# But ...(new for V2) allow for scaling of the input to each region, by varying the values in Minput
#
    outputdir, f = os.path.split(SAPMparametersname)
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)

    fintrinsic_region = []
    if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
        for nn in range(len(network)):
            sources = network[nn]['sources']
            if 'fintrinsic1' in sources:
                fintrinsic_region = network[nn]['targetnum']  # only one region should have this input

    region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
    region_properties = copy.deepcopy(region_data1['region_properties'])
    DBname = copy.deepcopy(region_data1['DBname'])
    DBnum = copy.deepcopy(region_data1['DBnum'])

    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)

    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
    nclusterstotal = np.sum(nclusterlist)

    tsize = region_properties[0]['tsize']
    nruns_per_person = region_properties[0]['nruns_per_person']
    nruns_total = np.sum(nruns_per_person)
    NP = len(nruns_per_person)  # number of people in the data set

    tcdata = []
    for i in range(nregions):
        tc = region_properties[i]['tc']
        if i == 0:
            tcdata = tc
        else:
            tcdata = np.append(tcdata, tc, axis=0)

    print('size of tcdata is {}'.format(np.shape(tcdata)))

    # setup index lists---------------------------------------------------------------------------
    # timepoints for full runs----------------------------------------------
    if timepoint == 'all':
        epoch = tsize
        timepoint = np.floor(tsize/2)

    tplist_full = []
    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        if np.floor(epoch/2).astype(int) == np.ceil(epoch/2).astype(int):   # even numbered epoch
            et1 = (timepoint - np.floor(epoch / 2)).astype(int)
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)
        else:
            et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    if et1 < 0: et1 = 0
    if et2 > tsize: et2 = tsize
    epoch = et2-et1

    dtsize = tsize - 1  # for using deriviation of tc wrt time
    tplist1 = []
    nclusterstotal, tsizetotal = np.shape(tcdata)
    tcdata_centered = copy.deepcopy(tcdata)
    tcdata_centered_original = copy.deepcopy(tcdata)
    for nn in range(NP):
        r1 = sum(nruns_per_person[:nn])
        r2 = sum(nruns_per_person[:(nn + 1)])
        tp = []  # initialize list
        tpoints = []
        for ee2 in range(r1, r2):
            # tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
            tp = list(range((ee2*tsize+et1),(ee2*tsize+et2)))
            tpoints = tpoints + tp  # concatenate lists
            temp = np.mean(tcdata[:, tp], axis=1)
            temp_mean = np.repeat(temp[:, np.newaxis], epoch, axis=1)
            tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean  # center each epoch, in each person

            if filter_tcdata:
                # low-pass filter the tcdata
                fs = 1.0  # use relative frequencies, to allow for different sampling rates
                cutoff = 0.2
                for ii in range(nregions):
                    tcdata_centered[ii, tp] = butter_lowpass_filter(tcdata_centered[ii, tp], cutoff, fs, order=5)

        tplist1.append({'tp': tpoints})

    # apply normalization to scale variance to within a desired range, while keeping the relative order and spacing of
    # of variance differences between regions/persons
    if normalizevar:
        tcdata_std = np.zeros((nclusterstotal, NP))
        for nn in range(NP):
            tpoints = copy.deepcopy(tplist1[nn]['tp'])
            # normalize the data to have the same variance, for each person
            tcdata_std[:,nn] = np.std(tcdata_centered[:,tpoints],axis=1)

        avg_std = np.mean(tcdata_std)
        max_std = np.max(tcdata_std)
        min_std = np.min(tcdata_std)

        new_min_std = 0.75*avg_std
        new_max_std = 1.25*avg_std

        new_std = copy.deepcopy(avg_std)    # normalize everything to the same variance
        std_scale = new_std/tcdata_std

        for nn in range(NP):
            tpoints = copy.deepcopy(tplist1[nn]['tp'])
            scale_factor = np.repeat(std_scale[:,nn][:,np.newaxis],len(tpoints),axis=1)
            # scale_factor = np.repeat(tcdata_std[:,nn][:,np.newaxis],len(tpoints),axis=1)/avg_std
            tcdata_centered[:, tpoints] *= scale_factor
    else:
        tcdata_std = []
        std_scale = []

    tplist_full.append(tplist1)

    if fullgroup:
        # special case to fit the full group together
        # treat the whole group like one person
        tpgroup_full = []
        tpgroup = []
        tp = []
        for nn in range(NP):
            tp += tplist_full[0][nn]['tp']   # concatenate timepoint lists
        tpgroup.append({'tp': tp})
        tpgroup_full.append(tpgroup)
        tplist_full = copy.deepcopy(tpgroup_full)
        nruns_per_person = [np.sum(nruns_per_person)]

    Nintrinsic = fintrinsic_count + vintrinsic_count
    nregions = len(rnamelist)

    beta_list = []
    nbeta = 0
    targetnumlist = []
    beta_id = []
    sourcelist = []
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        targetnumlist += [target]
        for mm in range(len(sources)):
            source = sources[mm]
            sourcelist += [source]
            betaname = '{}_{}'.format(source, target)
            entry = {'name': betaname, 'number': nbeta, 'pair': [source, target]}
            beta_list.append(entry)
            beta_id += [1000 * source + target]
            nbeta += 1

    ncon = nbeta - Nintrinsic

    # reorder to put intrinsic inputs at the end-------------
    beta_list2 = []
    beta_id2 = []
    x = np.where(np.array(sourcelist) < nregions)[0]
    for xx in x:
        beta_list2.append(beta_list[xx])
        beta_id2 += [beta_id[xx]]
    for sn in range(nregions, nregions + Nintrinsic):
        x = np.where(np.array(sourcelist) == sn)[0]
        for xx in x:
            beta_list2.append(beta_list[xx])
            beta_id2 += [beta_id[xx]]

    for nn in range(len(beta_list2)):
        beta_list2[nn]['number'] = nn

    beta_list = beta_list2
    beta_id = beta_id2

    beta_pair = []
    Mconn = np.zeros((nregions + Nintrinsic, nregions + Nintrinsic))
    count = 0
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        for mm in range(len(sources)):
            source = sources[mm]
            conn1 = beta_id.index(source * 1000 + target)

            count += 1
            beta_pair.append([target, source])
            Mconn[target, source] = count

            if source >= nregions:  # intrinsic input
                Mconn[source, source] = 1  # set the intrinsic beta values

    # prep to index Mconn for updating beta values
    beta_pair = np.array(beta_pair)
    ctarget = beta_pair[:, 0]
    csource = beta_pair[:, 1]

    latent_flag = np.zeros(len(ctarget))
    found_latent_list = []
    for nn in range(len(ctarget)):
        # if csource[nn] >= ncon  and ctarget[nn] < ncon:
        if csource[nn] >= nregions  and ctarget[nn] < nregions:
            found_latent_list += [csource[nn]]
            occurence = np.count_nonzero(found_latent_list == csource[nn])
            latent_flag[nn] = csource[nn]-nregions+1

    reciprocal_flag = np.zeros(len(ctarget))
    for nn in range(len(ctarget)):
        spair = beta_list[csource[nn]]['pair']
        tpair = beta_list[ctarget[nn]]['pair']
        if spair[0] == tpair[1]:
            reciprocal_flag[nn] = 1

    # setup Minput matrix--------------------------------------------------------------
    # Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
    # Sinput = Minput @ Mconn
    # Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
    Minput = np.zeros((nregions, nregions+Nintrinsic))  # mixing of connections to model the inputs to each region
    betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']
        for mm in range(len(sources)):
            source = sources[mm]
            # betaname = '{}_{}'.format(source, target)
            # x = betanamelist.index(betaname)
            # Minput[target, x] = 1
            Minput[target, source] = 1

    # flag which Minput values can be varied
    # (keep one output from each region at 1, vary the other outputs)
    Dvarflag = copy.deepcopy(Minput)
    for nn in range(nregions):
        onesource = copy.deepcopy(Minput[:,nn])
        c = np.where(onesource > 0)[0]
        if len(c) > 0: onesource[c[0]] = 0
        Dvarflag[:, nn] = copy.deepcopy(onesource)
    for nn in range(nregions,nregions+Nintrinsic):
        Dvarflag[:,nn] = 0
    dtarget,dsource = np.where(Dvarflag > 0)

    # save parameters for looking at results later
    SAPMparams = {'betanamelist': betanamelist, 'beta_list': beta_list, 'nruns_per_person': nruns_per_person,
                 'nclusterstotal': nclusterstotal, 'rnamelist': rnamelist, 'nregions': nregions,
                 'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
                 'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
                 'fintrinsic_region':fintrinsic_region, 'sem_region_list': sem_region_list,
                 'nclusterlist': nclusterlist, 'tsize': tsize, 'tplist_full': tplist_full,
                 'tcdata_centered': tcdata_centered, 'tcdata_centered_original': tcdata_centered_original,
                  'ctarget':ctarget ,'csource':csource, 'dtarget':dtarget ,'dsource':dsource,
                 'Mconn':Mconn, 'Minput':Minput, 'timepoint':timepoint, 'epoch':epoch, 'latent_flag':latent_flag,
                  'reciprocal_flag':reciprocal_flag, 'fintrinsic_base':fintrinsic_base, 'DBname':DBname, 'DBnum':DBnum}  # , 'ktarget':ktarget ,'ksource':ksource
    if normalizevar:
        SAPMparams['tcdata_std'] = tcdata_std
        SAPMparams['std_scale'] = std_scale

    print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
    np.save(SAPMparametersname, SAPMparams)

#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model1_V3(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [],
                      betascale = 0.1, Lweight = 1.0, normalizevar=False, nitermax = 300, verbose = True, initial_nitermax_stage1 = 150,
                      initial_nsteps_stage1 = 20):

# this version fits to principal components of Sinput

    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    initial_alpha = 1e-3
    initial_Lweight = copy.deepcopy(Lweight)
    initial_dval = 0.05
    alpha_limit = 1.0e-5
    repeat_limit = 2

    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    # load the data values
    betanamelist = SAPMparams['betanamelist']
    beta_list = SAPMparams['beta_list']
    nruns_per_person = SAPMparams['nruns_per_person']
    nclusterstotal = SAPMparams['nclusterstotal']
    rnamelist = SAPMparams['rnamelist']
    nregions = SAPMparams['nregions']
    cluster_properties = SAPMparams['cluster_properties']
    cluster_data = SAPMparams['cluster_data']
    network = SAPMparams['network']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    sem_region_list = SAPMparams['sem_region_list']
    nclusterlist = SAPMparams['nclusterlist']
    tsize = SAPMparams['tsize']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    tcdata_centered_original = SAPMparams['tcdata_centered_original']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    dtarget = SAPMparams['dtarget']
    dsource = SAPMparams['dsource']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    Mconn = SAPMparams['Mconn']
    Minput = SAPMparams['Minput']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    latent_flag = SAPMparams['latent_flag']
    reciprocal_flag = SAPMparams['reciprocal_flag']
    DBname = SAPMparams['DBname']
    DBnum = SAPMparams['DBnum']

    regular_flag = 1-latent_flag   # flag where connections are not latent or reciprocal

    ntime, NP = np.shape(tplist_full)
    Nintrinsics = vintrinsic_count + fintrinsic_count

    ncomponents_to_fit = copy.deepcopy(nregions)
#---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 3.0
    epochnum = 0
    SAPMresults = []
    first_pass_results = []
    second_pass_results = []
    beta_init_record = []
    for nperson in range(NP):
        if verbose:
            print('starting person {} at {}'.format(nperson,time.ctime()))
        tp = tplist_full[epochnum][nperson]['tp']
        tsize_total = len(tp)
        nruns = nruns_per_person[nperson]

        # get tc data for each region/cluster
        rnumlist = []
        clustercount = np.cumsum(nclusterlist)
        for aa in range(len(clusterlist)):
            x = np.where(clusterlist[aa] < clustercount)[0]
            rnumlist += [x[0]]

        Sinput = []
        for nc,cval in enumerate(clusterlist):
            tc1 = tcdata_centered[cval, tp]
            Sinput.append(tc1)
        Sinput = np.array(Sinput)

        Sinput_original = []
        for nc, cval in enumerate(clusterlist):
            tc1 = tcdata_centered_original[cval, tp]
            Sinput_original.append(tc1)
        Sinput_original = np.array(Sinput_original)

        # get principal components of Sinput--------------------------
        nr = np.shape(Sinput)[0]
        pca = sklearn.decomposition.PCA()
        pca.fit(Sinput)
        components = pca.components_
        loadings = pca.transform(Sinput)
        mu2 = np.mean(Sinput, axis=0)
        loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
        components = np.concatenate((mu2[np.newaxis, :], components), axis=0)
        # test_fit = loadings @ components

        # setup fixed intrinsic based on the model paradigm
        # need to account for timepoint and epoch....
        if fintrinsic_count > 0:
            if epoch >= tsize:
                et1 = 0
                et2 = tsize
            else:
                if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
                    et1 = (timepoint - np.floor(epoch / 2)).astype(int)
                    et2 = (timepoint + np.floor(epoch / 2)).astype(int)
                else:
                    et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
                    et2 = (timepoint + np.floor(epoch / 2)).astype(int)
            if et1 < 0: et1 = 0
            if et2 > tsize: et2 = tsize
            epoch = et2 - et1

            ftemp = fintrinsic_base[0,et1:et2]
            fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
            # print('shape of fintrinsic1 is {}'.format(np.shape(fintrinsic1)))
            if np.var(ftemp) > 1.0e-3:
                Sint = Sinput[fintrinsic_region,:]
                Sint = Sint - np.mean(Sint)
                # need to add constant to fit values
                G = np.concatenate((fintrinsic1[np.newaxis,:],np.ones((1,tsize_total))),axis=0)
                b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
                beta_int1 = b[0]
            else:
                beta_int1 = 0.0
        else:
            beta_int1 = 0.0
            fintrinsic1 = []

        lastgood_beta_int1 = copy.deepcopy(beta_int1)

        # initialize beta values-----------------------------------
        nbeta = len(csource)
        if isinstance(betascale,str):
            if betascale == 'shotgun':
                beta_initial = betaval_init_shotgun(initial_Lweight, csource, ctarget, Sinput, Minput, Mconn, components,
                                    loadings, fintrinsic_count, vintrinsic_count, np.ones(len(dtarget)), beta_int1, fintrinsic1,
                                                    ncomponents_to_fit, nreps=10000)

                beta_initial = beta_initial[np.newaxis,:]
                nitermax_stage1 = 0
            else:
                # read saved beta_initial values
                b = np.load(betascale,allow_pickle=True).flat[0]
                beta_initial = b['beta_initial']
                beta_initial = beta_initial[np.newaxis,:]
                nitermax_stage1 = 0
            nsteps_stage1 = 1
        else:
            nsteps_stage1 = copy.deepcopy(initial_nsteps_stage1)
            beta_initial = betascale*np.random.randn(nsteps_stage1,nbeta)
            nregion,ntotal = np.shape(Minput)

            nitermax_stage1 = copy.deepcopy(initial_nitermax_stage1)

        # initialize deltavals
        delta_initial = np.ones(len(dtarget))
        deltascale = np.std(Sinput,axis=1)
        meanscale = np.mean(deltascale)

        # initialize
        results_record = []
        ssqd_record = []

        # stage 1 - test the initial betaval settings
        stage1_ssqd = np.zeros(nsteps_stage1)
        stage1_results = []
        for ns in range(nsteps_stage1):
            ssqd_record_stage1 = []
            beta_init_record.append({'beta_initial':beta_initial[ns,:]})

            # initalize Sconn
            betavals = copy.deepcopy(beta_initial[ns,:]) # initialize beta values at zero
            lastgood_betavals = copy.deepcopy(betavals)
            deltavals = copy.deepcopy(delta_initial)
            lastgood_deltavals = copy.deepcopy(deltavals)

            alphalist = initial_alpha*np.ones(nbeta)
            alphabint = copy.deepcopy(initial_alpha)
            alpha = copy.deepcopy(initial_alpha)
            Lweight = copy.deepcopy(initial_Lweight)
            dval = copy.deepcopy(initial_dval)

            # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
            Mconn[ctarget,csource] = copy.deepcopy(betavals)
            Minput[dtarget, dsource] = copy.deepcopy(deltavals)
            fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
            # Soutput = Meigv @ Mintrinsic  # signalling over each connection
            ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

            ssqd_starting = copy.deepcopy(ssqd)
            ssqd_old = copy.deepcopy(ssqd)
            ssqd_record += [ssqd]

            iter = 0
            converging = True
            dssq_record = np.ones(3)
            dssq_count = 0
            sequence_count = 0

            while alpha > alpha_limit and iter < nitermax_stage1 and converging:
                iter += 1

                betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
                                                    update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals,
                                                    deltavals, betalimit, ctarget, csource, dtarget, dsource,
                                                    dval,fintrinsic_count,
                                                    vintrinsic_count, beta_int1,fintrinsic1, Lweight, regular_flag, alpha,alphabint,
                                                    ncomponents_to_fit, latent_flag=latent_flag)  # kappavals, ktarget, ksource,

                ssqd_record_stage1 += [ssqd]

                if ssqd > ssqd_original:
                    alpha *= 0.5
                    alphabint *= 0.5
                    betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
                    deltavals = copy.deepcopy(lastgood_deltavals)
                    beta_int1 = copy.deepcopy(lastgood_beta_int1)
                else:
                    lastgood_betavals = copy.deepcopy(betavals)
                    lastgood_deltavals = copy.deepcopy(deltavals)
                    lastgood_beta_int1 = copy.deepcopy(beta_int1)

                Mconn[ctarget, csource] = copy.deepcopy(betavals)
                Minput[dtarget, dsource] = copy.deepcopy(deltavals)

                fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                    Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
                                                    ncomponents_to_fit)
                # Soutput = Meigv @ Mintrinsic  # signalling over each connection
                ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

                err_total = Sinput - fit
                Smean = np.mean(Sinput)
                errmean = np.mean(err_total)

                R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
                R2avg = np.mean(R2list)
                R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

                results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

                ssqchange = ssqd - ssqd_original

                if verbose:
                    print('SAPM  {} stage1 pass {} iter {} alpha {:.3e}  ssqd {:.2f} error {:.2f} error2 {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson,
                                    ns, iter, alpha, ssqd, error, error2, ssqchange, 100.*ssqd/ssqd_starting, R2avg, R2total))
                ssqd_old = copy.deepcopy(ssqd)
                # now repeat it ...
            stage1_ssqd[ns] = ssqd
            stage1_results.append({'betavals':betavals, 'deltavals':deltavals})

        # get the best betavals from stage1 so far ...
        x = np.argmin(stage1_ssqd)
        betavals = stage1_results[x]['betavals']
        deltavals = stage1_results[x]['deltavals']

        # stage 2
        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        if verbose: print('starting stage 2 ....')
        lastgood_betavals = copy.deepcopy(betavals)
        alpha = copy.deepcopy(initial_alpha)
        alphabint = copy.deepcopy(initial_alpha)
        Lweight = copy.deepcopy(initial_Lweight)
        dval = copy.deepcopy(initial_dval)

        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)

        fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
                                                ncomponents_to_fit)
        # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv


        ssqd_starting = copy.deepcopy(ssqd)
        ssqd_old = copy.deepcopy(ssqd)
        ssqd_record += [ssqd]

        iter = 0
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        sequence_count = 0

        while alpha > alpha_limit and iter < nitermax and converging:
            iter += 1
            betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
                update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals, deltavals, betalimit,
                                   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
                                   vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
                                   ncomponents_to_fit, latent_flag=latent_flag)   #, kappavals, ktarget, ksource

            ssqd_record_stage1 += [ssqd]

            if ssqd > ssqd_original:
                alpha *= 0.5
                alphabint *= 0.5
                betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
                deltavals = copy.deepcopy(lastgood_deltavals)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                sequence_count = 0
            else:
                lastgood_betavals = copy.deepcopy(betavals)
                lastgood_deltavals = copy.deepcopy(deltavals)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)
                sequence_count += 1
                if sequence_count > 3:
                    alpha *= 1.3
                    alphabint *= 1.3
                    sequence_count = 0

            Mconn[ctarget, csource] = copy.deepcopy(betavals)
            Minput[dtarget, dsource] = copy.deepcopy(deltavals)

            fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                                Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
                                                ncomponents_to_fit)
            # Soutput = Meigv @ Mintrinsic  # signalling over each connection
            ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)

            R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
            R2avg = np.mean(R2list)
            R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

            results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

            ssqchange = ssqd - ssqd_original

            if verbose:
                print('SAPM  {} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} error {:.2f} error2 {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(
                        nperson,iter, alpha, ssqd, error, error2, ssqchange, 100. * ssqd / ssqd_starting, R2avg, R2total))
            ssqd_old = copy.deepcopy(ssqd)
            # now repeat it ...

        # fit the results now to determine output signaling from each region
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)
        fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
                                            Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
                                            ncomponents_to_fit)

        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'fit':fit, 'loadings_fit':loadings_fit, 'W':W, 'loadings':loadings, 'components':components,
                 'R2total':R2total, 'R2avg':R2avg, 'Mintrinsic':Mintrinsic, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
                 'Meigv':Meigv, 'betavals':betavals, 'deltavals':deltavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
                 'fintrinsic_base':fintrinsic_base, 'Sinput_original':Sinput_original, 'DBname':DBname, 'DBnum':DBnum}

        SAPMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()

    np.save(SAPMresultsname, SAPMresults)
    if verbose:
        print('finished SAPM at {}'.format(time.ctime()))
        print('     started at {}'.format(starttime))
        print('     results written to {}'.format(SAPMresultsname))
    return SAPMresultsname


def get_data_components(Sinput):
    # get principal components of Sinput--------------------------
    nr = np.shape(Sinput)[0]
    pca = sklearn.decomposition.PCA()
    pca.fit(Sinput)
    components = pca.components_
    loadings = pca.transform(Sinput)
    mu2 = np.mean(Sinput, axis=0)
    loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
    components = np.concatenate((mu2[np.newaxis, :], components), axis=0)
    return components, loadings


#-------------------------------initialize betavals--------------------------------
def betaval_init_shotgun(Lweight, csource, ctarget, Sinput, Minput, Mconn, components, loadings, fintrinsic_count,
                                vintrinsic_count, deltavals, beta_int1, fintrinsic1, ncomponents_to_fit, nreps = 100000):
    search_record = []
    betascale = 0.2
    for rr in range(nreps):
        # initialize beta values at random values-----------------------------------
        betavals = betascale * np.random.randn(len(csource))  # initialize beta values at zero
        Mconn[ctarget, csource] = betavals

        fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings,
                                                        Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1,
                                                        fintrinsic1, ncomponents_to_fit)

        ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight,
                                                                 betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        R2 = 1 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)
        search_record.append({'betavals': betavals, 'ssqd': ssqd, 'R2': R2})

    ssqd_list = np.array([search_record[x]['ssqd'] for x in range(nreps)])
    R2_list = np.array([search_record[x]['R2'] for x in range(nreps)])
    b_list = np.array([search_record[x]['betavals'] for x in range(nreps)])

    x = ssqd_list.argmin()
    best_betavals = b_list[x, :]
    print('best betavals gives R2 = {:.2f}'.format(R2_list[x]))

    return best_betavals


# gradient descent method to find best clusters------------------------------------
def SAPM_cluster_stepsearch(outputdir, SAPMresultsname, SAPMparametersname, networkfile, regiondataname,
                        clusterdataname, samplesplit, samplestart=0, initial_clusters=[], timepoint='all', epoch='all', betascale=0.1, Lweight = 1.0):

    overall_start_time_text = time.ctime()
    overall_start_time = time.time()

    if not os.path.exists(outputdir): os.mkdir(outputdir)

    # load some data, setup some parameters...
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
    cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
    not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
    ncluster_list = ncluster_list[not_latent]
    full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
    namelist = [cluster_name[x] for x in not_latent]
    namelist += ['Rtotal']
    namelist += ['R ' + cluster_name[x] for x in not_latent]

    nregions = len(ncluster_list)

    prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                  fullgroup=False, normalizevar=True, filter_tcdata = False)


    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
    # need to get principal components for each region to model the clusters as a continuum

    nclusters_total, tsize_total = np.shape(tcdata)

    maxiter = 50
    subsample = [samplesplit,samplestart]  # [2,0] use every 2nd data set, starting with samplestart

    full_rnum_base = np.array([np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]).astype(int)
    initial_clusters = np.array(initial_clusters)
    if (initial_clusters < 0).any():
        fixed_clusters = np.where(initial_clusters >= 0)[0]
    else:
        fixed_clusters = []

    if (len(initial_clusters) != nregions):
        temp_clusters = -1*np.ones(nregions)
        temp_clusters[:len(initial_clusters)] = copy.deepcopy(initial_clusters)   # pad list with -1
        initial_clusters = copy.deepcopy(temp_clusters)

    cluster_numbers = np.zeros(nregions)
    for nn in range(nregions):
        if initial_clusters[nn] < 0:
            cnum = np.random.choice(range(ncluster_list[nn]))
            cluster_numbers[nn] = copy.deepcopy(cnum)
        else:
            cluster_numbers[nn] = copy.deepcopy(initial_clusters[nn])
    cluster_numbers = np.array(cluster_numbers).astype(int)

    print('starting clusters: {}'.format(cluster_numbers))

    lastgood_clusters = copy.deepcopy(cluster_numbers)

    # gradient descent to find best cluster combination
    iter = 0
    costrecord = []
    print('starting step descent search of clusters at {}'.format(time.ctime()))
    converging = True

    if betascale == 0:
        nitermax = 50
        initial_nitermax_stage1 = 1
        initial_nsteps_stage1 = 10
    else:
        nitermax = 50
        initial_nitermax_stage1 = 50
        initial_nsteps_stage1 = 20

    output = sem_physio_model1_V3(cluster_numbers+full_rnum_base, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                                  fixed_beta_vals=[], betascale=betascale, Lweight = Lweight, nitermax=nitermax, verbose=False,normalizevar=False,
                                  initial_nitermax_stage1=initial_nitermax_stage1, initial_nsteps_stage1=initial_nsteps_stage1)

    # now, correct the results for normalizing the variance
    output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose = False)

    SAPMresults = np.load(output,allow_pickle=True)

    R2list = np.array([SAPMresults[x]['R2avg'] for x in range(len(SAPMresults))])
    basecost = np.sum(1 - R2list)
    lastcost = copy.deepcopy(basecost)
    results_record = []
    results_record_summary = []
    while converging and (iter < maxiter):
        iter += 1
        nbetterclusters = 0
        random_region_order = list(range(nregions))
        np.random.shuffle(random_region_order)
        for nnn in random_region_order:
            cost_values = np.zeros(ncluster_list[nnn])
            print('testing region {}'.format(nnn))
            if nnn in fixed_clusters:
                print('cluster for region {} is fixed at {}'.format(nnn,cluster_numbers[nnn]))
            else:
                for ccc in range(ncluster_list[nnn]):
                    test_clusters = copy.deepcopy(cluster_numbers)
                    if test_clusters[nnn] == ccc:   # no change in cluster number from last run
                        cost_values[ccc] = lastcost
                        print('  using cluster {}  total of (1-R2 avg) for the group is {:.3f} - current cluster'.format(ccc,cost_values[ccc]))
                    else:
                        test_clusters[nnn] = ccc

                        output = sem_physio_model1_V3(test_clusters+full_rnum_base, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                                                        fixed_beta_vals=[], betascale=betascale, Lweight = Lweight, nitermax=nitermax, verbose=False, normalizevar=False,
                                                        initial_nitermax_stage1 = initial_nitermax_stage1, initial_nsteps_stage1 = initial_nsteps_stage1)
                        output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose=False)
                        SAPMresults = np.load(output, allow_pickle=True)

                        # print('size of SAPMresults is {}'.format(np.shape(SAPMresults)))
                        R2list = np.array([SAPMresults[x]['R2avg'] for x in range(len(SAPMresults))])
                        R2list2 = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
                        cost_values[ccc] = np.sum(1 - R2list)

                        entry = {'R2list':R2list, 'R2list2':R2list2, 'region':nnn, 'cluster':ccc}
                        results_record.append(entry)
                        print('  using cluster {}  total of (1-R2 avg) for the group is {:.3f}'.format(ccc,cost_values[ccc]))

                x = np.argmin(cost_values)
                this_cost = cost_values[x]
                delta_cost = this_cost-lastcost
                if this_cost < lastcost:
                    cluster_numbers[nnn] = x
                    nbetterclusters += 1
                    lastcost = copy.deepcopy(this_cost)
                else:
                    print('no improvement in clusters found ... region {}'.format(nnn))

                print('iter {} region {} new cost = {:.3f}  previous cost = {:.3f} starting cost {:.3f}  delta cost = {:.3e} {}'.format(
                    iter, nnn, this_cost, lastcost, basecost, delta_cost, time.ctime()))

        if nbetterclusters == 0:
            converging = False
            print('no improvement in clusters found in any region ...')

        # peek at results
        print('\nbest cluster set so far is : {}'.format(cluster_numbers))
        print('average R2 across data sets = {:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list)))
        print('total R2 across data sets = {:.3f} {} {:.3f}'.format(np.mean(R2list2),chr(177),np.std(R2list2)))
        print('average R2 range {:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list)))

        R2avg_text = '{:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list))
        R2total_text = '{:.3f} {} {:.3f}'.format(np.mean(R2list2),chr(177),np.std(R2list2))
        R2range_text = '{:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list))
        results_record_summary.append({'best clusters':cluster_numbers, 'R2avg':R2avg_text, 'R2total':R2total_text, 'R2range':R2range_text})

    outputname = os.path.join(outputdir, 'step_descent_record.npy')
    np.save(outputname, results_record)
    print('results record written to {}'.format(outputname))

    outputname = os.path.join(outputdir, 'step_descent_record_summary.npy')
    np.save(outputname, results_record_summary)
    print('results record summary written to {}'.format(outputname))

    overall_end_time_text = time.ctime()
    overall_end_time = time.time()
    dtime = overall_end_time-overall_start_time
    dtimem = np.floor(dtime/60).astype(int)
    dtimes = np.round(dtime % 60).astype(int)
    print('Cluster search started at {}\n             and ended at {}\n     {} minutes {} sec total'.format(overall_start_time_text, overall_end_time_text, dtimem,dtimes))
    return cluster_numbers


def sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose = True):

    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    # load the data values
    fintrinsic_count = copy.deepcopy(SAPMparams['fintrinsic_count'])
    vintrinsic_count = copy.deepcopy(SAPMparams['vintrinsic_count'])
    tsize = copy.deepcopy(SAPMparams['tsize'])
    tplist_full = copy.deepcopy(SAPMparams['tplist_full'])
    tcdata_centered = copy.deepcopy(SAPMparams['tcdata_centered'])
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    dtarget = SAPMparams['dtarget']
    dsource = SAPMparams['dsource']
    Mconn = copy.deepcopy(SAPMparams['Mconn'])
    Minput = copy.deepcopy(SAPMparams['Minput'])
    tcdata_std = copy.deepcopy(SAPMparams['tcdata_std'])
    std_scale = copy.deepcopy(SAPMparams['std_scale'])

    ntime, NP = np.shape(tplist_full)
    Nintrinsics = vintrinsic_count + fintrinsic_count

    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
    for nperson in range(NP):
        Sinput = copy.deepcopy(SAPMresults_load[nperson]['Sinput'])
        Sinput_original = copy.deepcopy(SAPMresults_load[nperson]['Sinput_original'])
        Sconn = copy.deepcopy(SAPMresults_load[nperson]['Sconn'])
        Minput = copy.deepcopy(SAPMresults_load[nperson]['Minput'])
        Mconn = copy.deepcopy(SAPMresults_load[nperson]['Mconn'])
        beta_int1 = copy.deepcopy(SAPMresults_load[nperson]['beta_int1'])
        R2total = copy.deepcopy(SAPMresults_load[nperson]['R2total'])
        Meigv = copy.deepcopy(SAPMresults_load[nperson]['Meigv'])
        Mintrinsic = copy.deepcopy(SAPMresults_load[nperson]['Mintrinsic'])
        betavals = copy.deepcopy(SAPMresults_load[nperson]['betavals'])
        deltavals = copy.deepcopy(SAPMresults_load[nperson]['deltavals'])
        loadings = copy.deepcopy(SAPMresults_load[nperson]['loadings'])
        components = copy.deepcopy(SAPMresults_load[nperson]['components'])
        loadings_fit = copy.deepcopy(SAPMresults_load[nperson]['loadings_fit'])
        clusterlist = copy.deepcopy(SAPMresults_load[nperson]['clusterlist'])
        fintrinsic1 = copy.deepcopy(SAPMresults_load[nperson]['fintrinsic1'])

        nr, nr_nL = np.shape(Sinput)
        for aa in range(nr):
            loadings[aa,:] /= std_scale[clusterlist[aa],nperson]
            loadings_fit[aa,:] /= std_scale[clusterlist[aa],nperson]

        # correct deltavals and betavals
        deltavals = np.zeros(len(csource))
        for aa in range(len(csource)):
            ss = csource[aa]
            tt = ctarget[aa]
            Minput[tt, ss] /= std_scale[clusterlist[tt], nperson]
            deltavals[aa] = copy.deepcopy(Minput[tt,ss])

        # fit the result ...
        nr, ncomponents_to_fit = np.shape(loadings_fit)

        fit_original = Minput @ Sconn
        fit_original1 = Minput @ Meigv @ Mintrinsic

        R2list = 1.0 - np.sum((Sinput_original - fit_original) ** 2, axis=1) / np.sum(Sinput_original ** 2, axis=1)
        R2avg = np.mean(R2list)
        R2total = 1.0 - np.sum((Sinput_original - fit_original) ** 2) / np.sum(Sinput_original ** 2)

        if verbose:
            print('SAPM {} variance correction:  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson, R2avg, R2total))

        # put back the results
        SAPMresults_load[nperson]['Minput'] = copy.deepcopy(Minput)
        SAPMresults_load[nperson]['R2total'] = copy.deepcopy(R2total)
        SAPMresults_load[nperson]['R2avg'] = copy.deepcopy(R2avg)
        SAPMresults_load[nperson]['deltavals'] = copy.deepcopy(deltavals)
        SAPMresults_load[nperson]['loadings'] = copy.deepcopy(loadings)
        SAPMresults_load[nperson]['loadings_fit'] = copy.deepcopy(loadings_fit)

    p,fe = os.path.split(SAPMresultsname)
    f,e = os.path.splitext(fe)
    outputname = os.path.join(p,f+'_corr'+e)
    np.save(outputname, SAPMresults_load)
    return outputname

# main program
def SAPMrun_V2(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkfile, timepoint,
            epoch, betascale = 0.01, Lweight = 1.0, reload_existing = False, multiple_output = False):

    # load some data, setup some parameters...
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
    cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
    not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
    ncluster_list = ncluster_list[not_latent]
    full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
    namelist = [cluster_name[x] for x in not_latent]
    namelist += ['Rtotal']
    namelist += ['R ' + cluster_name[x] for x in not_latent]

    # starting values
    cnums_original = copy.deepcopy(cnums)
    excelsheetname = 'clusters'

    # run the analysis with SAPM
    clusterlist = np.array(cnums) + full_rnum_base
    if multiple_output:
        prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
    else:
        print('networkfile = {}'.format(networkfile))
        print('regiondataname = {}'.format(regiondataname))
        print('clusterdataname = {}'.format(clusterdataname))
        print('SAPMparametersname = {}'.format(SAPMparametersname))
        print('timepoint = {}'.format(timepoint))
        print('epoch = {}'.format(epoch))
        prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                  fullgroup=False, normalizevar=True, filter_tcdata = False)

    output = sem_physio_model1_V3(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                               fixed_beta_vals = [], betascale = betascale, Lweight = Lweight, normalizevar=False)

    # now, correct the results for normalizing the variance
    output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose = True)

    SAPMresults = np.load(output, allow_pickle=True)
    NP = len(SAPMresults)
    R2list =np.zeros(len(SAPMresults))
    R2list2 =np.zeros(len(SAPMresults))
    for nperson in range(NP):
        R2list[nperson] = SAPMresults[nperson]['R2avg']
        R2list2[nperson] = SAPMresults[nperson]['R2total']
    print('SAPM parameters computed for {} data sets'.format(NP))
    print('R2 values averaged {:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list)))
    print('average R2 values ranged from {:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list)))
    print('Total R2 values were {:.3f} {} {:.3f}'.format(np.mean(R2list2),chr(177),np.std(R2list2)))
    print('Total R2 values ranged from {:.3f} to {:.3f}'.format(np.min(R2list2),np.max(R2list2)))

#----------------------------------------------------------------------------------------
#
#    FUNCTIONS FOR DISPLAYING RESULTS IN VARIOUS FORMATS
#
#----------------------------------------------------------------------------------------

def plot_region_inputs_average(window, target, nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                               Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrange = [], TargetCanvas = 'none',
                               multiple_output = False):

    if isinstance(TargetCanvas,str):
        display_in_GUI = False
    else:
        display_in_GUI = True

    Zthresh = stats.norm.ppf(1-np.array([1.0, 0.05,0.01,0.001]))
    symbollist = [' ','*', chr(8868),chr(8903)]

    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
        print('yrange set to {} to {}'.format(ymin,ymax))
    else:
        setylim = False

    rtarget = rnamelist.index(target)
    m = Minput[rtarget, :]
    sources = np.where(m != 0)[0]
    rsources = [beta_list[ss]['pair'][0] for ss in sources]
    nsources = len(sources)
    nregions = len(rnamelist)
    checkdims = np.shape(Sinput_avg)
    if np.ndim(Sinput_avg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    # get beta values from Mconn
    if multiple_output:
        m = Mconn_avg[:,sources[0]]
        targets2ndlevel_list = np.where(m != 0.)[0]
        textlist = []
        for ss in sources:
            text = betanamelist[ss] + ': '
            beta = Mconn_avg[targets2ndlevel_list,ss]
            for ss2 in range(len(beta)):
                valtext = '{:.2f} '.format(beta[ss2])
                text1 = '{}{}'.format(valtext,betanamelist[targets2ndlevel_list[ss2]])
                text += text1 + ', '
            textlist += [text[:-1]]
    else:
        textlist = []
        for ss in sources:
            beta = Mconn_avg[rtarget, ss]
            valtext = '{:.3f} '.format(beta)
            if ss >= nregions:
                text = 'int{} {}'.format(ss - nregions, valtext)
            else:
                text = '{} {}'.format(rnamelist[ss], valtext)
            textlist += [text]

    fig1 = plt.figure(window)   # for plotting in GUI, expect "window" to refer to a figure
    if display_in_GUI:
        print('Displaying output in GUI window ...')
        plt.clf()
        axs = []
        for n1 in range(nsources):
            axrow = []
            for n2 in range(2):
                axrow += [fig1.add_subplot(nsources,2,n1*2+n2+1)]
            axs += [axrow]
        axs = np.array(axs)
    else:
        plt.close(window)
        fig1, axs = plt.subplots(nsources, 2, sharey=True, figsize=(12, 9), dpi=100, num=window)

    x = list(range(tsize))
    xx = x + x[::-1]
    tc1 = Sinput_avg[rtarget,:]
    tc1p = Sinput_sem[rtarget,:]
    tc1f = fit_avg[rtarget,:]
    tc1fp = fit_sem[rtarget,:]

    y1 = list(tc1f+tc1fp)
    y2 = list(tc1f-tc1fp)
    yy = y1 + y2[::-1]
    axs[0,1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
    axs[0,1].plot(x, tc1f, '-xr', linewidth=1, markersize=4)
    axs[0,1].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
    axs[0,1].plot(x, tc1f+tc1fp, color = (1,0,0), linestyle = '-', linewidth = 0.5)
    axs[0,1].set_title('target input {}'.format(rnamelist[rtarget]))
    # ymax = np.max(np.abs(yy))

    if not multiple_output:
        tc1 = Sconn_avg[rtarget,:]
        tc1p = Sconn_sem[rtarget,:]

        y1 = list(tc1+tc1p)
        y2 = list(tc1-tc1p)
        yy = y1 + y2[::-1]
        axs[1,1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
        axs[1,1].fill(xx,yy, facecolor=(0,0,1), edgecolor='None', alpha = 0.2)
        axs[1,1].set_title('target output {}'.format(rnamelist[rtarget]))

    for ss in range(nsources):
        tc1 = Sconn_avg[sources[ss], :]
        tc1p = Sconn_sem[sources[ss], :]
        y1 = list(tc1 + tc1p)
        y2 = list(tc1 - tc1p)
        yy = y1 + y2[::-1]
        axs[ss,0].plot(x, tc1, '-xr')
        axs[ss,0].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
        axs[ss,0].plot(x, tc1+tc1p, color = (1,0,0), linestyle = '-', linewidth = 0.5)
        if multiple_output:
            if rsources[ss] >= nregions:
                axs[ss, 0].set_title('source output {} {}'.format(betanamelist[sources[ss]], 'int'))
            else:
                axs[ss,0].set_title('source output {} {}'.format(betanamelist[sources[ss]], rnamelist[rsources[ss]]))
            axs[ss,0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                    horizontalalignment='left', verticalalignment='bottom', fontsize=10)
        else:
            if sources[ss] >= nregions:
                axs[ss, 0].set_title('source latent {}'.format(sources[ss]-nregions))
            else:
                axs[ss,0].set_title('source output {}'.format(rnamelist[sources[ss]]))
            axs[ss,0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                    horizontalalignment='left', verticalalignment='bottom', fontsize=10)

        if setylim:
            axs[ss,0].set_ylim((ymin,ymax))

    if display_in_GUI:
        svgname = 'output figure displayed in GUI ... not saved'
        TargetCanvas.draw()
    else:
        svgname = os.path.join(outputdir, 'Avg_' + nametag1 + '.svg')
        plt.savefig(svgname)

    return svgname


def plot_region_inputs_regression(window, target, nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg, beta_list,
                                  rnamelist, betanamelist, Mconn_avg, outputdir, yrange = [], TargetCanvas = 'none',
                                  multiple_output = False):

    if isinstance(TargetCanvas,str):
        display_in_GUI = False
    else:
        display_in_GUI = True

    Zthresh = stats.norm.ppf(1-np.array([1.0, 0.05,0.01,0.001]))
    symbollist = [' ','*', chr(8868),chr(8903)]

    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
    else:
        setylim = False

    rtarget = rnamelist.index(target)
    m = Minput[rtarget, :]
    sources = np.where(m != 0)[0]
    rsources = [beta_list[ss]['pair'][0] for ss in sources]
    nsources = len(sources)
    nregions = len(rnamelist)
    checkdims = np.shape(Sinput_reg)
    if np.ndim(Sinput_reg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    if multiple_output:
        m = Mconn_avg[:, sources[0]]
        targets2ndlevel_list = np.where(m != 0.)[0]
        textlist = []
        for ss in sources:
            text = betanamelist[ss] + ': '
            beta = Mconn_avg[targets2ndlevel_list, ss]
            for ss2 in range(len(beta)):
                valtext = '{:.2f} '.format(beta[ss2])
                text1 = '{}{}'.format(valtext, betanamelist[targets2ndlevel_list[ss2]])
                text += text1 + ', '
            textlist += [text[:-1]]
    else:
        textlist = []
        for ss in sources:
            if ss >= nregions:
                text = 'int{}'.format(ss - nregions)
            else:
                text = rnamelist[ss]
            textlist += [text]

    fig1 = plt.figure(window)
    if display_in_GUI:
        print('Displaying output in GUI window ...')

        plt.clf()
        axs = []
        for n1 in range(nsources):
            axrow = []
            for n2 in range(2):
                axrow += [fig1.add_subplot(nsources,2,n1*2+n2+1)]
            axs += [axrow]
        axs = np.array(axs)
    else:
        plt.close(window)
        fig1, axs = plt.subplots(nsources, 2, sharey=True, figsize=(12, 9), dpi=100, num=window)

    x = list(range(tsize))
    xx = x + x[::-1]
    tc1 = Sinput_reg[rtarget,:,0]
    tc1p = Sinput_reg[rtarget,:,1]
    tc1f = fit_reg[rtarget,:,0]
    tc1fp = fit_reg[rtarget,:,1]

    Z1 = Sinput_reg[rtarget,:,3]
    Z1f = fit_reg[rtarget,:,3]

    S = np.zeros(len(Z1)).astype(int)
    for n in range(len(Z1)): c = np.where(Zthresh < Z1[n])[0];  S[n] = np.max(c)
    Sf = np.zeros(len(Z1f)).astype(int)
    for n in range(len(Z1f)): c = np.where(Zthresh < Z1f[n])[0];  Sf[n] = np.max(c)

    y1 = list(tc1f+tc1fp)
    y2 = list(tc1f-tc1fp)
    yy = y1 + y2[::-1]
    axs[1,1].plot(x, tc1, '-ob', linewidth=1, markersize=4)

    axs[1,1].plot(x, tc1f, '-xr', linewidth=1, markersize=4)
    axs[1,1].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
    axs[1,1].plot(x, tc1f+tc1fp, color = (1,0,0), linestyle = '-', linewidth = 0.5)
    # axs[1,1].plot(x, tc1f-tc1fp, '--r')
    axs[1,1].set_title('target input {}'.format(rnamelist[rtarget]))

    # add marks for significant slope wrt pain
    ympos = np.max(np.abs(yy))
    for n,s in enumerate(S):
        if s > 0: axs[1,1].annotate(symbollist[s], xy = (x[n]-0.25, ympos), fontsize=8)

    for ss in range(nsources):
        tc1 = Sconn_reg[sources[ss], :, 0]
        tc1p = Sconn_reg[sources[ss], :, 1]

        Z1 = Sconn_reg[sources[ss], :, 3]
        S = np.zeros(len(Z1)).astype(int)
        for n in range(len(Z1)): c = np.where(Zthresh < Z1[n])[0];  S[n] = np.max(c)

        y1 = list(tc1 + tc1p)
        y2 = list(tc1 - tc1p)
        yy = y1 + y2[::-1]
        axs[ss,0].plot(x, tc1, '-xr')
        axs[ss,0].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
        axs[ss,0].plot(x, tc1+tc1p, color = (1,0,0), linestyle = '-', linewidth = 0.5)

        if multiple_output:
            if rsources[ss] >= nregions:
                axs[ss, 0].set_title('source output {} {}'.format(betanamelist[sources[ss]], 'int'))
            else:
                axs[ss,0].set_title('source output {} {}'.format(betanamelist[sources[ss]], rnamelist[rsources[ss]]))
            axs[ss,0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                    horizontalalignment='left', verticalalignment='bottom', fontsize=10)
        else:
            if sources[ss] >= nregions:
                axs[ss, 0].set_title('source latent {}'.format(sources[ss] - nregions))
            else:
                axs[ss, 0].set_title('source output {}'.format(rnamelist[sources[ss]]))
            axs[ss, 0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                                horizontalalignment='left', verticalalignment='bottom', fontsize=10)

        # add marks for significant slope wrt pain
        ympos = np.max(np.abs(yy))
        for n, s in enumerate(S):
            if s > 0: axs[ss,0].annotate(symbollist[s], xy = (x[n]-0.25, ympos), fontsize=8)

        if setylim:
            axs[ss,0].set_ylim((ymin,ymax))

    if display_in_GUI:
        svgname = 'output figure written to GUI ... not saved'
        TargetCanvas.draw()
    else:
        svgname = os.path.join(outputdir, 'Reg_' + nametag1 + '.svg')
        plt.savefig(svgname)

    return svgname


def plot_region_fits(window, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange = [], TargetCanvas = 'none'):  # display_in_GUI = False

    if isinstance(TargetCanvas,str):
        display_in_GUI = False
    else:
        display_in_GUI = True

    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
    else:
        setylim = False

    ndisplay = len(regionlist)

    fig2 = plt.figure(window)  # for plotting in GUI, expect "window" to refer to a figure
    if display_in_GUI:  # fix this up to have only integer window numbers as the input (not figures)
        print('Displaying output in GUI window ...')

        plt.clf()
        axs = []
        for nn in range(ndisplay):
            print('plot_region_fits:  creating axes in figure window ...')
            axs += [fig2.add_subplot(ndisplay,1,nn+1)]
        axs = np.array(axs)
    else:
        plt.close(window)
        if ndisplay > 1:
            fig2, axs = plt.subplots(ndisplay, sharey=False, figsize=(12, 6), dpi=100, num=window)
        else:
            fig2, axtemp = plt.subplots(ndisplay, sharey=False, figsize=(12, 6), dpi=100, num=window)
            axs = [axtemp]

    Rtext_record = []
    Rval_record = []
    for nn in range(ndisplay):
        print('plot_region_fits: plotting values ...')
        tc1 = Sinput_avg[regionlist[nn], :]
        if len(fit_avg) > 0:
            tcf1 = fit_avg[regionlist[nn], :]
        t = np.array(range(len(tc1)))

        if len(Sinput_sem) > 0:
            tc1_sem = Sinput_sem[regionlist[nn], :]
            axs[nn].errorbar(t, tc1, tc1_sem, marker = 'o', markerfacecolor = 'b', markeredgecolor = 'b', linestyle = '-', color = 'b', linewidth=1, markersize=4)
        else:
            axs[nn].plot(t, tc1, '-ob', linewidth=1, markersize=4)

        if len(fit_avg) > 0:
            if len(fit_sem) > 0:
                tcf1_sem = fit_sem[regionlist[nn], :]
                axs[nn].errorbar(t, tcf1, tcf1_sem, marker = 'o', markerfacecolor = 'r', markeredgecolor = 'r', linestyle = '-', color = 'r', linewidth=1, markersize=4)
            else:
                axs[nn].plot(t, tcf1, '-xr', linewidth=1, markersize=4)

        axs[nn].set_title('target {}'.format(rnamelist[regionlist[nn]]))
        if setylim:
            axs[nn].set_ylim((ymin,ymax))

        if len(fit_avg) > 0:
            ssq = np.sum((tc1-np.mean(tc1))**2)
            dtc = tc1-tcf1
            ssqd = np.sum((dtc-np.mean(dtc))**2)
            R2fit = 1-ssqd/ssq

            R = np.corrcoef(tc1,tcf1)
            Rtext = 'target {}  R2fit = {:.2f}'.format(rnamelist[regionlist[nn]], R2fit)
            print(Rtext)
            Rval = R[0,1]
            Rtext_record.append(Rtext)
            Rval_record.append([R2fit])
        else:
            Rtext_record.append('no fit values provided')
            Rval_record.append([1.0])

    if display_in_GUI:
        svgname = 'output figure written to GUI ... not saved'
        print(svgname)
        TargetCanvas.draw()
    else:
        svgname = os.path.join(outputdir, 'Avg_' + nametag + '.svg')
        plt.savefig(svgname)

    return svgname, Rtext_record, Rval_record


def write_Mconn_values2(Mconn, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format = 'f', pthresh = 0.05, statsref = '',
                        sigflag = [], multiple_output = False):
    # get beta values from Mconn
    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mconn)

    if np.shape(statsref) != (nr1,nr2):
        statsref = np.zeros((nr1,nr2))

    if np.size(sigflag) == 0:
        sigflag = np.zeros(np.shape(Mconn))

    Tvals = (Mconn-statsref) / (Mconn_sem + 1.0e-20)
    Tthresh = stats.t.ppf(1 - pthresh, NP - 1)
    if np.isnan(Tthresh):  Tthresh = 0.0

    if multiple_output:
        labeltext_record = []
        valuetext_record = []
        Ttext_record = []
        T_record = []
        reftext_record = []
        for n1 in range(nr1):
            tname = betanamelist[n1]
            tpair = beta_list[n1]['pair']
            if tpair[0] >= nregions:
                ts = 'int{}'.format(tpair[0]-nregions)
            else:
                ts = rnamelist[tpair[0]]
                if len(ts) > 4:  ts = ts[:4]
            tt = rnamelist[tpair[1]]
            if len(tt) > 4:  tt = tt[:4]
            # text1 = '{}-{} input from '.format(ts,tt)
            showval = False
            for n2 in range(nr2):
                if (np.abs(Tvals[n1,n2]) > Tthresh)  or (sigflag[n1,n2]):
                    showval = True
                    sname = betanamelist[n2]
                    spair = beta_list[n2]['pair']
                    if spair[0] >= nregions:
                        ss = 'int{}'.format(spair[0]-nregions)
                    else:
                        ss = rnamelist[spair[0]]
                        if len(ss) > 4:  ss = ss[:4]
                    st = rnamelist[spair[1]]
                    if len(st) > 4:  st = st[:4]

                    labeltext = '{}-{}-{}'.format(ss, st, tt)
                    T = Tvals[n1,n2]
                    if format == 'f':
                        valuetext = '{:.3f} {} {:.3f} '.format(Mconn[n1, n2], chr(177), Mconn_sem[n1, n2])
                        Ttext = 'T = {:.2f} '.format(Tvals[n1,n2])
                        reftext = '{:.3f}'.format(statsref[n1,n2])
                    else:
                        valuetext = '{:.3e} {} {:.3e} '.format(Mconn[n1, n2], chr(177), Mconn_sem[n1, n2])
                        Ttext = 'T = {:.2e} '.format(Tvals[n1,n2])
                        reftext = '{:.3e}'.format(statsref[n1,n2])

                    labeltext_record += [labeltext]
                    valuetext_record += [valuetext]
                    Ttext_record += [Ttext]
                    T_record += [T]
                    if showval:
                        print(labeltext)
                        print(valuetext)
                        print(Ttext)
                        print(reftext)
    else:
        labeltext_record = []
        valuetext_record = []
        Ttext_record = []
        T_record = []
        reftext_record = []
        for n1 in range(len(beta_list)):
            tpair = beta_list[n1]['pair']
            if tpair[0] >= nregions:
                ts = 'int{}'.format(tpair[0]-nregions)
            else:
                ts = rnamelist[tpair[0]]
                if len(ts) > 4:  ts = ts[:4]
            tt = rnamelist[tpair[1]]
            if len(tt) > 4:  tt = tt[:4]
            showval = False

            if (np.abs(Tvals[tpair[1],tpair[0]]) > Tthresh)  or (sigflag[tpair[1],tpair[0]]):
                showval = True
                labeltext = '{}-{}'.format(ts, tt)
                T = Tvals[tpair[1],tpair[0]]
                if format == 'f':
                    valuetext = '{:.3f} {} {:.3f} '.format(Mconn[tpair[1],tpair[0]], chr(177), Mconn_sem[tpair[1],tpair[0]])
                    Ttext = 'T = {:.2f} '.format(Tvals[tpair[1],tpair[0]])
                    reftext = '{:.3f}'.format(statsref[tpair[1],tpair[0]])
                else:
                    valuetext = '{:.3e} {} {:.3e} '.format(Mconn[tpair[1],tpair[0]], chr(177), Mconn_sem[tpair[1],tpair[0]])
                    Ttext = 'T = {:.2e} '.format(Tvals[tpair[1],tpair[0]])
                    reftext = '{:.3e}'.format(statsref[tpair[1],tpair[0]])

                labeltext_record += [labeltext]
                valuetext_record += [valuetext]
                Ttext_record += [Ttext]
                reftext_record += [reftext]
                T_record += [T]
                if showval:
                    print(labeltext)
                    print(valuetext)
                    print(Ttext)
                    print(reftext)
    return labeltext_record, valuetext_record, Ttext_record, T_record, Tthresh, reftext_record



def write_Mreg_values(Mint, Mslope, R2, betanamelist, rnamelist, beta_list, format = 'f', R2thresh = 0.1,
                      sigflag = [], multiple_output = False):

    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mslope)

    if np.size(sigflag) == 0:
        sigflag = np.zeros(np.shape(Mslope))

    labeltext_record = []
    inttext_record = []
    slopetext_record = []
    R2text_record = []
    R2_record = []
    for n1 in range(nr1):
        if multiple_output:
            tname = betanamelist[n1]
            tpair = beta_list[n1]['pair']
            if tpair[0] >= nregions:
                ts = 'int{}'.format(tpair[0]-nregions)
            else:
                ts = rnamelist[tpair[0]]
                if len(ts) > 4:  ts = ts[:4]
            tt = rnamelist[tpair[1]]
            if len(tt) > 4:  tt = tt[:4]
            # text1 = '{}-{} input from '.format(ts,tt)
            showval = False
            for n2 in range(nr2):
                if (np.abs(R2[n1,n2]) > R2thresh) or sigflag[n1,n2]:
                    showval = True
                    sname = betanamelist[n2]
                    spair = beta_list[n2]['pair']
                    if spair[0] >= nregions:
                        ss = 'int{}'.format(spair[0]-nregions)
                    else:
                        ss = rnamelist[spair[0]]
                        if len(ss) > 4:  ss = ss[:4]
                    st = rnamelist[spair[1]]
                    if len(st) > 4:  st = st[:4]
                    labeltext = '{}-{}-{}'.format(ss, st, tt)

                    if format == 'f':
                        inttext = '{:.3f}'.format(Mint[n1, n2])
                        slopetext = '{:.3f}'.format(Mslope[n1, n2])
                        R2text = 'R2 = {:.2f}'.format(R2[n1,n2])
                    else:
                        inttext = '{:.3e}'.format(Mint[n1, n2])
                        slopetext = '{:.3e}'.format(Mslope[n1, n2])
                        R2text = 'R2 = {:.2e}'.format(R2[n1,n2])

                    labeltext_record += [labeltext]
                    inttext_record += [inttext]
                    slopetext_record += [slopetext]
                    R2text_record += [R2text]
                    R2_record += [R2[n1,n2]]
                    if showval:
                        print(labeltext)
                        print(inttext)
                        print(slopetext)
                        print(R2text)
        else:
            if n1 >= nregions:
                tt = 'int{}'.format(n1 - nregions)
            else:
                tt = rnamelist[n1]
                if len(tt) > 4:  tt = tt[:4]
            showval = False
            for n2 in range(nr2):
                if (np.abs(R2[n1, n2]) > R2thresh) or sigflag[n1, n2]:
                    showval = True
                    if n2 >= nregions:
                        ss = 'int{}'.format(n2 - nregions)
                    else:
                        ss = rnamelist[n2]
                        if len(ss) > 4:  ss = ss[:4]
                    labeltext = '{}-{}'.format(ss, tt)

                    if format == 'f':
                        inttext = '{:.3f}'.format(Mint[n1, n2])
                        slopetext = '{:.3f}'.format(Mslope[n1, n2])
                        R2text = 'R2 = {:.2f}'.format(R2[n1, n2])
                    else:
                        inttext = '{:.3e}'.format(Mint[n1, n2])
                        slopetext = '{:.3e}'.format(Mslope[n1, n2])
                        R2text = 'R2 = {:.2e}'.format(R2[n1, n2])

                    labeltext_record += [labeltext]
                    inttext_record += [inttext]
                    slopetext_record += [slopetext]
                    R2text_record += [R2text]
                    R2_record += [R2[n1, n2]]
                    if showval:
                        print(labeltext)
                        print(inttext)
                        print(slopetext)
                        print(R2text)

    return labeltext_record, inttext_record, slopetext_record, R2text_record, R2_record, R2thresh


def display_anatomical_cluster(clusterdataname, targetnum, targetcluster, orientation = 'axial', regioncolor = [0,1,1], templatename = 'ccbs', write_output = False):
    # get the voxel coordinates for the target region
    clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = clusterdata['cluster_properties']
    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
    nclusterstotal = np.sum(nclusterlist)

    if type(targetnum) == int:
        r = targetnum
    else:
        # assume "targetnum" input is a region name
        r = rnamelist.index(targetnum)

    IDX = clusterdata['cluster_properties'][r]['IDX']
    if len(targetcluster) > 1:
        cx = []
        cy = []
        cz = []
        for tt in targetcluster:
            idxx = np.where(IDX == tt)
            cx += [clusterdata['cluster_properties'][r]['cx'][idxx]]
            cy += [clusterdata['cluster_properties'][r]['cy'][idxx]]
            cz += [clusterdata['cluster_properties'][r]['cz'][idxx]]
    else:
        idxx = np.where(IDX == targetcluster)
        cx = clusterdata['cluster_properties'][r]['cx'][idxx]
        cy = clusterdata['cluster_properties'][r]['cy'][idxx]
        cz = clusterdata['cluster_properties'][r]['cz'][idxx]

    # load template
    if templatename.lower() == 'brain':
        resolution = 2
    else:
        resolution = 1
    template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(
        templatename, resolution)

    outputimg = pydisplay.pydisplayvoxelregionslice(templatename, template_img, cx, cy, cz, orientation, displayslice = [], colorlist = regioncolor)

    imgname = 'undefined'
    if write_output:
        p,f = os.path.split(clusterdataname)
        imgname = os.path.join(p,'cluster_{}_{}_{}.png'.format(targetnum,targetcluster,orientation[:3]))
        matplotlib.image.imsave(imgname, outputimg)

    return outputimg, imgname


# make labels for each betavalue
def betavalue_labels(csource, ctarget, rnamelist, betanamelist, beta_list, Mconn):

    labeltext_record = []
    nregions = len(rnamelist)
    nbeta = len(csource)
    sources_per_target = np.zeros(nbeta)
    latent_flag = np.zeros(nbeta)
    for nn in range(nbeta):
        n1 = ctarget[nn]
        n2 = csource[nn]

        target_row = Mconn[n1,:]
        check = np.where(target_row > 0,1,0)
        nsources_for_target = np.sum(check)  # for this connection, how many sources contribute, in total?
        sources_per_target[nn] = nsources_for_target

        tname = betanamelist[n1]
        tpair = beta_list[n1]['pair']
        if tpair[0] >= nregions:
            ts = 'int{}'.format(tpair[0]-nregions)
            latent_flag[nn] = 1
        else:
            ts = rnamelist[tpair[0]]
            if len(ts) > 4:  ts = ts[:4]
        tt = rnamelist[tpair[1]]
        if len(tt) > 4:  tt = tt[:4]

        sname = betanamelist[n2]
        spair = beta_list[n2]['pair']
        if spair[0] >= nregions:
            ss = 'int{}'.format(spair[0] - nregions)
            latent_flag[nn] = 1
        else:
            ss = rnamelist[spair[0]]
            if len(ss) > 4:  ss = ss[:4]
        st = rnamelist[spair[1]]
        if len(st) > 4:  st = st[:4]
        labeltext = '{}-{}-{}'.format(ss, st, tt)

        labeltext_record += [labeltext]

    return labeltext_record, sources_per_target, latent_flag



def regress_signal_features_with_cov(target, covariates, Minput, Sinput_total, fit_total, Sconn_total, beta_list, rnamelist, pthresh, outputdir, descriptor):
    print('size of Minput is {}'.format(np.shape(Minput)))
    print('size of Sinput_total is {}'.format(np.shape(Sinput_total)))
    print('size of Sconn_total is {}'.format(np.shape(Sconn_total)))

    # regress signal magnitude, or variance, or something, with covariates, instead of looking only at
    # correlations with B values ...
    p = covariates[np.newaxis, :]
    p -= np.mean(p)
    G = np.concatenate((np.ones(np.shape(p)),p), axis=0) # put the intercept term first

    Sinput_pp = np.max(Sinput_total,axis=1) - np.min(Sinput_total,axis=1)
    Sinput_var = np.var(Sinput_total,axis=1)
    Sinput_std = np.std(Sinput_total,axis=1)

    Sconn_pp = np.max(Sconn_total,axis=1) - np.min(Sconn_total,axis=1)
    Sconn_var = np.var(Sconn_total,axis=1)
    Sconn_std = np.std(Sconn_total,axis=1)

    # Zthresh = stats.norm.ppf(1 - pthresh)
    # pval = 1 - stats.norm.cdf(Z)

    print('size of Sconn_var is {}'.format(np.shape(Sconn_std)))
    nc1, np1 = np.shape(Sconn_std)
    Sconn_feature_reg = np.zeros((nc1,5))

    print('Sconn_feature_reg:')
    nregions = len(rnamelist)
    cname_list = []
    for aa in range(nc1):
        m = Sconn_std[aa, :]
        if np.var(m) > 0:
            b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
            Z = np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(np1-3)
            pval = 1 - stats.norm.cdf(Z)
            Sconn_feature_reg[aa, :] = [b[0, 0], b[0, 1], R2, Z,pval]
            if beta_list[aa]['pair'][0] >= nregions:
                sname = 'int{}'.format(beta_list[aa]['pair'][0] - nregions)
            else:
                sname = rnamelist[beta_list[aa]['pair'][0]][:4]
            tname = rnamelist[beta_list[aa]['pair'][1]][:4]
            cname = '{}-{}'.format(sname,tname )
            cname_list += [cname]
            print('{} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {}'.format(aa, Sconn_feature_reg[aa,0], Sconn_feature_reg[aa,1], Sconn_feature_reg[aa,2], Sconn_feature_reg[aa,3], Sconn_feature_reg[aa,4], cname))

    # write results to excel file
    # sort output by magnitude of Z
    Zthresh = stats.norm.ppf(1-pthresh)
    pthresh_list = ['{:.3e}'.format(pthresh)] * nc1
    Zthresh_list = ['{:.3f}'.format(Zthresh)] * nc1
    si = np.argsort(np.abs(Sconn_feature_reg[:,3]))[::-1]
    int_text = np.array(['{:.2e}'.format(Sconn_feature_reg[x,0]) for x in si])
    slope_text = np.array(['{:.2e}'.format(Sconn_feature_reg[x,1]) for x in si])
    R2_text = np.array(['R2 = {:.2e}'.format(Sconn_feature_reg[x, 2]) for x in si])
    Z_text = np.array(['Z = {:.2f}'.format(Sconn_feature_reg[x, 3]) for x in si])
    p_text = np.array(['p = {:.2e}'.format(Sconn_feature_reg[x, 4]) for x in si])

    # Sinput_feature_reg_sorted = Sinput_feature_reg[si,:]
    textoutputs = {'regions': np.array(cname_list)[si], 'int': int_text, 'slope': slope_text,
                   'R2': R2_text, 'Z': Z_text, 'p': p_text,
                   'Z thresh': np.array(Zthresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
    df1 = pd.DataFrame(textoutputs)

    print('size of Sinput_var is {}'.format(np.shape(Sinput_std)))
    print('Sinput_feature_reg:')
    nc1, np1 = np.shape(Sinput_std)
    Sinput_feature_reg = np.zeros((nc1,5))
    cname_list = []
    for aa in range(nc1):
        m = Sinput_std[aa, :]
        if np.var(m) > 0:
            b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
            Z = np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(np1-3)
            pval = 1 - stats.norm.cdf(Z)
            Sinput_feature_reg[aa, :] = [b[0, 0], b[0, 1], R2, Z,pval]
            cname = '{}'.format(rnamelist[aa][:4])
            cname_list += [cname]
            print('{} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {}'.format(aa, Sinput_feature_reg[aa,0], Sinput_feature_reg[aa,1], Sinput_feature_reg[aa,2], Sinput_feature_reg[aa,3], Sinput_feature_reg[aa,4], cname))


    # write results to excel file
    # sort output by magnitude of Z
    Zthresh = stats.norm.ppf(1-pthresh)
    pthresh_list = ['{:.3e}'.format(pthresh)] * nc1
    Zthresh_list = ['{:.3f}'.format(Zthresh)] * nc1
    si = np.argsort(np.abs(Sinput_feature_reg[:,3]))[::-1]
    int_text = np.array(['{:.2e}'.format(Sinput_feature_reg[x, 0]) for x in si])
    slope_text = np.array(['{:.2e}'.format(Sinput_feature_reg[x, 1]) for x in si])
    R2_text = np.array(['R2 = {:.2e}'.format(Sinput_feature_reg[x, 2]) for x in si])
    Z_text = np.array(['Z = {:.2f}'.format(Sinput_feature_reg[x, 3]) for x in si])
    p_text = np.array(['p = {:.2e}'.format(Sinput_feature_reg[x, 4]) for x in si])

    # Sinput_feature_reg_sorted = Sinput_feature_reg[si,:]
    textoutputs = {'regions': np.array(cname_list)[si], 'int': int_text, 'slope': slope_text,
                   'R2': R2_text, 'Z': Z_text, 'p': p_text,
                   'Z thresh': np.array(Zthresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}

    df2 = pd.DataFrame(textoutputs)
    xlname = os.path.join(outputdir, descriptor + '.xlsx')
    with pd.ExcelWriter(xlname) as writer:
        df1.to_excel(writer, sheet_name='Sconn')
        df2.to_excel(writer, sheet_name = 'Sinput')

    outputname = xlname



def display_SAPM_results(window, outputnametag, covariates, outputtype, outputdir, SAPMparametersname, SAPMresultsname,
                         group, target = '', pthresh = 0.05, SAPMstatsfile = '', setylimits = [], TargetCanvas = [],
                         display_in_GUI = False, multiple_output = False,
                         SRresultsname2 = '', SRparametersname2 = '', covariates2 = []):

    # options of results to display:
    # 1) average input time-courses compared with model input
    # 2) modelled input signaling with corresponding source time-courses (outputs from source regions)
    # 3) t-test comparisons between groups, or w.r.t. zero (outputs to excel files)
    # 4) regression with continuous covariate
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram', 'Paired_diff']

    # load SAPM parameters
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    network = SAPMparams['network']
    beta_list = SAPMparams['beta_list']
    betanamelist = SAPMparams['betanamelist']
    nruns_per_person = SAPMparams['nruns_per_person']
    rnamelist = SAPMparams['rnamelist']
    fintrinsic_count = SAPMparams['fintrinsic_count']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    vintrinsic_count = SAPMparams['vintrinsic_count']
    nclusterlist = SAPMparams['nclusterlist']
    tplist_full = SAPMparams['tplist_full']
    tcdata_centered = SAPMparams['tcdata_centered']
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    tsize = SAPMparams['tsize']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    Nintrinsic = fintrinsic_count + vintrinsic_count
    # end of reloading parameters-------------------------------------------------------

    # load the SEM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
    if os.path.isfile(SRresultsname2):
        two_group_comparison = True
        SAPMresults_load2 = np.load(SRresultsname2, allow_pickle=True)
        SAPMparams2 = np.load(SRparametersname2, allow_pickle=True).flat[0]
    else:
        two_group_comparison = False

    NP = len(SAPMresults_load)
    if len(covariates) == NP:
        covariates_entered = True
    else:
        covariates_entered = False

    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic
    if fintrinsic_count > 0:
        paradigm_centered = SAPMresults_load[0]['fintrinsic_base']  # model paradigm used for fixed pattern latent inputs

    if epoch >= tsize:
        et1 = 0
        et2 = tsize
    else:
        if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
            et1 = (timepoint - np.floor(epoch / 2)).astype(int)
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)
        else:
            et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)
    if et1 < 0: et1 = 0
    if et2 > tsize: et2 = tsize
    epoch = et2 - et1
    if fintrinsic_count > 0:
        ftemp = paradigm_centered[0,et1:et2]

    DBref_mean = np.zeros((nbeta, nbeta))
    DBref_std = np.zeros((nbeta, nbeta))
    print('SAPMstatsfile = {}'.format(SAPMstatsfile))
    if os.path.isfile(SAPMstatsfile):
        xls = pd.ExcelFile(SAPMstatsfile, engine='openpyxl')
        df1 = pd.read_excel(xls, 'B stats')
        stats_conname = df1.loc[:, 'name']
        stats_mean = df1.loc[:, 'mean']
        stats_std = df1.loc[:, 'std']

        for nn in range(len(stats_conname)):
            nregions = len(rnamelist)
            cname = stats_conname[nn]
            c = cname.index('-')
            sname = cname[:c]
            tname = cname[(c + 1):]
            tnum = rnamelist.index(tname)
            if 'latent' in sname:
                lnum = int(sname[6:])
                snum = nregions + lnum
            else:
                snum = rnamelist.index(sname)
            DBref_mean[tnum, snum] = stats_mean[nn]
            DBref_std[tnum, snum] = stats_std[nn]

    DBrecord = np.zeros((nbeta, nbeta, NP))
    Brecord = np.zeros((nbeta, nbeta, NP))
    Drecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        Sinput_original = SAPMresults_load[nperson]['Sinput_original']
        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        Mintrinsic = SAPMresults_load[nperson]['Mintrinsic']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']

        nruns = nruns_per_person[nperson]
        if fintrinsic_count > 0:
            fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

        fit = Minput @ Sconn

        nr, tsize_total = np.shape(Sinput_original)
        tsize = (tsize_total / nruns).astype(int)
        nbeta,tsize2 = np.shape(Sconn)

        if nperson == 0:
            Sinput_total = np.zeros((nr,tsize, NP))
            Sconn_total = np.zeros((nbeta,tsize, NP))
            fit_total = np.zeros((nr,tsize, NP))
            Mintrinsic_total = np.zeros((Nintrinsic,tsize, NP))

        tc = Sinput_original
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        Sinput_total[:,:,nperson] = tc1

        tc = Sconn
        tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
        Sconn_total[:,:,nperson] = tc1

        tc = fit
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        fit_total[:,:,nperson] = tc1

        tc = Mintrinsic
        tc1 = np.mean(np.reshape(tc, (Nintrinsic, nruns, tsize)), axis=1)
        Mintrinsic_total[:,:,nperson] = tc1

        DBrecord[:, :, nperson] = Mconn
        Drecord[:ncon, :, nperson] = Minput
        Brecord[:ncon, :, nperson] = Mconn[:ncon,:]/(Minput + 1.0e-3)
        # Brecord[ktarget,ksource,nperson] = Mconn[ktarget,ksource]
        R2totalrecord[nperson] = R2total

    Brecord[np.abs(Brecord) > 1e2] = 0.0
    #----------------------------------------------------------------------------------
    # compare groups with T-tests------------------------------------------------------
    # or compare group average results to zero---------------------------------------
    # set the group
    # the input 'group' is a list of array indices for which data to use
    g = list(range(NP))
    if (len(group) == NP) or (len(group) == 0):    # all values were selected for the group
        g1 = g
        g2 = []
    else:
        g1 = group
        g2 = [x for x in g if x not in g1]

    print('g1:  {} values'.format(len(g1)))
    print('g2:  {} values'.format(len(g2)))

    #-------------------------------------------------------------------------------
    #-------------prep for regression with continuous covariate------------------------------
    if covariates_entered:
        p = covariates[np.newaxis, g1]
        if len(np.unique(p)) > len(g1)/3:  # assume the values are continuous
            continuouscov = True
            p -= np.mean(p)
            G = np.concatenate((np.ones((1, len(g1))),p), axis=0) # put the intercept term first
        else:
            continuouscov = False
    else:
        continuouscov = False

    #-------------------------------------------------------------------------------------
    # significance of average Mconn values -----------------------------------------------
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram', 'Paired_diff']
    DB_avg = np.mean(DBrecord[:, :, g1], axis=2)
    DB_sem = np.std(DBrecord[:, :, g1], axis=2) / np.sqrt(len(g1))

    D_avg = np.mean(Drecord[:, :, g1], axis=2)
    D_sem = np.std(Drecord[:, :, g1], axis=2) / np.sqrt(len(g1))

    B_avg = np.mean(Brecord[:, :, g1], axis=2)
    B_sem = np.std(Brecord[:, :, g1], axis=2) / np.sqrt(len(g1))

    if two_group_comparison:
        nruns_per_person2 = SAPMparams2['nruns_per_person']
        NP2 = len(SAPMresults_load2)
        DBrecord2 = np.zeros((nbeta, nbeta, NP2))
        Brecord2 = np.zeros((nbeta, nbeta, NP2))
        Drecord2 = np.zeros((nbeta, nbeta, NP2))
        R2totalrecord2 = np.zeros(NP2)
        for nperson in range(NP2):
            Sinput2 = SAPMresults_load2[nperson]['Sinput']
            Sinput_original2 = SAPMresults_load2[nperson]['Sinput_original']
            Sconn2 = SAPMresults_load2[nperson]['Sconn']
            Minput2 = SAPMresults_load2[nperson]['Minput']
            Mconn2 = SAPMresults_load2[nperson]['Mconn']
            Mintrinsic2 = SAPMresults_load2[nperson]['Mintrinsic']
            beta_int12 = SAPMresults_load2[nperson]['beta_int1']
            R2total2 = SAPMresults_load2[nperson]['R2total']
            Meigv2 = SAPMresults_load2[nperson]['Meigv']
            betavals2 = SAPMresults_load2[nperson]['betavals']

            nruns = nruns_per_person2[nperson]
            if fintrinsic_count > 0:
                fintrinsic1 = np.array(list(ftemp) * nruns_per_person2[nperson])

            fit2 = Minput2 @ Sconn2

            nr, tsize_total = np.shape(Sinput_original2)
            tsize = (tsize_total / nruns).astype(int)
            nbeta, tsize2 = np.shape(Sconn2)

            if nperson == 0:
                Sinput_total2 = np.zeros((nr, tsize, NP2))
                Sconn_total2 = np.zeros((nbeta, tsize, NP2))
                fit_total2 = np.zeros((nr, tsize, NP2))
                Mintrinsic_total2 = np.zeros((Nintrinsic, tsize, NP2))

            tc = Sinput_original2
            tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
            Sinput_total2[:, :, nperson] = tc1

            tc = Sconn2
            tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
            Sconn_total2[:, :, nperson] = tc1

            tc = fit2
            tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
            fit_total2[:, :, nperson] = tc1

            tc = Mintrinsic2
            tc1 = np.mean(np.reshape(tc, (Nintrinsic, nruns, tsize)), axis=1)
            Mintrinsic_total2[:, :, nperson] = tc1

            DBrecord2[:, :, nperson] = Mconn2
            Drecord2[:ncon, :, nperson] = Minput2
            Brecord2[:ncon, :, nperson] = Mconn2[:ncon, :] / (Minput2 + 1.0e-3)
            R2totalrecord2[nperson] = R2total2

        Brecord2[np.abs(Brecord2) > 1e2] = 0.0

        # -------------------------------------------------------------------------------
        # -------------prep for regression with continuous covariate------------------------------
        p2 = covariates2[np.newaxis, :]
        if continuouscov & covariates_entered:  # use the mode determined for the first set of results
            p2 -= np.mean(p2)
            G2 = np.concatenate((np.ones((1, NP2)), p), axis=0)  # put the intercept term first

        # -------------------------------------------------------------------------------------
        # significance of average Mconn values -----------------------------------------------
        # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram', 'Paired_diff']
        DB_avg2 = np.mean(DBrecord2, axis=2)
        DB_sem2 = np.std(DBrecord2, axis=2) / np.sqrt(NP2)

        D_avg2 = np.mean(Drecord2, axis=2)
        D_sem2 = np.std(Drecord2, axis=2) / np.sqrt(NP2)

        B_avg2 = np.mean(Brecord2, axis=2)
        B_sem2 = np.std(Brecord2, axis=2) / np.sqrt(NP2)


    if outputtype == 'B_Significance':

        # significant B values-------------------------------------
        descriptor = outputnametag + '_Bsig'
        print('\n\nAverage B values')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(B_avg, B_sem, NP, betanamelist, rnamelist,
                                                beta_list, format='f', pthresh=pthresh, statsref = DBref_mean, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)]*len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)]*len(Ttext)

        Rtextlist = [' ']*10
        Rvallist = [0]*10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where(T < 1e3)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'B': np.array(valuetext)[si[si2]], 'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]], 'stat ref': np.array(reftext)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Bsig')

        # significant D values-------------------------------------
        descriptor = outputnametag + '_Dsig'
        print('\n\nAverage D values')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(D_avg, D_sem, NP, betanamelist, rnamelist,
                                                beta_list, format='f', pthresh=pthresh, statsref = '', multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where(T < 1e3)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'D': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]], 'stat ref': np.array(reftext)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Dsig')


        # significant DB values-------------------------------------
        descriptor = outputnametag + '_DBsig'
        print('\n\nAverage DB values')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(DB_avg, DB_sem, NP, betanamelist, rnamelist,
                                                beta_list, format='f', pthresh=pthresh, statsref = DBref_mean, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where(T < 1e3)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'DB': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]], 'stat ref': np.array(reftext)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='DBsig')

        return xlname

    #-------------------------------------------------------------------------------
    #-------------B-value regression with continuous covariate------------------------------
    # regression of Mrecord with continuous covariate
    # glm_fit
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']
    if outputtype == 'B_Regression':

        # regression of B values with covariate-----------------------------------
        print('\n\ngenerating results for B_Regression...')
        descriptor = outputnametag + '_Breg'
        Mregression = np.zeros((nbeta,nbeta,3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = Brecord[aa,bb,g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis,:], G)
                    Mregression[aa,bb,:] = [b[0,0],b[0,1],R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        # Z = arctanh(R)*sqrt(NP-1)
        Rthresh = np.tanh(Zthresh/np.sqrt(NP-1))
        R2thresh = Rthresh**2

        print('B regression with continuous covariate values')
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:,:,0], Mregression[:,:,1],
                                            Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f',
                                            R2thresh=R2thresh, multiple_output = multiple_output)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si],'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            with pd.ExcelWriter(xlname) as writer:
                df.to_excel(writer, sheet_name='Breg')

        else:
            xlname = 'NA'
            print('Regression of B values with covariate ... no significant values found at p < {}'.format(pthresh))
        outputname = xlname

        print('finished generating results for B_Regression...')

        # regression of D values with covariate-----------------------------------
        print('\n\ngenerating results for D_Regression...')
        descriptor = outputnametag + '_Breg'
        Mregression = np.zeros((nbeta, nbeta, 3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = Drecord[aa, bb, g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
                    Mregression[aa, bb, :] = [b[0, 0], b[0, 1], R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        # Z = arctanh(R)*sqrt(NP-1)
        Rthresh = np.tanh(Zthresh / np.sqrt(NP - 1))
        R2thresh = Rthresh ** 2

        print('D regression with continuous covariate values')
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:, :, 0],
                                            Mregression[:, :, 1], Mregression[:, :, 2], betanamelist, rnamelist,
                                            beta_list, format='f', R2thresh=R2thresh, multiple_output=multiple_output)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si], 'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            with pd.ExcelWriter(xlname) as writer:
                df.to_excel(writer, sheet_name='Dreg')

        else:
            xlname = 'NA'
            print('Regression of D values with covariate ... no significant values found at p < {}'.format(pthresh))
        outputname = xlname


        # regression of DB values with covariate-----------------------------------
        print('\n\ngenerating results for DB_Regression...')
        descriptor = outputnametag + '_DBreg'
        Mregression = np.zeros((nbeta, nbeta, 3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = DBrecord[aa, bb, g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
                    Mregression[aa, bb, :] = [b[0, 0], b[0, 1], R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        # Z = arctanh(R)*sqrt(NP-1)
        Rthresh = np.tanh(Zthresh / np.sqrt(NP - 1))
        R2thresh = Rthresh ** 2

        print('DB regression with continuous covariate values')
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:, :, 0],
                                            Mregression[:, :, 1], Mregression[:, :, 2], betanamelist, rnamelist,
                                            beta_list, format='f', R2thresh=R2thresh, multiple_output=multiple_output)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si], 'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            with pd.ExcelWriter(xlname) as writer:
                df.to_excel(writer, sheet_name='DBreg')

        else:
            xlname = 'NA'
            print('Regression of DB values with covariate ... no significant values found at p < {}'.format(pthresh))
        outputname = xlname

        print('finished generating regression results ...\n')

        # testing other regression options
        # target = []
        # descriptor = outputnametag + '_BOLDstdev_vs_Covariate'
        # regress_signal_features_with_cov(target, covariates[g1], Minput, Sinput_total[:, :, g1], fit_total[:, :, g1],
        #                                  Sconn_total[:, :, g1], beta_list, rnamelist, pthresh, outputdir, descriptor)
        return outputname

    #-------------------------------------------------------------------------------
    # get the group averages etc-----------------------------------------------------
    print('calculating group average values...')
    Sinput_avg = np.mean(Sinput_total[:, :, g1], axis=2)
    Sinput_sem = np.std(Sinput_total[:, :, g1], axis=2) / np.sqrt(len(g1))
    Sconn_avg = np.mean(Sconn_total[:, :, g1], axis=2)
    Sconn_sem = np.std(Sconn_total[:, :, g1], axis=2) / np.sqrt(len(g1))
    fit_avg = np.mean(fit_total[:, :, g1], axis=2)
    fit_sem = np.std(fit_total[:, :, g1], axis=2) / np.sqrt(len(g1))
    Mintrinsic_avg = np.mean(Mintrinsic_total[:, :, g1], axis=2)
    Mintrinsic_sem = np.std(Mintrinsic_total[:, :, g1], axis=2) / np.sqrt(len(g1))

    #---------------------------------------------------------------------------
    # plot time-course averages and fits with continuous covariate--------------
    # need to specify which region to display ***
    # include options to display more than one region? ***
    # need to specify where to display it ***

    # show some regions
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']
    if outputtype == 'Plot_BOLDModel':
        print('generating results for Plot_BOLDModel...')
        descriptor = outputnametag + '_BOLDmodel'

        nregions = len(rnamelist)
        rnamelist_full = copy.deepcopy(rnamelist)
        if fintrinsic_count > 0: rnamelist_full += ['latent0']
        for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count + nn)]

        regionnum = [rnamelist_full.index(target) ]   # input a region
        nametag = rnamelist_full[regionnum[0]] + '_' + outputnametag   # create name for saving figure
        print('Plotting Sinput data for region {}, number {}'.format(target, regionnum))

        if len(setylimits) > 0:
            ylim = setylimits[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []
        if regionnum[0] >= nregions:   # latent input
            latentnum = regionnum[0] - nregions
            svgname, Rtext, Rvals = plot_region_fits(window, [latentnum], nametag, Mintrinsic_avg, Mintrinsic_sem, [], [], rnamelist_full[nregions:], outputdir, yrangethis, TargetCanvas) # display_in_GUI
        else:
            svgname, Rtext, Rvals = plot_region_fits(window, regionnum, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrangethis, TargetCanvas) # display_in_GUI
        outputname = svgname

        print('finished generating results for Plot_BOLDModel...')
        return outputname

    #-------------------------------------------------------------------------------
    #-------------time-course regression with continuous covariate------------------------------
    # prepare time-course values to plot
    if continuouscov:
        print('generating regression values ...')
        Sinput_reg = np.zeros((nr,tsize,4))
        fit_reg = np.zeros((nr,tsize,4))
        Sconn_reg = np.zeros((nbeta,tsize,4))
        for tt in range(tsize):
            for nn in range(nr):
                m = Sinput_total[nn,tt,g1]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                Sinput_reg[nn,tt,:2] = b
                Sinput_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g)-3)]

                m = fit_total[nn,tt,g1]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                fit_reg[nn,tt,:2] = b
                fit_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g1)-3)]

            for nn in range(nbeta):
                m = Sconn_total[nn,tt,g1]
                b, fit, R2, total_var, res_var = pysem.general_glm(m, G)
                Sconn_reg[nn,tt,:2] = b
                Sconn_reg[nn,tt,-2:] = [R2, np.sign(R2)*np.arctanh(np.sqrt(np.abs(R2)))*np.sqrt(len(g1)-3)]

            # need to save Sinput_reg, Sinput_R2, etc., somewhere for later use....

        print('finished generating regression values ...')

    #-------------------------------------------------------------------------------
    # plot region input time-courses averages and fits with continuous covariate----
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']
    if (outputtype == 'Plot_SourceModel'):
        print('generating outputs for Plot_SourceModel...')
        descriptor = outputnametag + '_SourceModel'

        nametag1 = target + descriptor

        if len(setylimits) > 0:
            ylim = setylimits[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []

        if continuouscov:
            outputname = plot_region_inputs_regression(window, target, nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg,
                                beta_list, rnamelist, betanamelist, DB_avg, outputdir, yrangethis, TargetCanvas, multiple_output)

        outputname = plot_region_inputs_average(window, target, nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                                   Sconn_sem, beta_list, rnamelist, betanamelist, DB_avg, outputdir, yrangethis, TargetCanvas, multiple_output)

        print('finished generating outputs for Plot_SourceModel...')
        return outputname


    if outputtype == 'Group_Diff':
        # significant B values-------------------------------------
        descriptor = outputnametag + '_groupBdiff'
        print('\n\nAverage difference of B values')

        DBdiff_avg = np.mean(DBrecord, axis=2) - np.mean(DBrecord2, axis=2)
        V1 = np.var(DBrecord,axis=2)
        V2 = np.var(DBrecord2,axis=2)
        Sp = np.sqrt( ((NP-1)*V1 + (NP-2)*V2)/(NP+NP2-2) )
        DBdiff_sem = Sp*np.sqrt( 1/NP + 1/NP2 )

        Ddiff_avg = np.mean(Drecord, axis=2) - np.mean(Drecord2, axis=2)
        V1 = np.var(Drecord,axis=2)
        V2 = np.var(Drecord2,axis=2)
        Sp = np.sqrt( ((NP-1)*V1 + (NP-2)*V2)/(NP+NP2-2) )
        Ddiff_sem = Sp*np.sqrt( 1/NP + 1/NP2 )

        Bdiff_avg = np.mean(Brecord, axis=2) - np.mean(Brecord2, axis=2)
        V1 = np.var(Brecord,axis=2)
        V2 = np.var(Brecord2,axis=2)
        Sp = np.sqrt( ((NP-1)*V1 + (NP-2)*V2)/(NP+NP2-2) )
        Bdiff_sem = Sp*np.sqrt( 1/NP + 1/NP2 )

        print('size of Bdiff_avg is {}'.format(np.shape(Bdiff_avg)))

        DB_avg = np.mean(DBrecord, axis=2)
        DB_sem = np.std(DBrecord, axis=2) / np.sqrt(NP)
        DB_avg2 = np.mean(DBrecord2, axis=2)
        DB_sem2 = np.std(DBrecord2, axis=2) / np.sqrt(NP2)

        D_avg = np.mean(Drecord, axis=2)
        D_sem = np.std(Drecord, axis=2) / np.sqrt(NP)
        D_avg2 = np.mean(Drecord2, axis=2)
        D_sem2 = np.std(Drecord2, axis=2) / np.sqrt(NP2)

        B_avg = np.mean(Brecord, axis=2)
        B_sem = np.std(Brecord, axis=2) / np.sqrt(NP)
        B_avg2 = np.mean(Brecord2, axis=2)
        B_sem2 = np.std(Brecord2, axis=2) / np.sqrt(NP2)

        sigflag = np.ones(np.shape(Bdiff_avg))
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(Bdiff_avg, Bdiff_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(B_avg, B_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(B_avg2, B_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where((T < 1e3) & (np.abs(T) > Tthresh))  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'B': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]],
                       'B group1': np.array(valuetext1)[si[si2]], 'B group2': np.array(valuetext2)[si[si2]], 'stat ref1': np.array(reftext1)[si[si2]], 'stat ref2': np.array(reftext2)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Bdiffsig')

        # significant D values-------------------------------------
        descriptor = outputnametag + '_groupDdiff'
        print('\n\nAverage D values difference')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(Ddiff_avg, Ddiff_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(D_avg, D_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(D_avg2, D_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where((T < 1e3) & (np.abs(T) > Tthresh))   # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'D': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]],
                       'D group1': np.array(valuetext1)[si[si2]], 'D group2': np.array(valuetext2)[si[si2]], 'stat ref1': np.array(reftext1)[si[si2]], 'stat ref2': np.array(reftext2)[si[si2]]}

        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Ddiffsig')

        # significant DB values-------------------------------------
        descriptor = outputnametag + '_groupDBdiff'
        print('\n\nAverage DB values difference')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(DBdiff_avg, DBdiff_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(DB_avg, DB_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(DB_avg2, DB_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where((T < 1e3) & (np.abs(T) > Tthresh))   # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'DB': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]],
                       'DB group1': np.array(valuetext1)[si[si2]], 'DB group2': np.array(valuetext2)[si[si2]], 'stat ref1': np.array(reftext1)[si[si2]], 'stat ref2': np.array(reftext2)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='DBdiffsig')

        return xlname


    if outputtype == 'Paired_Diff':
        # significant B values-------------------------------------
        descriptor = outputnametag + '_pairedBdiff'
        print('\n\nAverage difference of B values')

        DBdiff_avg = np.mean(DBrecord-DBrecord2, axis=2)
        DBdiff_sem = np.std(DBrecord-DBrecord2, axis=2) / np.sqrt(NP)

        Ddiff_avg = np.mean(Drecord-Drecord2, axis=2)
        Ddiff_sem = np.std(Drecord-Drecord2, axis=2) / np.sqrt(NP)

        Bdiff_avg = np.mean(Brecord-Brecord2, axis=2)
        Bdiff_sem = np.std(Brecord-Brecord2, axis=2) / np.sqrt(NP)


        DB_avg = np.mean(DBrecord, axis=2)
        DB_sem = np.std(DBrecord, axis=2) / np.sqrt(NP)
        DB_avg2 = np.mean(DBrecord2, axis=2)
        DB_sem2 = np.std(DBrecord2, axis=2) / np.sqrt(NP2)

        D_avg = np.mean(Drecord, axis=2)
        D_sem = np.std(Drecord, axis=2) / np.sqrt(NP)
        D_avg2 = np.mean(Drecord2, axis=2)
        D_sem2 = np.std(Drecord2, axis=2) / np.sqrt(NP2)

        B_avg = np.mean(Brecord, axis=2)
        B_sem = np.std(Brecord, axis=2) / np.sqrt(NP)
        B_avg2 = np.mean(Brecord2, axis=2)
        B_sem2 = np.std(Brecord2, axis=2) / np.sqrt(NP2)

        sigflag = np.ones(np.shape(Bdiff_avg))
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(Bdiff_avg, Bdiff_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(B_avg, B_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(B_avg2, B_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where((T < 1e3) & (np.abs(T) > Tthresh))   # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'B': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]],
                       'B group1': np.array(valuetext1)[si[si2]], 'B group2': np.array(valuetext2)[si[si2]], 'stat ref1': np.array(reftext1)[si[si2]], 'stat ref2': np.array(reftext2)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Bdiffsig')

        # significant D values-------------------------------------
        descriptor = outputnametag + '_pairedDdiff'
        print('\n\nAverage D values difference')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(Ddiff_avg, Ddiff_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(D_avg, D_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(D_avg2, D_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)
        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where((T < 1e3) & (np.abs(T) > Tthresh))   # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'D': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]],
                       'D group1': np.array(valuetext1)[si[si2]], 'D group2': np.array(valuetext2)[si[si2]], 'stat ref1': np.array(reftext1)[si[si2]], 'stat ref2': np.array(reftext2)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Ddiffsig')

        # significant DB values-------------------------------------
        descriptor = outputnametag + '_pairedDBdiff'
        print('\n\nAverage DB values difference')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(DBdiff_avg, DBdiff_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(DB_avg, DB_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(DB_avg2, DB_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where((T < 1e3) & (np.abs(T) > Tthresh))   # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'DB': np.array(valuetext)[si[si2]],
                       'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]],
                       'DB group1': np.array(valuetext1)[si[si2]], 'DB group2': np.array(valuetext2)[si[si2]], 'stat ref1': np.array(reftext1)[si[si2]], 'stat ref2': np.array(reftext2)[si[si2]]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='DBdiffsig')

        return xlname


    if outputtype == 'Regress_diff_v_diff':
        # significant B values-------------------------------------
        descriptor = outputnametag + '_diff_v_Covdiff'
        print('\n\nRegression of difference in B values with differences in covariate values')

        DBdiff = DBrecord-DBrecord2
        Ddiff = Drecord-Drecord2
        Bdiff = Brecord-Brecord2

        # -------------------------------------------------------------------------------
        # -------------prep for regression with continuous covariate------------------------------
        cov_diff = covariates - covariates2
        p = cov_diff[np.newaxis, :]
        p -= np.mean(p)
        G = np.concatenate((np.ones((1, NP)), p), axis=0)  # put the intercept term first

        # regression of B values with covariate-----------------------------------
        print('\n\ngenerating results for B_Regression...')
        descriptor = outputnametag + '_dB_v_dCreg'
        Mregression = np.zeros((nbeta,nbeta,3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = Bdiff[aa,bb,g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis,:], G)
                    Mregression[aa,bb,:] = [b[0,0],b[0,1],R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        Rthresh = np.tanh(Zthresh / np.sqrt(NP - 1))
        R2thresh = Rthresh ** 2

        print('deltaB regression with delta covariate values')
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:, :, 0],
                                            Mregression[:, :, 1], Mregression[:, :, 2], betanamelist, rnamelist,
                                            beta_list, format='f', R2thresh=R2thresh, multiple_output=multiple_output)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si], 'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            with pd.ExcelWriter(xlname) as writer:
                df.to_excel(writer, sheet_name='dB_v_dCreg')
        else:
            xlname = 'NA'
            print('Regression of B values with covariate ... no significant values found at p < {}'.format(pthresh))


        # regression of D values with covariate-----------------------------------
        print('\n\ngenerating results for D_Regression...')
        descriptor = outputnametag + '_dD_v_dCreg'
        Mregression = np.zeros((nbeta, nbeta, 3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = Ddiff[aa, bb, g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
                    Mregression[aa, bb, :] = [b[0, 0], b[0, 1], R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        Rthresh = np.tanh(Zthresh / np.sqrt(NP - 1))
        R2thresh = Rthresh ** 2

        print('deltaD regression with delta covariate values')
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:, :, 0],
                                Mregression[:, :, 1], Mregression[:, :, 2], betanamelist,  rnamelist,
                                beta_list, format='f', R2thresh=R2thresh,  multiple_output=multiple_output)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si], 'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            with pd.ExcelWriter(xlname) as writer:
                df.to_excel(writer, sheet_name='dD_v_dCreg')
        else:
            xlname = 'NA'
            print('Regression of D values with covariate ... no significant values found at p < {}'.format(pthresh))

        # regression of DB values with covariate-----------------------------------
        print('\n\ngenerating results for DB_Regression...')
        descriptor = outputnametag + '_dDB_v_dCreg'
        Mregression = np.zeros((nbeta, nbeta, 3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = DBdiff[aa, bb, g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], G)
                    Mregression[aa, bb, :] = [b[0, 0], b[0, 1], R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        Rthresh = np.tanh(Zthresh / np.sqrt(NP - 1))
        R2thresh = Rthresh ** 2

        print('deltaDB regression with delta covariate values')
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:, :, 0],
                                    Mregression[:, :, 1], Mregression[:, :, 2], betanamelist, rnamelist,
                                    beta_list, format='f', R2thresh=R2thresh,  multiple_output=multiple_output)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si], 'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            # p, f = os.path.split(SAPMresultsname)
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            with pd.ExcelWriter(xlname) as writer:
                df.to_excel(writer, sheet_name='dDB_v_dCreg')
        else:
            xlname = 'NA'
            print('Regression of DB values with covariate ... no significant values found at p < {}'.format(pthresh))

        return xlname


#-----------------------------------------------------------------------------------
#   Functions for plotting SAPM network results
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

def define_drawing_regions_from_file(regionfilename):
    # setup region labels and positions
    xls = pd.ExcelFile(regionfilename, engine='openpyxl')
    df1 = pd.read_excel(xls, 'regions')
    keylist = df1.keys()
    names = df1['name']
    posx = df1['posx']
    posy = df1['posy']
    offset_x = df1['labeloffset_x']
    offset_y = df1['labeloffset_y']

    if 'outputangle' in keylist:
        outputangle = df1['outputangle']
    else:
        outputangle = []

    regions = []
    for nn in range(len(names)):
        entry = {'name': names[nn], 'pos':[posx[nn],posy[nn]], 'labeloffset':np.array([offset_x[nn],offset_y[nn]]),
                 'outputangle':outputangle[nn]}
        regions.append(entry)

    return regions


def display_anatomical_slices(clusterdataname, regionname, clusternum, templatename):
    orientation = 'axial'
    regioncolor = [1,1,0]

    # get the connection and region information
    clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = clusterdata['cluster_properties']
    template_img = clusterdata['template_img']
    rnamelist = [cluster_properties[x]['rname'] for x in range(len(cluster_properties))]
    targetnum = rnamelist.index(regionname)

    # get the voxel coordinates for the target region
    IDX = clusterdata['cluster_properties'][targetnum]['IDX']
    idxx = np.where(IDX == clusternum)
    cx = clusterdata['cluster_properties'][targetnum]['cx'][idxx]
    cy = clusterdata['cluster_properties'][targetnum]['cy'][idxx]
    cz = clusterdata['cluster_properties'][targetnum]['cz'][idxx]

    #-------------------------------------------------------------------------------------
    # display one slice of an anatomical region in the selected target figure
    outputimg = pydisplay.pydisplayvoxelregionslice(templatename, template_img, cx, cy, cz, orientation, displayslice = [], colorlist = regioncolor)
    return outputimg


def points_on_ellipses1(pos0, pos1, ovalsize):
    # point on ellipse 0 on line from region 0 to region 1
    ovd = np.array(ovalsize)/2.0

    v01 = np.array(pos1)-np.array(pos0)
    d01 = np.sqrt(1/((v01[0]/ovd[0])**2 + (v01[1]/ovd[1])**2))
    pe0 = pos0 + d01*v01

    # point on ellipse 1 on line from region 1 to region 0
    v10 = np.array(pos0)-np.array(pos1)
    d10 = np.sqrt(1/((v10[0]/ovd[0])**2 + (v10[1]/ovd[1])**2))
    pe1a = pos1 + d10*v10

    # angle of line along vector v01, wrt x axis
    angleA = (180/np.pi)*np.arctan2(v01[1],v01[0])
    angleA = np.round(angleA).astype(int)

    return pe0, pe1a



def points_on_ellipses2(pos0, pos1, pos2, ovalsize, offset = 0.007):
    # point on ellipse 0 on line from region 0 to region 1
    ovd = np.array(ovalsize)/2.0

    v01 = np.array(pos1)-np.array(pos0)
    d01 = np.sqrt(1/((v01[0]/ovd[0])**2 + (v01[1]/ovd[1])**2))
    pe0 = pos0 + d01*v01

    # point on ellipse 1 on line from region 1 to region 0
    v10 = np.array(pos0)-np.array(pos1)
    d10 = np.sqrt(1/((v10[0]/ovd[0])**2 + (v10[1]/ovd[1])**2))
    pe1a = pos1 + d10*v10

    v12 = np.array(pos2)-np.array(pos1)
    d12 = np.sqrt(1/((v12[0]/ovd[0])**2 + (v12[1]/ovd[1])**2))
    pe1b = pos1 + d12*v12

    # point on ellipse 1 on line from region 1 to region 0
    v21 = np.array(pos1)-np.array(pos2)
    d21 = np.sqrt(1/((v21[0]/ovd[0])**2 + (v21[1]/ovd[1])**2))
    pe2 = pos2 + d21*v21

    # smooth arc line in region 1, betwen arrows for pos0-->pos1 and pos1-->pos2
    # line starts along vector v01 at point pe1a
    # line ends along vector v12 at point pe1b

    # angle of line along vector v01, wrt x axis
    angleA = (180/np.pi)*np.arctan2(v01[1],v01[0])
    angleA = np.round(angleA).astype(int)

    # angle of line along vector v12, wrt x axis
    angleB = (180/np.pi)*np.arctan2(v12[1],v12[0])
    angleB = np.round(angleB).astype(int)
    anglediff = np.abs(angleB-angleA)

    pe1ab_connectionstyle = "angle3,angleA={},angleB={}".format(angleA,angleB)

    # special case
    specialcase = False
    if np.abs(anglediff-180.0) < 1.0:
        specialcase = True
        pe1ab_connectionstyle = "arc3,rad=0"
        pe1ab_connectionstyle = "bar,fraction=0"

    if np.abs(anglediff) < 1.0:
        specialcase = False
        pe1ab_connectionstyle = "arc3,rad=0"

    # shift lines slightly to allow for reciprocal connections
    dpos1 = np.array([offset*np.sin(angleA*np.pi/180.0), offset*np.cos(angleA*np.pi/180.0)])
    dpos2 = np.array([offset*np.sin(angleB*np.pi/180.0), offset*np.cos(angleB*np.pi/180.0)])

    pe0 += dpos1
    pe1a += dpos1
    pe1b += dpos2
    pe2 += dpos2

    return pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase


def points_on_ellipses_SO(pos0, pos1, pos2, ovalsize, offset = 0.0, outputangle1 = [], outputangle2 = []):
    throughconnection = (len(pos2) > 0)
    ovd = np.array(ovalsize)/2.0

    if type(outputangle1) == list:
        # point on ellipse 0 on line from region 0 center to region 1 center
        v01 = np.array(pos1)-np.array(pos0)
        d01 = np.sqrt(1/((v01[0]/ovd[0])**2 + (v01[1]/ovd[1])**2))
        pe0 = pos0 + d01*v01
    else:
        # point on ellipse 0 at output point
        v01 = np.array([np.cos(outputangle1*np.pi/180.), np.sin(outputangle1*np.pi/180.)])
        d01 = np.sqrt(1/((v01[0]/ovd[0])**2 + (v01[1]/ovd[1])**2))
        pe0 = pos0 + d01*v01

    # point on ellipse 1 on line from region 1 to region 0
    v10 = np.array(pe0)-np.array(pos1)
    d10 = np.sqrt(1/((v10[0]/ovd[0])**2 + (v10[1]/ovd[1])**2))
    pe1a = pos1 + d10*v10

    if throughconnection:
        if type(outputangle2) == list:
            # point on ellipse 1 on line from region 1 center to region 2 center
            v12 = np.array(pos2) - np.array(pos1)
            d12 = np.sqrt(1 / ((v12[0] / ovd[0]) ** 2 + (v12[1] / ovd[1]) ** 2))
            pe1b = pos1 + d12 * v12
        else:
            # point on ellipse 1 at output point
            v12 = np.array([np.cos(outputangle2 * np.pi / 180.), np.sin(outputangle2 * np.pi / 180.)])
            d12 = np.sqrt(1 / ((v12[0] / ovd[0]) ** 2 + (v12[1] / ovd[1]) ** 2))
            pe1b = pos1 + d12 * v12

        # point on ellipse 1 on line from region 2 to region 1
        v21 = np.array(pe1b)-np.array(pos2)
        d21 = np.sqrt(1/((v21[0]/ovd[0])**2 + (v21[1]/ovd[1])**2))
        pe2 = pos2 + d21*v21

    # smooth arc line in region 1, betwen arrows for pos0-->pos1 and pos1-->pos2
    # line starts along vector v01 at point pe1a
    # line ends along vector v12 at point pe1b

    # angle of line along vector v01, wrt x axis
    angleA = (180/np.pi)*np.arctan2(v01[1],v01[0])
    angleA = np.round(angleA).astype(int)

    pe1ab_connectionstyle = "arc3,rad=0"
    specialcase = False
    if throughconnection:
        # angle of line along vector v12, wrt x axis
        angleB = (180/np.pi)*np.arctan2(v12[1],v12[0])
        angleB = np.round(angleB).astype(int)
        anglediff = np.abs(angleB-angleA)

        pe1ab_connectionstyle = "angle3,angleA={},angleB={}".format(angleA,angleB)

        # special case
        specialcase = False
        if np.abs(anglediff-180.0) < 1.0:
            specialcase = True
            pe1ab_connectionstyle = "arc3,rad=0"
            pe1ab_connectionstyle = "bar,fraction=0"

        if np.abs(anglediff) < 1.0:
            specialcase = False
            pe1ab_connectionstyle = "arc3,rad=0"

    # shift lines slightly to allow for reciprocal connections
    dpos1 = np.array([offset*np.sin(angleA*np.pi/180.0), offset*np.cos(angleA*np.pi/180.0)])
    pe0 += dpos1
    pe1a += dpos1
    if throughconnection:
        dpos2 = np.array([offset*np.sin(angleB*np.pi/180.0), offset*np.cos(angleB*np.pi/180.0)])
        pe1b += dpos2
        pe2 += dpos2
    else:
        pe1b = np.array([0,0])
        pe2 = np.array([0,0])

    return pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase


def parse_statval(val):
    if isinstance(val,float):
        m = val
        s = 0
        return m,s

    foundpattern = False
    t = chr(177)   # check for +/- sign
    if t in val:
        x = val.index(t)
        m = float(val[:x])
        s = float(val[(x+1):])
        foundpattern = True

    if '=' in val:
        x = val.index('=')
        m = float(val[(x+1):])
        s = []
        foundpattern = True

    if not foundpattern:
        m = float(val)
        s = 0

    return m,s


def parse_threshold_text(thresholdtext):
    # parse thresholdtext
    if '<' in thresholdtext:
        c = thresholdtext.index('<')
        comparisontext = '<'
    else:
        c = thresholdtext.index('>')
        comparisontext = '>'
    threshold = float(thresholdtext[(c+1):])

    if c > 0:
        if 'mag' in thresholdtext[:c]:
            absval = False
        if 'abs' in thresholdtext[:c]:
            absval = True
    else:
        absval = False

    if absval:
        print('threshold is set to absolute value {} {}'.format(comparisontext, threshold))
    else:
        print('threshold is set to {} {}'.format(comparisontext, threshold))
    return comparisontext, absval, threshold


def parse_connection_name(connection, regionlist):
    h1 = connection.index('-')
    if '-' in connection[(h1+2):]:
        h2 = connection[(h1+2):].index('-') + h1 + 2
        r1 = connection[:h1]
        r2 = connection[(h1+1):h2]
        r3 = connection[(h2+1):]

        i1 = regionlist.index(r1)
        i2 = regionlist.index(r2)
        i3 = regionlist.index(r3)
    else:
        r1 = connection[:h1]
        r2 = connection[(h1+1):]
        r3 = 'none'

        i1 = regionlist.index(r1)
        i2 = regionlist.index(r2)
        i3 = -1

    return (r1,r2,r3),(i1,i2,i3)


def draw_sapm_plot(results_file, sheetname, regionnames, regions, statname, figurenumber, scalefactor, cnums, thresholdtext = 'abs>0', writefigure = False):
    # plot diagram is written to a figure window and saved
    #
    xls = pd.ExcelFile(results_file, engine='openpyxl')
    df1 = pd.read_excel(xls, sheetname)
    connections = df1[regionnames]
    statvals = df1[statname]

    statval_values = []
    for nn in range(len(statvals)):
        val1 = statvals[nn]
        m, s = parse_statval(val1)
        statval_values += [m]
    statval_values = np.array(statval_values)

    # set scale factor if it is set to 'auto'
    if isinstance(scalefactor,str):
        maxval = 5.0
        maxstat = np.max(np.abs(statval_values))
        scalefactor = maxval/maxstat

    # parse thresholdtext
    comparisontext, absval, threshold = parse_threshold_text(thresholdtext)

    plt.close(figurenumber)

    regionlist = [regions[x]['name'] for x in range(len(regions))]
    regionlist_trunc = [regions[x]['name'][:4] for x in range(len(regions))]

    # set some drawing parameters
    ovalsize = (0.1,0.05)
    width = 0.001
    ovalcolor = [0,0,0]

    # start drawing
    plt.close(figurenumber)
    fig = plt.figure(figurenumber)
    ax = fig.add_axes([0,0,1,1])

    # add ellipses and labels
    for nn in range(len(regions)):
        ellipse = mpatches.Ellipse(regions[nn]['pos'],ovalsize[0],ovalsize[1], alpha = 0.3)
        ax.add_patch(ellipse)
        if nn < len(cnums):
            ax.annotate('{}{}'.format(regions[nn]['name'],cnums[nn]),regions[nn]['pos']+regions[nn]['labeloffset'])
        else:
            ax.annotate(regions[nn]['name'],regions[nn]['pos']+regions[nn]['labeloffset'])

    an_list = []
    connection_list = []
    acount = 0
    for nn in range(len(connections)):
        # plot lines for connections
        c1 = connections[nn]
        m = statval_values[nn]
        if comparisontext == '>':
            if absval:
                statcondition = np.abs(m) > threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
            else:
                statcondition = m > threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
                if threshold < 0:
                    linethick = np.max([0.5, linethick])
        else:
            if absval:
                statcondition = np.abs(m) < threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
                linethick = np.max([0.5, linethick])
            else:
                statcondition = m < threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])

        if statcondition:
            if m > 0:
                linecolor = 'k'
            else:
                linecolor = 'r'
            rlist,ilist = parse_connection_name(c1,regionlist_trunc)
            if rlist[2] == 'none':
                throughconnection = False   # this is always the case for single output, leave this for future expansion
            else:
                throughconnection = True

            # get positions of ends of lines,arrows, etc... for one connection
            p0 = regions[ilist[0]]['pos']
            p1 = regions[ilist[1]]['pos']
            if ilist[2] >= 0:
                p2 = regions[ilist[2]]['pos']
            else:
                p2 = [0,0]

            if p0 != p1  and  p1 != p2:
                pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase = points_on_ellipses2(p0,p1,p2,ovalsize)
                print('{}  {}'.format(c1,pe1ab_connectionstyle))

                connection_type1 = {'con':'{}-{}'.format(rlist[0],rlist[1]), 'type':'input'}
                if throughconnection:
                    connection_type2 = {'con':'{}-{}'.format(rlist[1],rlist[2]), 'type':'output'}
                    connection_joiner = {'con':'{}-{}'.format(rlist[1],rlist[1]), 'type':'joiner'}

                if specialcase:
                    print('special case...')
                    an1 = ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type1)
                    if throughconnection:
                        an1 = ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                        acount+= 1
                        an_list.append(an1)
                        connection_list.append(connection_type2)
                else:
                    an1 = ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                    acount+= 1
                    an_list.append(an1)
                    connection_list.append(connection_type1)
                    if throughconnection:
                        an1 = ax.annotate('',xy=pe2,xytext = pe1b, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                        acount+= 1
                        an_list.append(an1)
                        connection_list.append(connection_type2)
                        an1 = ax.annotate('',xy=pe1b,xytext = pe1a, arrowprops=dict(arrowstyle="->", connectionstyle=pe1ab_connectionstyle, linewidth = linethick/2.0, color = linecolor, shrinkA = 0.0, shrinkB = 0.0))
                        acount+= 1
                        an_list.append(an1)
                        connection_list.append(connection_joiner)
            else:
                print('ambiguous connection not drawn:  {}'.format(c1))

    # look for inputs and outputs drawn for the same connection.  If both exist, only show the input connection
    conlist = [connection_list[x]['con'] for x in range(len(connection_list))]
    typelist = [connection_list[x]['type'] for x in range(len(connection_list))]
    for nn in range(len(connection_list)):
        con = conlist[nn]
        c = np.where([conlist[x] == con for x in range(len(conlist))])[0]
        if len(c) > 1:
            t = [typelist[x] for x in c]
            if 'input' in t:   # if some of the connections are inputs, do not draw outputs at the same place
                c2 = np.where([typelist[x] == 'output' for x in c])[0]
                if len(c2) > 0:
                    redundant_c = c[c2]
                    # remove the redundant connections
                    for c3 in redundant_c:
                        a = an_list[c3]
                        a.remove()
                        typelist[c3] = 'removed'
                        connection_list[c3]['type'] = 'removed'

    svgname = 'none'
    if writefigure:
        p,f1 = os.path.split(results_file)
        f,e = os.path.splitext(f1)
        svgname = os.path.join(p,f+'_'+statname+'_SAPMnetwork.svg')
        plt.figure(figurenumber)
        plt.savefig(svgname, format='svg')

    return svgname


# draw SAPM diagram for single output model----------------------------------------------------
def draw_sapm_plot_SO(results_file, sheetname, regionnames, regions, statname, figurenumber, scalefactor, cnums, thresholdtext = 'abs>0', writefigure = False):
    # plot diagram is written to a figure window and saved
    #
    xls = pd.ExcelFile(results_file, engine='openpyxl')
    df1 = pd.read_excel(xls, sheetname)
    connections = df1[regionnames]
    statvals = df1[statname]

    statval_values = []
    for nn in range(len(statvals)):
        val1 = statvals[nn]
        m, s = parse_statval(val1)
        statval_values += [m]
    statval_values = np.array(statval_values)

    # set scale factor if it is set to 'auto'
    if isinstance(scalefactor,str):
        maxval = 5.0
        maxstat = np.max(np.abs(statval_values))
        if maxstat < 1.0e-6:
            scalefactor = 1.0
        else:
            scalefactor = maxval/maxstat

    # parse thresholdtext
    comparisontext, absval, threshold = parse_threshold_text(thresholdtext)

    plt.close(figurenumber)

    regionlist = [regions[x]['name'] for x in range(len(regions))]
    regionlist_trunc = [regions[x]['name'][:4] for x in range(len(regions))]

    # set some drawing parameters
    ovalsize = (0.1,0.05)
    width = 0.001
    ovalcolor = [0,0,0]

    # start drawing
    plt.close(figurenumber)
    fig = plt.figure(figurenumber)
    ax = fig.add_axes([0,0,1,1])

    # add ellipses and labels
    for nn in range(len(regions)):
        ellipse = mpatches.Ellipse(regions[nn]['pos'],ovalsize[0],ovalsize[1], alpha = 0.3)
        ax.add_patch(ellipse)
        if nn < len(cnums):
            ax.annotate('{}{}'.format(regions[nn]['name'],cnums[nn]),regions[nn]['pos']+regions[nn]['labeloffset'])
        else:
            ax.annotate(regions[nn]['name'],regions[nn]['pos']+regions[nn]['labeloffset'])

    an_list = []
    connection_list = []
    acount = 0
    for nn in range(len(connections)):
        # plot lines for connections
        c1 = connections[nn]
        m = statval_values[nn]
        if comparisontext == '>':
            if absval:
                statcondition = np.abs(m) > threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
            else:
                statcondition = m > threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
                if threshold < 0:
                    linethick = np.max([0.5, linethick])
        else:
            if absval:
                statcondition = np.abs(m) < threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])
                linethick = np.max([0.5, linethick])
            else:
                statcondition = m < threshold
                linethick = np.min([5.0, np.abs(m)*scalefactor])

        linethick = np.max([0.1, linethick])

        if statcondition:
            if m > 0:
                linecolor = 'k'
            else:
                linecolor = 'r'
            rlist,ilist = parse_connection_name(c1,regionlist_trunc)

            if rlist[0] != rlist[1]:
                # get positions of ends of lines,arrows, etc... for one connection
                p0 = regions[ilist[0]]['pos']
                p1 = regions[ilist[1]]['pos']
                outputangle1 = regions[ilist[0]]['outputangle']
                outputangle2 = []

                # pe0, pe1a = points_on_ellipses1(p0,p1,ovalsize)
                pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase = points_on_ellipses_SO(p0, p1, [], ovalsize, offset = 0.0, outputangle1 = outputangle1, outputangle2 = outputangle2)

                connection_type1 = {'con':'{}-{}'.format(rlist[0],rlist[1]), 'type':'input', 'rlist':rlist, 'ilist':ilist, 'p0':p0, 'p1':p1, 'linecolor':linecolor, 'outputangle':outputangle1}

                an1 = ax.annotate('',xy=pe1a,xytext = pe0, arrowprops=dict(arrowstyle="->", connectionstyle='arc3', linewidth = linethick, color = linecolor, shrinkA = 0.01, shrinkB = 0.01))
                acount+= 1
                an_list.append(an1)
                connection_list.append(connection_type1)

    # after the connections are drawn, look for through connections that can be added
    for nn in range(len(connection_list)):
        # see if a region is plotted as both an input and an output
        rlist_in = connection_list[nn]['rlist']
        p0 = connection_list[nn]['p0']
        p1 = connection_list[nn]['p1']
        outputangle1 = connection_list[nn]['outputangle']
        linecolor = connection_list[nn]['linecolor']
        for mm in range(len(connection_list)):
            rlist_out = connection_list[mm]['rlist']

            if rlist_in[1] == rlist_out[0]:
                print('need a joiner betweeen {}-{} and {}-{}'.format(rlist_in[0],rlist_in[1],rlist_out[0],rlist_out[1]))

                p1 = connection_list[mm]['p0']
                p2 = connection_list[mm]['p1']
                outputangle2 = connection_list[mm]['outputangle']

                pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase = points_on_ellipses_SO(p0, p1, p2, ovalsize, offset = 0.0, outputangle1 = outputangle1, outputangle2 = outputangle2)
                an1 = ax.annotate('', xy=pe1b, xytext=pe1a,
                                  arrowprops=dict(arrowstyle="->", connectionstyle=pe1ab_connectionstyle,
                                                  linewidth=linethick / 2.0, color=linecolor, shrinkA=0.0, shrinkB=0.0))

    svgname = 'none'
    if writefigure:
        p,f1 = os.path.split(results_file)
        f,e = os.path.splitext(f1)
        svgname = os.path.join(p,f+'_'+statname+'_SAPMnetwork.svg')
        plt.figure(figurenumber)
        plt.savefig(svgname, format='svg')

    return svgname


def generate_null_data_set(regiondataname, covariatesname, npeople=0, variable_variance = False):
    r = np.load(regiondataname, allow_pickle=True).flat[0]
    # dict_keys(['region_properties', 'DBname', 'DBnum'])
    region_properties = r['region_properties']
    DBname = r['DBname']
    DBnum = r['DBnum']

    nregions = len(region_properties)
    # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize', 'rname', 'DBname', 'DBnum', 'prefix'])
    for nn in range(nregions):

        if npeople > 0:  # override the number of runs in region_properties
            tc = copy.deepcopy(region_properties[nn]['tc'])
            nruns_per_person = copy.deepcopy(region_properties[nn]['nruns_per_person'])
            nclusters, tsize_big = np.shape(tc)
            tsize = copy.deepcopy(region_properties[nn]['tsize'])
            avg_runs_per_person = np.round(np.mean(nruns_per_person)).astype(int)
            nruns_per_person = (avg_runs_per_person*np.ones(npeople)).astype(int)
            nruns_total = np.sum(nruns_per_person).astype(int)
            tsize_big = tsize*nruns_total
            new_tc = np.zeros((nclusters,tsize_big))
            new_tc_sem = np.zeros((nclusters,tsize_big))
        else:
            tc = copy.deepcopy(region_properties[nn]['tc'])
            nclusters, tsize_big = np.shape(tc)
            tsize = copy.deepcopy(region_properties[nn]['tsize'])
            nruns_per_person = copy.deepcopy(region_properties[nn]['nruns_per_person'])
            nruns_total = np.sum(nruns_per_person)
            new_tc = np.zeros(np.shape(tc))
            new_tc_sem = copy.deepcopy(region_properties[nn]['tc_sem'])

        for cc in range(nclusters):
            for tt in range(nruns_total):
                t1 = tt*tsize
                t2 = (tt+1)*tsize
                tc_run = np.random.normal(0,1,tsize)
                if variable_variance:
                    std_original = np.std(tc[cc,t1:t2])
                    tc_run *= std_original/np.std(tc_run)  # vary the standard deviation randomly across runs/clusters
                tc_run -= np.mean(tc_run)
                new_tc[cc,t1:t2] = copy.deepcopy(tc_run)

        region_properties[nn]['tc'] = copy.deepcopy(new_tc)
        region_properties[nn]['tc_sem'] = copy.deepcopy(new_tc_sem)
        region_properties[nn]['nruns_per_person'] = copy.deepcopy(nruns_per_person)

    r['region_properties'] = copy.deepcopy(region_properties)
    p,f = os.path.split(regiondataname)
    outputname = os.path.join(p,'nulldata_' + f)
    np.save(outputname, r)
    print('wrote null data to {}'.format(outputname))

    # covariates
    if len(covariatesname) > 0:
        p,f = os.path.split(covariatesname)
        cc = np.load(covariatesname, allow_pickle=True).flat[0]
        if npeople > 0:
            ncov,numcovpeople = np.shape(cc['GRPcharacteristicsvalues'])
            cc['GRPcharacteristicsvalues'] = np.random.normal(0,1,(ncov,npeople))
        outputcovname = os.path.join(p, 'nulldata_' + f)
        np.save(outputcovname, cc)
        print('wrote null covariates data to {}'.format(outputcovname))
    else:
        outputcovname = covariatesname

    return outputname, outputcovname



def generate_simulated_data_set(regiondataname, covariatesname, networkfile, clusterdataname, npeople=0, variable_variance = False, timepoint = 'all', epoch = 'all'):
    r = np.load(regiondataname, allow_pickle=True).flat[0]
    region_properties = r['region_properties']
    DBname = r['DBname']
    DBnum = r['DBnum']

    p,f = os.path.split(regiondataname)
    paramsname = os.path.join(p,'simparams.npy')
    prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, paramsname, timepoint, epoch,
                                     fullgroup=False, normalizevar=True, filter_tcdata=False)

    params = np.load(paramsname, allow_pickle=True).flat[0]
    Minput = copy.deepcopy(params['Minput'])
    Mconn = copy.deepcopy(params['Mconn'])
    ctarget = copy.deepcopy(params['ctarget'])
    csource = copy.deepcopy(params['csource'])
    dtarget = copy.deepcopy(params['dtarget'])
    dsource = copy.deepcopy(params['dsource'])
    nregions = copy.deepcopy(params['nregions'])
    rnamelist = copy.deepcopy(params['rnamelist'])
    fintrinsic_count = copy.deepcopy(params['fintrinsic_count'])
    vintrinsic_count = copy.deepcopy(params['vintrinsic_count'])
    Nintrinsic = fintrinsic_count + vintrinsic_count

    tsize = copy.deepcopy(params['tsize'])
    fintrinsic_base = copy.deepcopy(params['fintrinsic_base'])
    deltavals = np.ones(len(dtarget))

    bsample = [0.62, 0.46, 0.81, 0.41, 0.36, -0.45, -0.25, -0.1, -0.1, -0.32, 0.15, -0.01, 0.76, -0.25, 0.18, -0.1,
                -0.1, -0.1, 0.07, 0.40, 0.08, 0.35, 0.02, 0.71, 0.51, 0.44, 0.16, 0.27, -0.1, -0.36, 0.15, 0.1, 0.13,
               -0.02, -0.05, -0.1, 0.16] * (np.ceil(len(ctarget)/37).astype(int) + 1)

    bsample2 = [0.58, 0.47, 0.02, 0.12, 0.01, -0.07, -0.12, 0.09, -0.04, 0.01, 0.13, -0.33, 1.45, -0.51, 0.09,
                0.42, 0.1, 0.04, -0.06, 0.46, 0.19, 0.04, 0.45, 0.25, 0.69, 0.13, -0.24, 0.08, 0.13, -0.79, -0.22,
                -0.02, -0.24, -0.1, 0.1, -0.2, -0.05] * (np.ceil(len(ctarget)/37).astype(int) + 1)


    betavals = np.array(bsample[:len(ctarget)])

    sim_reference = []
    for nn in range(len(ctarget)):
        if csource[nn] >= nregions:
            lnum = csource[nn] - nregions
            sname = 'latent{}'.format(lnum)
            # betavals[nn] = 1.0  # special case for latent inputs
        else:
            sname = rnamelist[csource[nn]]
        name = '{}-{}'.format(sname,rnamelist[ctarget[nn]])
        sim_reference.append({'name':name, 'B':betavals[nn]})

    df = pd.DataFrame(sim_reference)
    p,f = os.path.split(regiondataname)
    xlname = os.path.join(p,'sim_reference_values.xlsx')
    with pd.ExcelWriter(xlname) as writer:
        df.to_excel(writer, sheet_name='sim ref')

    Mintrinsic_base = np.zeros((Nintrinsic,tsize))
    if fintrinsic_count > 0:
        Mintrinsic_base[0,:] = copy.deepcopy(fintrinsic_base)
    shapes = ['fourier', 'square', 'saw']
    for nn in range(vintrinsic_count):
        n0 = nn % 3
        shape = shapes[n0]
        tt = (np.array(range(tsize))).astype(float)
        period = tsize/(3*nn+1)   # use Fourier terms to ensure independence
        phase = float(nn)*np.pi/2
        print('period = {}  shape = {}'.format(period,shape))
        if shape == 'fourier':
            tc = np.sin(2.0*np.pi*tt/period + phase)
        if shape == 'square':
            tc = np.sin(2.0*np.pi*tt/period + phase)
            tc[tc <= 0] = -0.5
            tc[tc > 0] = 0.5
        if shape == 'saw':
            tc = (tt % period)/period
            tc -= np.mean(tc)
        Mintrinsic_base[nn+fintrinsic_count,:] = copy.deepcopy(tc)

    # work backwards to create data sets
    Mconn[ctarget,csource] = copy.deepcopy(betavals)
    Minput[dtarget,dsource] = copy.deepcopy(deltavals)

    e,v = np.linalg.eig(Mconn)    # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
    Meigv = np.real(v[:,-Nintrinsic:])
    # Meigv = (Gram_Schmidt_orthogonalization(Meigv.T)).T  # make them a set of linearly indpendent eigvenvectors

    Sinput_base = Minput @ Meigv @ Mintrinsic_base   # simulated data
    Sinput_base += 0.05*np.random.normal(0,1,np.shape(Sinput_base))

    nregions = len(region_properties)
    # dict_keys(['tc', 'tc_sem', 'nruns_per_person', 'tsize', 'rname', 'DBname', 'DBnum', 'prefix'])
    for nn in range(nregions):

        if npeople > 0:  # override the number of runs in region_properties
            tc = copy.deepcopy(region_properties[nn]['tc'])
            nruns_per_person = copy.deepcopy(region_properties[nn]['nruns_per_person'])
            nclusters, tsize_big = np.shape(tc)
            tsize = copy.deepcopy(region_properties[nn]['tsize'])
            avg_runs_per_person = np.round(np.mean(nruns_per_person)).astype(int)
            nruns_per_person = (avg_runs_per_person*np.ones(npeople)).astype(int)
            nruns_total = np.sum(nruns_per_person).astype(int)
            tsize_big = tsize*nruns_total

            temp_tc = Sinput_base[nn,:][np.newaxis,:]
            temp_tc = np.repeat(temp_tc, nclusters, axis = 0)
            temp_tc = np.tile(temp_tc, nruns_total)

            new_tc = copy.deepcopy(temp_tc)
            new_tc_sem = np.zeros((nclusters,tsize_big))
        else:
            tc = copy.deepcopy(region_properties[nn]['tc'])
            nclusters, tsize_big = np.shape(tc)
            tsize = copy.deepcopy(region_properties[nn]['tsize'])
            nruns_per_person = copy.deepcopy(region_properties[nn]['nruns_per_person'])
            nruns_total = np.sum(nruns_per_person)

            temp_tc = Sinput_base[nn,:][np.newaxis,:]
            temp_tc = np.repeat(temp_tc, nclusters, axis = 0)
            temp_tc = np.repeat(temp_tc, nruns_total, axis = 1)

            new_tc = copy.deepcopy(temp_tc)
            new_tc_sem = copy.deepcopy(region_properties[nn]['tc_sem'])

        region_properties[nn]['tc'] = copy.deepcopy(new_tc)
        region_properties[nn]['tc_sem'] = copy.deepcopy(new_tc_sem)
        region_properties[nn]['nruns_per_person'] = copy.deepcopy(nruns_per_person)

    r['region_properties'] = copy.deepcopy(region_properties)
    p,f = os.path.split(regiondataname)
    outputname = os.path.join(p,'simdata_' + f)
    np.save(outputname, r)
    print('wrote simulated data to {}'.format(outputname))

    # covariates
    if len(covariatesname) > 0:
        p,f = os.path.split(covariatesname)
        cc = np.load(covariatesname, allow_pickle=True).flat[0]
        if npeople > 0:
            ncov,numcovpeople = np.shape(cc['GRPcharacteristicsvalues'])
            cc['GRPcharacteristicsvalues'] = np.random.randn(ncov,npeople)
        outputcovname = os.path.join(p, 'simdata_' + f)
        np.save(outputcovname, cc)
        print('wrote sim covariates data to {}'.format(outputcovname))
    else:
        outputcovname = covariatesname

    return outputname, outputcovname


def run_null_test_on_network(nsims, networkmodel, cnums, regiondataname, clusterdataname, timepoint = 'all', epoch = 'all', betascale = 0.1, Lweight = 1.0):
    resultsdir, networkfilename = os.path.split(networkmodel)
    networkbasename, ext = os.path.splitext(networkfilename)

    covariatesname = []
    null_regiondataname, null_covariates = generate_null_data_set(regiondataname, covariatesname, npeople=nsims, variable_variance = False)

    SAPMresultsname = os.path.join(resultsdir,'null_results.npy')
    SAPMparametersname = os.path.join(resultsdir,'null_params.npy')

    SAPMrun_V2(cnums, null_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
                epoch, betascale = betascale, Lweight = Lweight, reload_existing = False, multiple_output = False)

    # compile stats distributions for each connection
    results = np.load(SAPMresultsname, allow_pickle=True)
    params = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    csource = params['csource']
    ctarget = params['ctarget']
    rnamelist = params['rnamelist']
    fintrinsic_count = params['fintrinsic_count']
    vintrinsic_count = params['vintrinsic_count']
    rnamelist_full = copy.deepcopy(rnamelist)
    if fintrinsic_count > 0: rnamelist_full += ['latent0']
    for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count+nn)]

    ncon = len(results[0]['betavals'])
    betavals = np.zeros((ncon,nsims))
    for nn in range(nsims): betavals[:,nn] = results[nn]['betavals']
    bstats = []
    for nn in range(ncon):
        conname = '{}-{}'.format(rnamelist_full[csource[nn]], rnamelist_full[ctarget[nn]])
        b = copy.deepcopy(betavals[nn,:])
        entry = {'name':conname, 'mean':np.mean(b), 'std':np.std(b), 'skewness':scipy.stats.skew(b), 'kurtosis':scipy.stats.kurtosis(b)}
        bstats.append(entry)

    npyname = os.path.join(resultsdir, networkbasename + '_bstats.npy')
    np.save(npyname,bstats)

    try:
        xlname = os.path.join(resultsdir, networkbasename + '_bstats.xlsx')
        df = pd.DataFrame(bstats)
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='B stats')
    except:
        dateflag = time.ctime()
        dateflag = dateflag.replace(':','')
        dateflag = dateflag.replace(' ','')
        xlname = os.path.join(resultsdir, networkbasename + '_bstats ' + dateflag + '.xlsx')
        df = pd.DataFrame(bstats)
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='B stats')

    return xlname


def run_sim_test_on_network(nsims, networkmodel, cnums, regiondataname, clusterdataname, timepoint = 'all', epoch = 'all', betascale = 0.1, Lweight = 1.0):
    resultsdir, networkfilename = os.path.split(networkmodel)
    networkbasename, ext = os.path.splitext(networkfilename)

    covariatesname = []
    sim_regiondataname, sim_covariates = generate_simulated_data_set(regiondataname, covariatesname, networkmodel, clusterdataname, npeople=nsims,
                                variable_variance=False, timepoint='all', epoch='all')

    SAPMresultsname = os.path.join(resultsdir,'sim_results.npy')
    SAPMparametersname = os.path.join(resultsdir,'sim_params.npy')

    SAPMrun_V2(cnums, sim_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
                epoch, betascale = betascale, Lweight = Lweight, reload_existing = False, multiple_output = False)

    # compile stats distributions for each connection
    results = np.load(SAPMresultsname, allow_pickle=True)
    params = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    csource = params['csource']
    ctarget = params['ctarget']
    rnamelist = params['rnamelist']
    fintrinsic_count = params['fintrinsic_count']
    vintrinsic_count = params['vintrinsic_count']
    rnamelist_full = copy.deepcopy(rnamelist)
    if fintrinsic_count > 0: rnamelist_full += ['latent0']
    for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count+nn)]

    ncon = len(results[0]['betavals'])
    betavals = np.zeros((ncon,nsims))
    for nn in range(nsims): betavals[:,nn] = results[nn]['betavals']
    bstats = []
    for nn in range(ncon):
        conname = '{}-{}'.format(rnamelist_full[csource[nn]], rnamelist_full[ctarget[nn]])
        b = copy.deepcopy(betavals[nn,:])
        entry = {'name':conname, 'mean':np.mean(b), 'std':np.std(b), 'skewness':scipy.stats.skew(b), 'kurtosis':scipy.stats.kurtosis(b)}
        bstats.append(entry)

    npyname = os.path.join(resultsdir, networkbasename + '_bstats.npy')
    np.save(npyname,bstats)

    try:
        xlname = os.path.join(resultsdir, networkbasename + '_bstats.xlsx')
        df = pd.DataFrame(bstats)
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='B stats')
    except:
        dateflag = time.ctime()
        dateflag = dateflag.replace(':','')
        dateflag = dateflag.replace(' ','')
        xlname = os.path.join(resultsdir, networkbasename + '_bstats ' + dateflag + '.xlsx')
        df = pd.DataFrame(bstats)
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='B stats')

    return xlname


def Gram_Schmidt_orthogonalization(V):
	# take a set of vectors and make an orthogonal set out of them
	nv,tsize = np.shape(V)
	U = np.zeros((nv,tsize))  # new set
	for nn in range(nv):
		U[nn,:] = copy.deepcopy(V[nn,:])
		if nn > 0:
			projections = np.zeros(tsize)
			for mm in range(nn):
				proj = U[mm,:] * np.dot(U[mm,:],V[nn,:])/np.dot(U[mm,:],U[mm,:])
				projections += proj
			U[nn,:] -= projections
	return U


#
#
# if __name__ == '__main__':
#     main()
#
