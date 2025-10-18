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
# import pydatabase
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


def fft_lowpass_filter(data, cutoff):
    npoints = len(data)
    npoints2 = np.floor(npoints/2).astype(int)
    p1 = np.floor(cutoff*npoints2).astype(int) + 1
    p2 = npoints - p1 + 1

    f = np.fft.fft(data)
    f[p1:p2] = 0.0

    fdata = np.real(np.fft.ifft(f))
    return fdata


def fft_bandpass_filter(data, sampling_interval, cutoff_low, cutoff_high):

    npoints = len(data)
    npoints2 = np.floor(npoints/2).astype(int)

    frequencies = np.fft.fftfreq(npoints, d = sampling_interval)

    # low pass part
    phigh = np.where(np.abs(frequencies) >= cutoff_high)
    # p1 = np.floor(cutoff_high*npoints2).astype(int) + 1
    # p2 = npoints - p1 + 1

    f = np.fft.fft(data)
    f[phigh] = 0.0

    # high pass part
    plow = np.where(np.abs(frequencies) < cutoff_low)
    # p3 = np.floor(cutoff_low*npoints2).astype(int) + 1
    # f[:p3] = 0.0
    # f[-p3:] = 0.0
    f[plow] = 0.0

    fdata = np.real(np.fft.ifft(f))
    return fdata


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
    nclusterdict = []
    fintrinsic_base = []
    for nn in range(nregions):
        sem_region_list.append(dnclusters.loc[nn,'name'])
        cname = dnclusters.loc[nn,'name']
        if 'vintrinsic' in cname:  vintrinsic_count += 1
        if 'fintrinsic' in cname:  fintrinsic_count += 1
        entry = {'name':dnclusters.loc[nn,'name'],'nclusters':dnclusters.loc[nn,'nclusters']}
        nclusterdict.append(entry)

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

    return network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base


def sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag):  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv
    nr, ncomponents_to_fit = np.shape(loadings_fit)

    nv,nt = np.shape(Sinput)
    S_fit_diff_squared_per_person = np.sum((Sinput - fit) ** 2, axis=1)
    # S_fit_diff_squared_total = np.sum((Sinput - fit) ** 2)
    S_fit_diff_squared_total = np.sum(S_fit_diff_squared_per_person)
    S_squared_per_person = np.sum(Sinput ** 2, axis=1)
    # S_squared_total = np.sum(Sinput ** 2)
    S_squared_total = np.sum(S_squared_per_person)
    S_var = np.sqrt(nt)*np.var(Sinput, axis=1)   # keep the error terms consistent in scale across different numbers of time points

    R2list = 1.0 - S_fit_diff_squared_per_person / (S_squared_per_person + 1.0e-10)
    R2avg = np.mean(R2list)
    R2total = 1.0 - S_fit_diff_squared_total / (S_squared_total + 1.0e-10)

    error = np.mean(S_fit_diff_squared_per_person / (S_var + 1.0e-10) )

    # change cost function June 23 2024 - PWS
    # cr = np.where(regular_flag > 0)[0]
    # cost = np.mean(np.abs(betavals[cr]))  # L1 regularization, ignoring latents
    # cost2 = np.mean(np.abs(deltavals-1.0))  # L1 regularization, ignoring latents

    cost1 = np.mean(np.abs(betavals))  # L1 regularization
    # cost2 = np.mean(np.abs(deltavals-1.0))  # L1 regularization

    # cost = np.sqrt(nt)*(cost1 + cost2)   # try to keep the magnitude similar to the error term for different size data sets
    cost = np.sqrt(nt)*cost1   # try to keep the magnitude similar to the error term for different size data sets

    costfactor = Lweight*cost
    ssqd = error + costfactor
    return ssqd, error, cost, costfactor


def sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals, regular_flag):
    nv,nt = np.shape(Sinput)

    S_std = np.std(Sinput, axis=1)

    S_std = np.repeat(S_std[:, np.newaxis], nt, axis=1)
    S_std[S_std == 0.0] = 1.0
    diff = (Sinput - fit)/S_std   # impose similar effects to normalizing Sinput variance

    # back to the original
    S_fit_diff_squared_per_region = np.sum(diff**2, axis = 1)

    # # Huber method for resistance to outliers
    # S_fit_diff_squared_per_region = np.zeros(nv)
    # delta = 4.0
    # for nn in range(nv):
    #     alow= np.where(np.abs(diff[nn,:]) <= delta)[0]
    #     ahigh = np.where(np.abs(diff[nn,:]) > delta)[0]
    #     S_fit_diff_squared_per_region[nn] = np.sum(diff[nn,alow]**2) + np.sum(np.abs(diff[nn,ahigh]))

    # original---------------
    # S_fit_diff_squared_per_region = np.sum((Sinput - fit) ** 2, axis=1)
    # S_var = np.sqrt(nt)*np.var(Sinput, axis=1)   # keep the error terms consistent in scale across different numbers of time points

    # R2list = 1.0 - S_fit_diff_squared_per_region / (S_squared_per_region + 1.0e-10)
    # R2avg = np.mean(R2list)
    # R2total = 1.0 - S_fit_diff_squared_total / (S_squared_total + 1.0e-10)

    # original---------------
    # error = np.mean(S_fit_diff_squared_per_region / (S_var + 1.0e-10) )
    error = np.mean(S_fit_diff_squared_per_region)/np.sqrt(nt)   # scaling is to keep size of error similar for different numbers of time points

    cost1 = np.mean(np.abs(betavals))  # L1 regularization
    cost = np.sqrt(nt)*cost1   # try to keep the magnitude similar to the error term for different size data sets

    costfactor = Lweight*cost
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

    # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
    #                                     Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
    # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    # ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals, regular_flag)

    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd += 0.001*latent_cost

    # gradients for betavals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
        #                                         Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        # ssqdp, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, b, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, b, deltavals,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp += 0.001*latent_cost

        b = copy.deepcopy(betavals)
        b[nn] -= dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
        #                                         Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        # ssqdp2, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, b, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, b, deltavals,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqd2 += 0.001*latent_cost

        dssq_db[nn] = (ssqdp - ssqdp2) /( dval)

    # gradients for deltavals
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    dssq_dd = np.zeros(ndeltavals)
    for nn in range(ndeltavals):
        d = copy.deepcopy(deltavals)
        d[nn] += dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
        #                                         Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        # ssqdp, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, d, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, d,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp += 0.001*latent_cost

        d = copy.deepcopy(deltavals)
        d[nn] -= dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
        #                                         Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
        # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
        # ssqdp2, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, d, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, d,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp2 += 0.001*latent_cost

        dssq_dd[nn] = (ssqdp - ssqdp2) /( dval)

    #------------------------------------------
    # gradients for beta_int1

    b = copy.deepcopy(beta_int1)
    b += dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
    #                                             Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1, ncomponents_to_fit)
    # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    # ssqdp, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, b, fintrinsic1)
    ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                               regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqdp += 0.001*latent_cost

    b = copy.deepcopy(beta_int1)
    b -= dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)

    # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, components, loadings, Minput,
    #                                             Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1, ncomponents_to_fit)
    # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    # ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, b, fintrinsic1)
    ssqdps, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                               regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqdps += 0.001*latent_cost

    dssq_dbeta1 = (ssqdp - ssqdp2) / (dval)
    # dssq_dbeta1 = 0.0

    return dssq_db, dssq_dd, ssqd, dssq_dbeta1




def gradients_for_betavals_V4(Sinput, Minput, Mconn, betavals, deltavals, ctarget,
                              csource, dtarget, dsource, dval, fintrinsic_count, vintrinsic_count,
                              beta_int1, fintrinsic1, Lweight, regular_flag):

    # calculate change in error term with small changes in betavalues
    # include beta_int1
    nbetavals = len(betavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    ndeltavals = len(deltavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                             regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd += 0.001*latent_cost

    # gradients for betavals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, b, deltavals,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp += 0.001*latent_cost

        b = copy.deepcopy(betavals)
        b[nn] -= dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, b, deltavals,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp2 += 0.001*latent_cost

        dssq_db[nn] = (ssqdp - ssqdp2) /( dval)

    # gradients for deltavals
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    dssq_dd = np.zeros(ndeltavals)
    for nn in range(ndeltavals):
        d = copy.deepcopy(deltavals)
        d[nn] += dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, d,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp += 0.001*latent_cost

        d = copy.deepcopy(deltavals)
        d[nn] -= dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, d,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp2 += 0.001*latent_cost

        dssq_dd[nn] = (ssqdp - ssqdp2) /( dval)

    #-------------------------------------------
    # gradients for beta_int1

    b = copy.deepcopy(beta_int1)
    b += dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
    #                                             Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1, ncomponents_to_fit)
    # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    # ssqdp, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, b, fintrinsic1)
    ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                               regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqdp += 0.001*latent_cost

    b = copy.deepcopy(beta_int1)
    b -= dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)

    # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, components, loadings, Minput,
    #                                             Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1, ncomponents_to_fit)
    # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    # ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, b, fintrinsic1)
    ssqdps, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                               regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqdps += 0.001*latent_cost

    dssq_dbeta1 = (ssqdp - ssqdp2) / (dval)

    return dssq_db, dssq_dd, ssqd, dssq_dbeta1




def gradients_for_DBvals_V1(Sinput, Minput, Mconn, betavals, deltavals, ctarget,
                              csource, dtarget, dsource, dval, fintrinsic_count, vintrinsic_count,
                              beta_int1, fintrinsic1, Lweight, regular_flag):

    # calculate change in error term with small changes in DB values
    nbetavals = len(betavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    ndeltavals = len(deltavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                             regular_flag)
    # # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd += 0.001*latent_cost

    # gradients for DB vals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, b, deltavals,
                                                                     regular_flag)

        b = copy.deepcopy(betavals)
        b[nn] -= dval/2.0
        Mconn[ctarget, csource] = copy.deepcopy(b)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, b, deltavals,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp2 += 0.001 * latent_cost

        dssq_db[nn] = (ssqdp - ssqdp2) /( dval)

    return dssq_db, ssqd



def gradients_for_Dvals_V1(Sinput, Minput, Mconn, betavals, deltavals, ctarget,
                              csource, dtarget, dsource, dval, fintrinsic_count, vintrinsic_count,
                              beta_int1, fintrinsic1, Lweight, regular_flag):

    # calculate change in error term with small changes in D values
    # include beta_int1
    nbetavals = len(betavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    ndeltavals = len(deltavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                             regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd += 0.001*latent_cost

    # gradients for deltavals
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    dssq_dd = np.zeros(ndeltavals)
    for nn in range(ndeltavals):
        d = copy.deepcopy(deltavals)
        d[nn] += dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, d,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp += 0.001*latent_cost

        d = copy.deepcopy(deltavals)
        d[nn] -= dval/2.0
        Minput[dtarget, dsource] = copy.deepcopy(d)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp2, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, d,
                                                                     regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqdp2 += 0.001*latent_cost

        dssq_dd[nn] = (ssqdp - ssqdp2) /( dval)

    #-------------------------------------------
    # gradients for beta_int1

    b = copy.deepcopy(beta_int1)
    b += dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, b, fintrinsic1)
    ssqdp, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                               regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqdp += 0.001*latent_cost

    b = copy.deepcopy(beta_int1)
    b -= dval/2.0
    Mconn[ctarget, csource] = copy.deepcopy(betavals)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, b, fintrinsic1)
    ssqdps, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                               regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqdps += 0.001*latent_cost

    dssq_dbeta1 = (ssqdp - ssqdp2) / (dval)

    return dssq_dd, dssq_dbeta1, ssqd




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

    # if len(latent_flag) < len(betavals): latent_flag = np.zeros(len(betavals))
    nbetavals = len(betavals)
    updatebflag = np.zeros(nbetavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    dssq_db, dssq_dd, ssqd, dssq_dbeta1 = gradients_for_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals,
                                        deltavals, ctarget, csource, dtarget, dsource, dval,
                                        fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, ncomponents_to_fit)

    # c = np.where(csource >= nregion)[0]   # latent inputs
    # dssq_db[c] = 0.0   # over-ride changing the beta values from 1.0, for latent inputs
    #              get rid of this because the outputs (DB values) can be scaled, but the inputs (D) are not

    change_betavals = alpha * dssq_db
    change_deltavals = alpha * dssq_dd

    change_betavals[change_betavals > change_weight_limit] = change_weight_limit
    change_betavals[change_betavals < -change_weight_limit] = -change_weight_limit
    change_deltavals[change_deltavals > change_weight_limit] = change_weight_limit
    change_deltavals[change_deltavals < -change_weight_limit] = -change_weight_limit

    betavals -= change_betavals
    deltavals -= change_deltavals
    deltavals[deltavals < 0] = 0.0   # deltavals must be positive

    betavals[betavals > betalimit] = copy.deepcopy(betalimit)
    betavals[betavals < -betalimit] = copy.deepcopy(-betalimit)

    # keep beta_int1 fixed at 1.0
    beta_int1 -= alphabint * dssq_dbeta1

    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    # scale deltavals to max value = 1.0, the rest are relative values
    # Minput = scale_deltavals(Minput)
    # deltavals = copy.deepcopy(Minput[dtarget, dsource]) # update values in case they have changed

    # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
    #                                                     Mconn, fintrinsic_count, vintrinsic_count, beta_int1,
    #                                                     fintrinsic1, ncomponents_to_fit)
    #
    # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    # ssqd_new, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd_new, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd_new += 0.001*latent_cost

    return betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd, ssqd_new, alpha, alphabint


def update_betavals_V4(Sinput, Minput, Mconn, betavals, deltavals, betalimit, ctarget, csource,
                       dtarget, dsource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag,
                       alpha, alphabint, latent_flag = []):

    # calculate change in error term with small changes in betavalues
    # include beta_int1
    nregion,tsize = np.shape(Sinput)
    change_weight_limit = 0.1

    starting_betavals = copy.deepcopy(betavals)
    starting_deltavals = copy.deepcopy(deltavals)
    starting_beta_int1 = copy.deepcopy(beta_int1)

    # if len(latent_flag) < len(betavals): latent_flag = np.zeros(len(betavals))
    nbetavals = len(betavals)
    updatebflag = np.zeros(nbetavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    # dssq_db, dssq_dd, ssqd, dssq_dbeta1 = gradients_for_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals,
    #                                     deltavals, ctarget, csource, dtarget, dsource, dval,
    #                                     fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, ncomponents_to_fit)

    dssq_db, dssq_dd, ssqd, dssq_dbeta1 = gradients_for_betavals_V4(Sinput, Minput, Mconn, betavals,
                                        deltavals, ctarget, csource, dtarget, dsource, dval,
                                        fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
                                        regular_flag)


    # c = np.where(csource >= nregion)[0]   # latent inputs
    # dssq_db[c] = 0.0   # over-ride changing the beta values from 1.0, for latent inputs
    #              get rid of this because the outputs (DB values) can be scaled, but the inputs (D) are not

    change_betavals = alpha * dssq_db
    change_deltavals = alpha * dssq_dd

    change_betavals[change_betavals > change_weight_limit] = change_weight_limit
    change_betavals[change_betavals < -change_weight_limit] = -change_weight_limit
    change_deltavals[change_deltavals > change_weight_limit] = change_weight_limit
    change_deltavals[change_deltavals < -change_weight_limit] = -change_weight_limit

    betavals -= change_betavals
    deltavals -= change_deltavals
    deltavals[deltavals < 0] = 0.0   # deltavals must be positive

    betavals[betavals > betalimit] = copy.deepcopy(betalimit)
    betavals[betavals < -betalimit] = copy.deepcopy(-betalimit)

    # keep beta_int1 fixed at 1.0
    beta_int1 -= alphabint * dssq_dbeta1

    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    # scale deltavals to max value = 1.0, the rest are relative values
    # Minput = scale_deltavals(Minput)
    # deltavals = copy.deepcopy(Minput[dtarget, dsource]) # update values in case they have changed

    # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
    #                                                     Mconn, fintrinsic_count, vintrinsic_count, beta_int1,
    #                                                     fintrinsic1, ncomponents_to_fit)
    #
    # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
    # ssqd_new, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd_new, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd_new += 0.001*latent_cost

    return betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd, ssqd_new, alpha, alphabint




def update_DBvals_V1(Sinput, Minput, Mconn, betavals, deltavals, betalimit, ctarget, csource,
                       dtarget, dsource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag,
                       alpha, alphabint, latent_flag = []):
    # update only DB values, not D

    # calculate change in error term with small changes in betavalues
    # include beta_int1
    nregion,tsize = np.shape(Sinput)
    change_weight_limit = 0.1

    starting_betavals = copy.deepcopy(betavals)
    starting_deltavals = copy.deepcopy(deltavals)
    starting_beta_int1 = copy.deepcopy(beta_int1)

    # if len(latent_flag) < len(betavals): latent_flag = np.zeros(len(betavals))
    nbetavals = len(betavals)
    updatebflag = np.zeros(nbetavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    dssq_db, ssqd = gradients_for_DBvals_V1(Sinput, Minput, Mconn, betavals,
                                        deltavals, ctarget, csource, dtarget, dsource, dval,
                                        fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
                                        regular_flag)

    change_betavals = alpha * dssq_db

    change_betavals[change_betavals > change_weight_limit] = change_weight_limit
    change_betavals[change_betavals < -change_weight_limit] = -change_weight_limit

    betavals -= change_betavals

    betavals[betavals > betalimit] = copy.deepcopy(betalimit)
    betavals[betavals < -betalimit] = copy.deepcopy(-betalimit)

    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd_new, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd_new += 0.001*latent_cost

    return betavals, deltavals, beta_int1, fit, dssq_db, ssqd, ssqd_new


def update_Dvals_V1(Sinput, Minput, Mconn, betavals, deltavals, betalimit, ctarget, csource,
                       dtarget, dsource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag,
                       alpha, alphabint, latent_flag = []):

    # calculate change in error term with small changes in betavalues
    # include beta_int1
    nregion,tsize = np.shape(Sinput)
    change_weight_limit = 0.3

    starting_betavals = copy.deepcopy(betavals)
    starting_deltavals = copy.deepcopy(deltavals)
    starting_beta_int1 = copy.deepcopy(beta_int1)

    # if len(latent_flag) < len(betavals): latent_flag = np.zeros(len(betavals))
    nbetavals = len(betavals)
    updatebflag = np.zeros(nbetavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    dssq_dd, dssq_dbeta1, ssqd = gradients_for_Dvals_V1(Sinput, Minput, Mconn, betavals,
                                        deltavals, ctarget, csource, dtarget, dsource, dval,
                                        fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
                                        regular_flag)

    change_deltavals = alpha * dssq_dd

    change_deltavals[change_deltavals > change_weight_limit] = change_weight_limit
    change_deltavals[change_deltavals < -change_weight_limit] = -change_weight_limit

    deltavals -= change_deltavals
    deltavals[deltavals < 0] = 0.0   # deltavals must be positive

    beta_int1 -= alphabint * dssq_dbeta1

    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    Minput[dtarget, dsource] = copy.deepcopy(deltavals)

    fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                vintrinsic_count, beta_int1, fintrinsic1)
    ssqd_new, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
    # add cost for correlated latents
    # lcc = np.corrcoef(Mintrinsic)
    # latent_cost = np.sum(np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
    # ssqd_new += 0.001*latent_cost

    return betavals, deltavals, beta_int1, fit, dssq_dd, dssq_dbeta1, ssqd, ssqd_new


def scale_deltavals(Minput):
    nt,nl = np.shape(Minput)
    for nn in range(nt):
        v = Minput[:,nn]  # outputs from one source
        v /= np.max(np.abs(v))   # scale to max magnitude = 1.0
        Minput[:,nn] = v
    return Minput

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

    # -----------------------------------------
    # how to handle eigenvectors when intrinsic inputs are to more than one region?  Is there a difference?
    for aa in range(Nintrinsic):
        Meigv[:,aa] = Meigv[:,aa]/Meigv[(-Nintrinsic+aa),aa]

    if fintrinsic_count > 0:
        # if len(beta_int1) > 1:
        #     Nfintrinsic = len(beta_int1)
        # else:
        #     Nfintrinsic = 1

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

        # Sinput[:nregions,:] = Minput @ Meigv @ Mintrinsic   #  [nr x tsize] = [nr x ncon] @ [ncon x Nint] @ [Nint x tsize]
        # Mintrinsic = np.linalg.inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sin

        partial_fit = (Minput @ Meigv[:,0])[:,np.newaxis] @ Mint_fixed    # is this right?

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



def network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1):
    # # Sinput[:nregions,:] = Minput @ Meigv @ Mintrinsic   #  [nr x tsize] = [nr x ncon] @ [ncon x Nint] @ [Nint x tsize]
    # # Mintrinsic = np.linalg.inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sin
    # fit based on eigenvectors alone, with intrinsic values calculated

    # if ncomponents_to_fit < 1:
    #     nregion,tsize = np.shape(Sinput)
    #     ncomponents_to_fit = copy.deepcopy(nregion)

    nregions,tsize_total = np.shape(Sinput)
    Nintrinsic = fintrinsic_count + vintrinsic_count
    e,v = np.linalg.eig(Mconn)    # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
    Meigv = np.real(v[:,-Nintrinsic:])

    # -----------------------------------------
    for aa in range(Nintrinsic):
        Meigv[:,aa] = Meigv[:,aa]/Meigv[(-Nintrinsic+aa),aa]

    if fintrinsic_count > 0:
        # separate fintrinsic1 from components and loadings
        # fit the fixed intrinsic, remove it, and then fit the variable intrinsics to the remainder
        Mintrinsic = np.zeros((Nintrinsic, tsize_total))

        # Mint_variable = np.linalg.inv(M1[:,1:].T @ M1[:,1:]) @ M1[:,1:].T @ Sinput
        Mint_fixed = beta_int1*fintrinsic1[np.newaxis,:]
        partial_fit = (Minput @ Meigv[:,0])[:,np.newaxis] @ Mint_fixed    # is this right?

        residual = Sinput-partial_fit
        M1r = Minput @ Meigv[:,1:]

        Mint_variable = np.linalg.inv(M1r.T @ M1r) @ M1r.T @ residual

        Mintrinsic[0,:] = copy.deepcopy(Mint_fixed)
        Mintrinsic[1:,:] = copy.deepcopy(Mint_variable)

        fit = Minput @ Meigv @ Mintrinsic
    else:
        M1 = Minput @ Meigv
        Mintrinsic = np.linalg.inv(M1.T @ M1) @ M1.T @ Sinput
        fit = Minput @ Meigv @ Mintrinsic

    err = np.sum((Sinput - fit)**2)

    return fit, Mintrinsic, Meigv, err

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

    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    networktargetlist = [network[x]['target'] for x in range(len(network))]

    filtered_cluster_properties = []
    for nn, targetname in enumerate(networktargetlist):
        if targetname in clusterregionlist:
            x = clusterregionlist.index(targetname)
            filtered_cluster_properties.append(cluster_properties[x])

    return filtered_cluster_properties


def load_filtered_regiondata(regiondataname, networkfile):
    region_data = np.load(regiondataname, allow_pickle=True).flat[0]
    region_properties = copy.deepcopy(region_data['region_properties'])

    regionnamelist = [region_properties[x]['rname'] for x in range(len(region_properties))]

    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    networktargetlist = [network[x]['target'] for x in range(len(network))]

    filtered_region_properties = []
    for nn, targetname in enumerate(networktargetlist):
        if targetname in regionnamelist:
            x = regionnamelist.index(targetname)
            filtered_region_properties.append(region_properties[x])

    region_data['region_properties'] = copy.deepcopy(filtered_region_properties)
    return region_data

#---------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint = 'all', epoch = 'all', run_whole_group = False, normalizevar = False):

    outputdir, f = os.path.split(SAPMparametersname)
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)

    fintrinsic_region = []
    if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
        for nn in range(len(network)):
            sources = network[nn]['sources']
            if 'fintrinsic1' in sources:
                fintrinsic_region = network[nn]['targetnum']  # only one region should have this input

    # region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
    region_data1 = load_filtered_regiondata(regiondataname, networkfile)

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

    if run_whole_group:
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
                                  epoch = 'all', run_whole_group = False, normalizevar = False, filter_tcdata = False,
                                  subsample = []):
# model each region as having a single output that is common to all regions it projects to
# But ...(new for V2) allow for scaling of the input to each region, by varying the values in Minput
#
    outputdir, f = os.path.split(SAPMparametersname)
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)

    fintrinsic_region = []
    if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
        for nn in range(len(network)):
            sources = network[nn]['sources']
            if 'fintrinsic1' in sources:
                fintrinsic_region = network[nn]['targetnum']  # only one region should have this input

    # region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
    region_data1 = load_filtered_regiondata(regiondataname, networkfile)

    region_properties = copy.deepcopy(region_data1['region_properties'])
    DBname = copy.deepcopy(region_data1['DBname'])
    DBnum = copy.deepcopy(region_data1['DBnum'])


    # subsample the data if requested
    if len(subsample) > 0:
        samplesplit = subsample[0]
        samplestart = subsample[1]
        # filename_list, dbnum_person_list, NP = pydatabase.get_datanames_by_person(DBname, DBnum, '', mode='list')

        nruns_per_person = copy.deepcopy(region_properties[0]['nruns_per_person'])
        tsize = copy.deepcopy(region_properties[0]['tsize'])
        nruns_total = np.sum(nruns_per_person)
        NP = len(nruns_per_person)
        sample = list(range(samplestart,NP,samplesplit))
        runs_to_keep = []
        for ss in sample:
            rbase = np.sum(nruns_per_person[:ss])
            runs_to_keep += list(np.array(range(nruns_per_person[ss])) + rbase)
        tp_keep = []
        for rr in runs_to_keep:
            t1 = rr*tsize
            t2 = (rr+1)*tsize
            tp_keep += list(range(t1,t2))

        nregions = len(region_properties)
        region_properties2 = copy.deepcopy(region_properties)
        DBnum2 = copy.deepcopy(DBnum[runs_to_keep])
        nruns_per_person2 = copy.deepcopy(nruns_per_person[sample])

        for rr in range(nregions):
            region_properties2[rr]['tc'] = copy.deepcopy(region_properties[rr]['tc'][:,tp_keep])
            region_properties2[rr]['tc_sem'] = copy.deepcopy(region_properties[rr]['tc_sem'][:,tp_keep])
            region_properties2[rr]['tc_original'] = copy.deepcopy(region_properties[rr]['tc_original'][:,tp_keep])
            region_properties2[rr]['tc_sem_original'] = copy.deepcopy(region_properties[rr]['tc_sem_original'][:,tp_keep])
            region_properties2[rr]['nruns_per_person'] = copy.deepcopy(nruns_per_person2)
            region_properties2[rr]['DBnum'] = copy.deepcopy(DBnum2)

        region_properties = copy.deepcopy(region_properties2)
        DBnum = copy.deepcopy(DBnum2)
        region_data1 = {'region_properties':region_properties, 'DBname':DBname, 'DBnum':DBnum}


#---------------------------------------------------------
    # check for bad data--------------------------------------
    nregions = len(region_properties)
    nruns_per_person = copy.deepcopy(region_properties[0]['nruns_per_person'])
    nruns_cumulative = np.cumsum(nruns_per_person)
    nruns_total = np.sum(nruns_per_person)
    tsize = copy.deepcopy(region_properties[0]['tsize'])
    bad_runs = []
    bad_tp = []
    nbad = 0
    for rr in range(nregions):
        tc = copy.deepcopy(region_properties[rr]['tc'])
        for nn in range(nruns_total):
            t1 = nn * tsize
            t2 = (nn + 1) * tsize
            tp = list(range(t1, t2))
            check = (np.sum(tc[:, tp] ** 2, axis=1) == 0.).any()
            if check:
                nbad += 1
                bad_runs += [nn]
                bad_tp += [tp]

    if nbad > 0:
        bad_runs = np.unique(bad_runs)
        all_runs = list(range(nruns_total))
        good_runs = [all_runs[x] for x in range(len(all_runs)) if all_runs[x] not in bad_runs]
        bad_tp = np.unique(bad_tp)
        region_properties2 = copy.deepcopy(region_properties)
        nruns_per_person2 = copy.deepcopy(nruns_per_person)
        DBnum2 = copy.deepcopy(DBnum[good_runs])
        bad_DBnum = copy.deepcopy(DBnum[bad_runs])
        tp_all = list(range(nruns_total*tsize))
        good_tp = [tp_all[x] for x in range(len(tp_all)) if tp_all[x] not in bad_tp]
        bad_people = []
        for bb in bad_runs:
            cc = np.where(nruns_cumulative > bb)[0][0]
            nruns_per_person2[cc] -= 1
            bad_people += [cc]
        nruns_per_person2 = nruns_per_person2[nruns_per_person2 > 0]

        for rr in range(nregions):
            region_properties2[rr]['tc'] = copy.deepcopy(region_properties[rr]['tc'][:,good_tp])
            region_properties2[rr]['tc_sem'] = copy.deepcopy(region_properties[rr]['tc_sem'][:,good_tp])
            region_properties2[rr]['tc_original'] = copy.deepcopy(region_properties[rr]['tc_original'][:,good_tp])
            region_properties2[rr]['tc_sem_original'] = copy.deepcopy(region_properties[rr]['tc_sem_original'][:,good_tp])
            region_properties2[rr]['nruns_per_person'] = copy.deepcopy(nruns_per_person2)
            region_properties2[rr]['DBnum'] = copy.deepcopy(DBnum2)

        region_properties = copy.deepcopy(region_properties2)
        DBnum = copy.deepcopy(DBnum2)
        region_data1 = {'region_properties':region_properties, 'DBname':DBname, 'DBnum':DBnum}

        bad_record, bad_counts = np.unique(bad_people, return_counts=True)
        print('Appears to be bad data in  DBnums {}'.format(bad_DBnum))
        for bb in range(len(bad_record)):
            print('   person {}  with {} bad runs'.format(bad_record[bb], bad_counts[bb]))
    #-------------end of check -------------------------------
    #---------------------------------------------------------

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
                # cutoff = 0.5

                # eventually these will be inputs if filtering is added as an option
                cutoff_high = 0.1  # for lowpass filter
                cutoff_low = 0.01  # for highpass filter
                TR = 6.75
                for ii in range(nregions):
                    # tcdata_centered[ii, tp] = butter_lowpass_filter(tcdata_centered[ii, tp], cutoff, fs, order=5)
                    for rr in range(nruns_per_person[nn]):
                        rt1 = rr*tsize
                        rt2 = (rr+1)*tsize
                        tcdata_centered[ii, rt1:rt2] = fft_bandpass_filter(tcdata_centered[ii, rt1:rt2], TR, cutoff_low, cutoff_high)

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

        new_std = copy.deepcopy(avg_std)    # normalize everything to the same variance
        std_scale = new_std/(tcdata_std + 1.0e-20)

        # flag extreme values
        print('max/min of std_scale = {:.3e} / {:.3e}'.format(np.max(std_scale), np.min(std_scale)))
        checkr, checkp = np.where( (std_scale < 1e-6) | (std_scale > 1e6))
        std_scale[checkr,checkp] = 1.0
        print('after check max/min of std_scale = {:.3e} / {:.3e}'.format(np.max(std_scale), np.min(std_scale)))


        for nn in range(NP):
            tpoints = copy.deepcopy(tplist1[nn]['tp'])
            scale_factor = np.repeat(std_scale[:,nn][:,np.newaxis],len(tpoints),axis=1)
            # scale_factor = np.repeat(tcdata_std[:,nn][:,np.newaxis],len(tpoints),axis=1)/avg_std
            tcdata_centered[:, tpoints] *= scale_factor
    else:
        tcdata_std = []
        std_scale = []

    tplist_full.append(tplist1)

    if run_whole_group:  # concatentated data
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

    if run_whole_group_averaged:  # averaged data
        # special case to fit the full group together
        # treat the whole group like one person
        tcdata_centered_avg = np.zeros((nregions, tsize))
        tpgroup_full = []
        tpgroup = []
        tp = []
        for nn in range(NP):
            r1 = sum(nruns_per_person[:nn])
            r2 = sum(nruns_per_person[:(nn + 1)])
            for ee2 in range(r1, r2):
                t1 = ee2*tsize
                t2 = (ee2+1)*tsize
                tcdata_centered_avg += tcdata_centered[:,t1:t2]

        tcdata_centered_avg /= nruns_total
        nruns_per_person_new = [1]
        tplist_full_new = []
        tplist_full_new[0][0]['tp'] = tplist_full[0][0]['tp']

        nruns_per_person = copy.deepcopy(nruns_per_person_new)
        tplist_full = copy.deepcopy(tplist_full_new)
        tcdata_centered = copy.deepcopy(tcdata_centered_avg)

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
            latent_flag[nn] = csource[nn] >= nregions

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
# prep data for single-output fully-connected model
#---------------------------------------------------------------------------------
def prep_data_sem_physio_model_SO_FC(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint = 'all',
                                  epoch = 'all', cnums = [], run_whole_group = False, normalizevar = False, filter_tcdata = False):
# model each region as having a single output that is common to all regions it projects to
# But ...(new for V2) allow for scaling of the input to each region, by varying the values in Minput
#
# Nov 11, 2024:   New fully-connected model - every cluster projects to every cluster of other regions
#
    outputdir, f = os.path.split(SAPMparametersname)
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)

    fintrinsic_region = []
    if fintrinsic_count > 0:  # find which regions have fixed intrinsic input
        for nn in range(len(network)):
            sources = network[nn]['sources']
            if 'fintrinsic1' in sources:
                fintrinsic_region += [network[nn]['targetnum']]  # more than one region can have this input for FC models

    # region_data1 = np.load(regiondataname, allow_pickle=True).flat[0]
    region_data1 = load_filtered_regiondata(regiondataname, networkfile)

    region_properties = copy.deepcopy(region_data1['region_properties'])
    DBname = copy.deepcopy(region_data1['DBname'])
    DBnum = copy.deepcopy(region_data1['DBnum'])

    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)

    nregions = len(cluster_properties)
    # nclusterlist refers to the clusters in the data set, not in the network model
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]

    # if len(cnums) < len(nclusterlist):  # use all clusters if not specified
    #     cnums = [{'cnums':list(range(nclusterlist[x]))} for x in range(len(nclusterlist))]
    # nclusters_to_use = [len(cnums[x]['cnums']) for x in range(len(cnums))]

    # default is to use all clusters for fully-connected
    if len(cnums) < len(rnamelist):
        # cnums = [{'cnums':list(range(nclusterlist[x]))} for x in range(len(nclusterlist))]
        cnums_temp = []
        for x in range(len(cnums)):
            if np.min(cnums[x]['cnums']) < 0:
                cnums_temp.append({'cnums':list(range(nclusterlist[x]))})
            else:
                cnums_temp.append({'cnums':cnums[x]['cnums']})
        for x in range(len(cnums),len(nclusterlist)):
            cnums_temp.append({'cnums':list(range(nclusterlist[x]))})
        cnums = copy.deepcopy(cnums_temp)

    nclusters_to_use = [len(cnums[x]['cnums']) for x in range(len(cnums))]
    nclusters_to_use_total = np.sum(nclusters_to_use).astype(int)

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
                # cutoff = 0.2
                cutoff = 1.0
                for ii in range(nregions):
                    # tcdata_centered[ii, tp] = butter_lowpass_filter(tcdata_centered[ii, tp], cutoff, fs, order=5)
                    tcdata_centered[ii, :] = fft_lowpass_filter(tcdata_centered[ii, :], cutoff)

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

        # new_min_std = 0.75*avg_std
        # new_max_std = 1.25*avg_std

        new_std = copy.deepcopy(avg_std)    # normalize everything to the same variance
        std_scale = new_std/(tcdata_std + 1.0e-20)
        # testing idea about std_scale
        # std_scale = (0.5*tcdata_std + 0.5*new_std)/(tcdata_std + 1.0e-20)

        # flag extreme values
        print('max/min of std_scale = {:.3e} / {:.3e}'.format(np.max(std_scale), np.min(std_scale)))
        checkr, checkp = np.where( (std_scale < 1e-6) | (std_scale > 1e6))
        std_scale[checkr,checkp] = 1.0
        print('after check max/min of std_scale = {:.3e} / {:.3e}'.format(np.max(std_scale), np.min(std_scale)))


        for nn in range(NP):
            tpoints = copy.deepcopy(tplist1[nn]['tp'])
            scale_factor = np.repeat(std_scale[:,nn][:,np.newaxis],len(tpoints),axis=1)
            # scale_factor = np.repeat(tcdata_std[:,nn][:,np.newaxis],len(tpoints),axis=1)/avg_std
            tcdata_centered[:, tpoints] *= scale_factor
    else:
        tcdata_std = []
        std_scale = []

    tplist_full.append(tplist1)

    if run_whole_group:  # concatentated data
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
        target = copy.deepcopy(network[nn]['targetnum'])
        sources = copy.deepcopy(network[nn]['sourcenums'])
        print('sources = {}'.format(sources))

        target_fc = int(np.sum(nclusters_to_use[:target]))
        nclusters_target_fc = int(nclusters_to_use[target])

        for ttfc in range(target_fc,(target_fc + nclusters_target_fc)):
            targetnumlist += [ttfc]
            for mm in range(len(sources)):
                source = copy.deepcopy(sources[mm])

                if source >= nregions:  # latent input
                    source_fc = nclusters_to_use_total + (source-nregions)
                    nclusters_source_fc = 1
                else:
                    source_fc = np.sum(nclusters_to_use[:source]).astype(int)
                    nclusters_source_fc = int(nclusters_to_use[source])

                for ssfc in range(source_fc,(source_fc + nclusters_source_fc)):
                    sourcelist += [ssfc]
                    betaname = '{}_{}'.format(ssfc, ttfc)
                    entry = {'name': betaname, 'number': nbeta, 'pair': [ssfc, ttfc]}
                    beta_list.append(entry)
                    beta_id += [1000 * ssfc + ttfc]
                    nbeta += 1

    ncon = nbeta - Nintrinsic

    # reorder to put intrinsic inputs at the end-------------
    beta_list2 = []
    beta_id2 = []
    x = np.where(np.array(sourcelist) < nclusters_to_use_total)[0]
    for xx in x:
        beta_list2.append(beta_list[xx])
        beta_id2 += [beta_id[xx]]
    for sn in range(nclusters_to_use_total, nclusters_to_use_total + Nintrinsic):
        x = np.where(np.array(sourcelist) == sn)[0]
        for xx in x:
            beta_list2.append(beta_list[xx])
            beta_id2 += [beta_id[xx]]

    for nn in range(len(beta_list2)):
        beta_list2[nn]['number'] = nn

    beta_list = beta_list2
    beta_id = beta_id2

    beta_pair = []
    Mconn = np.zeros((nclusters_to_use_total + Nintrinsic, nclusters_to_use_total + Nintrinsic))
    count = 0
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']

        target_fc = np.sum(nclusters_to_use[:target]).astype(int)
        nclusters_target_fc = int(nclusters_to_use[target])

        for ttfc in range(target_fc,(target_fc + nclusters_target_fc)):
            for mm in range(len(sources)):
                source = sources[mm]
                if source >= nregions:  # latent input
                    source_fc = nclusters_to_use_total + (source-nregions)
                    nclusters_source_fc = 1
                else:
                    source_fc = np.sum(nclusters_to_use[:source]).astype(int)
                    nclusters_source_fc = int(nclusters_to_use[source])

                for ssfc in range(source_fc,(source_fc + nclusters_source_fc)):
                    conn1 = beta_id.index(ssfc * 1000 + ttfc)

                    count += 1
                    beta_pair.append([ttfc, ssfc])
                    Mconn[ttfc, ssfc] = count

                    if ssfc >= nclusters_to_use_total:  # intrinsic input
                        Mconn[ssfc, ssfc] = 1  # set the intrinsic beta values

    # prep to index Mconn for updating beta values
    beta_pair = np.array(beta_pair)
    ctarget = beta_pair[:, 0]
    csource = beta_pair[:, 1]

    latent_flag = np.zeros(len(ctarget))
    found_latent_list = []
    for nn in range(len(ctarget)):
        # if csource[nn] >= ncon  and ctarget[nn] < ncon:
        if csource[nn] >= nclusters_to_use_total  and ctarget[nn] < nclusters_to_use_total:
            found_latent_list += [csource[nn]]
            occurence = np.count_nonzero(found_latent_list == csource[nn])
            latent_flag[nn] = csource[nn] >= nclusters_to_use_total

    reciprocal_flag = np.zeros(len(ctarget))
    for nn in range(len(ctarget)):
        spair = beta_list[csource[nn]]['pair']
        tpair = beta_list[ctarget[nn]]['pair']
        if spair[0] == tpair[1]:
            reciprocal_flag[nn] = 1

    # setup Minput matrix--------------------------------------------------------------
    # Sconn = Mconn @ Sconn    # propagate the intrinsic inputs through the network
    # Sinput = Minput @ Sconn
    # Minput = np.zeros((nregions, nbeta))  # mixing of connections to model the inputs to each region
    Minput = np.zeros((nclusters_to_use_total, nclusters_to_use_total+Nintrinsic))  # mixing of connections to model the inputs to each region
    betanamelist = [beta_list[a]['name'] for a in range(nbeta)]
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']

        target_fc = np.sum(nclusters_to_use[:target]).astype(int)
        nclusters_target_fc = nclusters_to_use[target]

        for ttfc in range(target_fc,(target_fc + nclusters_target_fc)):
            for mm in range(len(sources)):
                source = sources[mm]
                if source >= nregions:  # latent input
                    source_fc = nclusters_to_use_total + (source-nregions)
                    nclusters_source_fc = 1
                else:
                    source_fc = np.sum(nclusters_to_use[:source]).astype(int)
                    nclusters_source_fc = int(nclusters_to_use[source])

                for ssfc in range(source_fc,(source_fc + nclusters_source_fc)):
                    # betaname = '{}_{}'.format(source, target)
                    # x = betanamelist.index(betaname)
                    # Minput[target, x] = 1
                    Minput[ttfc, ssfc] = 1

    # flag which Minput values can be varied
    # (keep one output from each region at 1, vary the other outputs)
    Dvarflag = copy.deepcopy(Minput)
    for nn in range(nclusters_to_use_total):
        onesource = copy.deepcopy(Minput[:,nn])
        c = np.where(onesource > 0)[0]
        if len(c) > 0: onesource[c[0]] = 0
        Dvarflag[:, nn] = copy.deepcopy(onesource)
    for nn in range(nclusters_to_use_total+fintrinsic_count,nclusters_to_use_total+Nintrinsic):
        # do the same thing for latent inputs for multiple clusters
        # Dvarflag[:,nn] = 0  # original
        onesource = copy.deepcopy(Minput[:, nn])
        c = np.where(onesource > 0)[0]
        if len(c) > 0: onesource[c[0]] = 0
        Dvarflag[:, nn] = copy.deepcopy(onesource)

    dtarget,dsource = np.where(Dvarflag > 0)

    # save parameters for looking at results later
    SAPMparams = {'betanamelist': betanamelist, 'beta_list': beta_list, 'nruns_per_person': nruns_per_person,
                 'nclusterstotal': nclusterstotal, 'cnums':cnums, 'rnamelist': rnamelist, 'nregions': nregions,
                 'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
                 'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
                 'fintrinsic_region':fintrinsic_region, 'sem_region_list': sem_region_list,
                 'nclusterlist': nclusterlist, 'tsize': tsize,'tplist_full': tplist_full,
                 'tcdata_centered': tcdata_centered, 'tcdata_centered_original': tcdata_centered_original,
                  'ctarget':ctarget ,'csource':csource, 'dtarget':dtarget ,'dsource':dsource,
                 'Mconn':Mconn, 'Minput':Minput, 'timepoint':timepoint, 'epoch':epoch, 'latent_flag':latent_flag,
                  'reciprocal_flag':reciprocal_flag, 'fintrinsic_base':fintrinsic_base, 'DBname':DBname, 'DBnum':DBnum}  # , 'ktarget':ktarget ,'ksource':ksource
    if normalizevar:
        SAPMparams['tcdata_std'] = tcdata_std
        SAPMparams['std_scale'] = std_scale

    print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
    np.save(SAPMparametersname, SAPMparams)


#
# #----------------------------------------------------------------------------------
# # primary function--------------------------------------------------------------------
# def sem_physio_model1_V3(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [],
#                       betascale = 0.1, Lweight = 1.0, normalizevar=False, nitermax_stage2 = 1000, nitermax_stage1 = 100,
#                       nsteps_stage1 = 30, converging_slope_limit = [0.01,0.001], verbose = True):
#
# # this version fits to principal components of Sinput
#     save_test_record = True
#     test_record_name = r'E:/FM2021data/gradient_descent_record.npy'
#     test_record = []
#     test_person = 1
#
#     starttime = time.ctime()
#
#     # initialize gradient-descent parameters--------------------------------------------------------------
#     initial_alpha = 1e-3
#     initial_Lweight = copy.deepcopy(Lweight)
#     initial_dval = 0.05
#     alpha_limit = 1.0e-5
#     repeat_limit = 2
#
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     # load the data values
#     betanamelist = SAPMparams['betanamelist']
#     beta_list = SAPMparams['beta_list']
#     nruns_per_person = SAPMparams['nruns_per_person']
#     nclusterstotal = SAPMparams['nclusterstotal']
#     rnamelist = SAPMparams['rnamelist']
#     nregions = SAPMparams['nregions']
#     cluster_properties = SAPMparams['cluster_properties']
#     cluster_data = SAPMparams['cluster_data']
#     network = SAPMparams['network']
#     fintrinsic_count = SAPMparams['fintrinsic_count']
#     vintrinsic_count = SAPMparams['vintrinsic_count']
#     sem_region_list = SAPMparams['sem_region_list']
#     nclusterlist = SAPMparams['nclusterlist']
#     tsize = SAPMparams['tsize']
#     tplist_full = SAPMparams['tplist_full']
#     tcdata_centered = SAPMparams['tcdata_centered']
#     tcdata_centered_original = SAPMparams['tcdata_centered_original']
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     dtarget = SAPMparams['dtarget']
#     dsource = SAPMparams['dsource']
#     fintrinsic_region = SAPMparams['fintrinsic_region']
#     Mconn = SAPMparams['Mconn']
#     Minput = SAPMparams['Minput']
#     timepoint = SAPMparams['timepoint']
#     epoch = SAPMparams['epoch']
#     latent_flag = SAPMparams['latent_flag']
#     reciprocal_flag = SAPMparams['reciprocal_flag']
#     DBname = SAPMparams['DBname']
#     DBnum = SAPMparams['DBnum']
#
#     regular_flag = 1-latent_flag   # flag where connections are not latent
#
#     ntime, NP = np.shape(tplist_full)
#     Nintrinsics = vintrinsic_count + fintrinsic_count
#
#     ncomponents_to_fit = copy.deepcopy(nregions)
# #---------------------------------------------------------------------------------------------------------
#     #---------------------------------------------------------------------------------------------------------
#     # repeat the process for each participant-----------------------------------------------------------------
#     betalimit = 3.0
#     epochnum = 0
#     SAPMresults = []
#     first_pass_results = []
#     second_pass_results = []
#     beta_init_record = []
#     for nperson in range(NP):
#         if verbose:
#             print('starting person {} at {}'.format(nperson,time.ctime()))
#         tp = tplist_full[epochnum][nperson]['tp']
#         tsize_total = len(tp)
#         nruns = nruns_per_person[nperson]
#
#         # get tc data for each region/cluster
#         rnumlist = []
#         clustercount = np.cumsum(nclusterlist)
#         for aa in range(len(clusterlist)):
#             x = np.where(clusterlist[aa] < clustercount)[0]
#             rnumlist += [x[0]]
#
#         Sinput = []
#         for nc,cval in enumerate(clusterlist):
#             tc1 = tcdata_centered[cval, tp]
#             Sinput.append(tc1)
#         Sinput = np.array(Sinput)
#
#         Sinput_original = []
#         for nc, cval in enumerate(clusterlist):
#             tc1 = tcdata_centered_original[cval, tp]
#             Sinput_original.append(tc1)
#         Sinput_original = np.array(Sinput_original)
#
#         # get principal components of Sinput--------------------------
#         nr = np.shape(Sinput)[0]
#         pca = sklearn.decomposition.PCA()
#         pca.fit(Sinput)
#         components = pca.components_
#         loadings = pca.transform(Sinput)
#         mu2 = np.mean(Sinput, axis=0)
#         loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
#         components = np.concatenate((mu2[np.newaxis, :], components), axis=0)
#         # test_fit = loadings @ components
#
#         # setup fixed intrinsic based on the model paradigm
#         # need to account for timepoint and epoch....
#         if fintrinsic_count > 0:
#             if epoch >= tsize:
#                 et1 = 0
#                 et2 = tsize
#             else:
#                 if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
#                     et1 = (timepoint - np.floor(epoch / 2)).astype(int)
#                     et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#                 else:
#                     et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
#                     et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#             if et1 < 0: et1 = 0
#             if et2 > tsize: et2 = tsize
#             epoch = et2 - et1
#
#             ftemp = fintrinsic_base[0,et1:et2]
#             fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
#             # print('shape of fintrinsic1 is {}'.format(np.shape(fintrinsic1)))
#             if np.var(ftemp) > 1.0e-3:
#                 Sint = Sinput[fintrinsic_region,:]
#                 Sint = Sint - np.mean(Sint)
#                 # need to add constant to fit values
#                 G = np.concatenate((fintrinsic1[np.newaxis,:],np.ones((1,tsize_total))),axis=0)
#                 b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
#                 beta_int1 = b[0]
#             else:
#                 beta_int1 = 0.0
#         else:
#             beta_int1 = 0.0
#             fintrinsic1 = []
#
#         lastgood_beta_int1 = copy.deepcopy(beta_int1)
#
#         # initialize beta values-----------------------------------
#         nbeta = len(csource)
#         if isinstance(betascale,str):
#             if betascale == 'shotgun':
#                 beta_initial = betaval_init_shotgun(initial_Lweight, csource, ctarget, Sinput, Minput, Mconn, components,
#                                     loadings, fintrinsic_count, vintrinsic_count, np.ones(len(dtarget)), beta_int1, fintrinsic1,
#                                                     ncomponents_to_fit, nreps=10000)
#
#                 beta_initial = beta_initial[np.newaxis,:]
#                 nitermax_stage1 = 1
#             else:
#                 # read saved beta_initial values
#                 b = np.load(betascale,allow_pickle=True).flat[0]
#                 beta_initial = b['beta_initial']
#                 beta_initial = beta_initial[np.newaxis,:]
#                 nitermax_stage1 = 1
#             nsteps_stage1 = 1
#         else:
#             beta_initial = betascale*np.random.randn(nsteps_stage1,nbeta)
#             nregion,ntotal = np.shape(Minput)
#
#         # initialize deltavals
#         delta_initial = np.ones(len(dtarget))
#         deltascale = np.std(Sinput,axis=1)
#         meanscale = np.mean(deltascale)
#
#         # initialize
#         results_record = []
#         ssqd_record = []
#
#         # stage 1 - test the initial betaval settings
#         stage1_ssqd = np.zeros(nsteps_stage1)
#         stage1_slope = np.zeros(nsteps_stage1)
#         stage1_results = []
#         for ns in range(nsteps_stage1):
#             ssqd_record_stage1 = []
#             beta_init_record.append({'beta_initial':beta_initial[ns,:]})
#
#             # initalize Sconn
#             betavals = copy.deepcopy(beta_initial[ns,:])
#             lastgood_betavals = copy.deepcopy(betavals)
#             deltavals = copy.deepcopy(delta_initial)
#             lastgood_deltavals = copy.deepcopy(deltavals)
#
#             alphalist = initial_alpha*np.ones(nbeta)
#             alphabint = copy.deepcopy(initial_alpha)
#             alpha = copy.deepcopy(initial_alpha)
#             Lweight = copy.deepcopy(initial_Lweight)
#             dval = copy.deepcopy(initial_dval)
#
#             # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
#             Mconn[ctarget,csource] = copy.deepcopy(betavals)
#             Minput[dtarget, dsource] = copy.deepcopy(deltavals)
#             # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, ncomponents_to_fit)
#             # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
#             # ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv
#
#             fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                         vintrinsic_count, beta_int1, fintrinsic1)
#             ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
#                                                                        regular_flag)
#
#             ssqd_starting = copy.deepcopy(ssqd)
#             ssqd_starting0 = copy.deepcopy(ssqd)
#             ssqd_old = copy.deepcopy(ssqd)
#             ssqd_record += [ssqd]
#
#             iter = 0
#             converging = True
#             dssq_record = np.ones(3)
#             dssq_count = 0
#             sequence_count = 0
#             R2avg_record = []
#
#             while alpha > alpha_limit and iter < nitermax_stage1 and converging:
#                 iter += 1
#
#                 betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
#                                                     update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals,
#                                                     deltavals, betalimit, ctarget, csource, dtarget, dsource,
#                                                     dval,fintrinsic_count,
#                                                     vintrinsic_count, beta_int1,fintrinsic1, Lweight, regular_flag, alpha,alphabint,
#                                                     ncomponents_to_fit, latent_flag=latent_flag)  # kappavals, ktarget, ksource,
#
#                 ssqd_record_stage1 += [ssqd]
#
#                 if ssqd > ssqd_original:
#                     alpha *= 0.5
#                     alphabint *= 0.5
#                     betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
#                     deltavals = copy.deepcopy(lastgood_deltavals)
#                     beta_int1 = copy.deepcopy(lastgood_beta_int1)
#                 else:
#                     lastgood_betavals = copy.deepcopy(betavals)
#                     lastgood_deltavals = copy.deepcopy(deltavals)
#                     lastgood_beta_int1 = copy.deepcopy(beta_int1)
#
#                 Mconn[ctarget, csource] = copy.deepcopy(betavals)
#                 Minput[dtarget, dsource] = copy.deepcopy(deltavals)
#
#                 # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
#                 #                                     Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
#                 #                                     ncomponents_to_fit)
#                 # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
#                 # ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv
#
#                 fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                             vintrinsic_count, beta_int1, fintrinsic1)
#                 ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals,
#                                                                          deltavals,
#                                                                          regular_flag)
#
#
#                 err_total = Sinput - fit
#                 Smean = np.mean(Sinput)
#                 errmean = np.mean(err_total)
#
#                 R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
#                 R2avg = np.mean(R2list)
#                 R2avg_record += [R2avg]
#                 if len(R2avg_record) > 10:
#                     # R2avg_slope = (R2avg_record[-1] - R2avg_record[-5])/5.0
#                     N = 5
#                     y = np.array(R2avg_record[-N:])
#                     x = np.array(range(N))
#                     R2avg_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )
#                     if R2avg_slope < converging_slope_limit[0]:
#                         converging = False
#
#                 R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)
#
#                 results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
#
#                 ssqchange = ssqd - ssqd_original
#
#                 if verbose:
#                     print('SAPM  {} stage1 pass {} iter {} alpha {:.3e}  ssqd {:.2f} error {:.2f} error2 {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson,
#                                     ns, iter, alpha, ssqd, error, error2, ssqchange, 100.*ssqd/ssqd_starting0, R2avg, R2total))
#                 ssqd_old = copy.deepcopy(ssqd)
#                 # now repeat it ...
#             stage1_ssqd[ns] = ssqd
#             stage1_slope[ns] = R2avg_slope
#             stage1_results.append({'betavals':betavals, 'deltavals':deltavals})
#
#             if save_test_record and nperson == test_person:
#                 test_record.append({'stage':1, 'R2avg_record':R2avg_record})
#
#         # get the best betavals from stage1 so far ...
#         x = np.argmin(stage1_ssqd)
#         betavals = stage1_results[x]['betavals']
#         deltavals = stage1_results[x]['deltavals']
#
#         if save_test_record and nperson == test_person:
#             stage2_start = copy.deepcopy(x)
#
#         # stage 2
#         # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
#         if verbose: print('starting stage 2 ....')
#
#         # stage2_ssqd = np.zeros(nsteps_stage2)
#         # stage2_results = []
#         # for ns in range(nsteps_stage2):
#         #     ssqd_record_stage2 = []
#
#         lastgood_betavals = copy.deepcopy(betavals)
#         alpha = copy.deepcopy(initial_alpha)
#         alphabint = copy.deepcopy(initial_alpha)
#         Lweight = copy.deepcopy(initial_Lweight)
#         dval = copy.deepcopy(initial_dval)
#
#         # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
#         Mconn[ctarget, csource] = copy.deepcopy(betavals)
#         Minput[dtarget, dsource] = copy.deepcopy(deltavals)
#
#         # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
#         #                                         Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
#         #                                         ncomponents_to_fit)
#         # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
#         # ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv
#
#         fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                     vintrinsic_count, beta_int1, fintrinsic1)
#         ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
#                                                                  regular_flag)
#
#         ssqd_starting = copy.deepcopy(ssqd)
#         ssqd_old = copy.deepcopy(ssqd)
#         ssqd_record += [ssqd]
#
#         iter = 0
#         converging = True
#         dssq_record = np.ones(3)
#         dssq_count = 0
#         sequence_count = 0
#         R2avg_record = []
#
#         while alpha > alpha_limit and iter < nitermax_stage2 and converging:
#             iter += 1
#             betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
#                 update_betavals_V3(Sinput, components, loadings, Minput, Mconn, betavals, deltavals, betalimit,
#                                    ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
#                                    vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
#                                    ncomponents_to_fit, latent_flag=latent_flag)   #, kappavals, ktarget, ksource
#
#             ssqd_record_stage1 += [ssqd]
#
#             if ssqd > ssqd_original:
#                 alpha *= 0.5
#                 alphabint *= 0.5
#                 betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
#                 deltavals = copy.deepcopy(lastgood_deltavals)
#                 beta_int1 = copy.deepcopy(lastgood_beta_int1)
#                 sequence_count = 0
#             else:
#                 lastgood_betavals = copy.deepcopy(betavals)
#                 lastgood_deltavals = copy.deepcopy(deltavals)
#                 lastgood_beta_int1 = copy.deepcopy(beta_int1)
#                 sequence_count += 1
#                 if sequence_count > 2:
#                     alpha *= 1.3
#                     alphabint *= 1.3
#                     sequence_count = 0
#
#             Mconn[ctarget, csource] = copy.deepcopy(betavals)
#             Minput[dtarget, dsource] = copy.deepcopy(deltavals)
#
#             # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
#             #                                     Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
#             #                                     ncomponents_to_fit)
#             # # Soutput = Meigv @ Mintrinsic  # signalling over each connection
#             # ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight, betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv
#
#             fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                         vintrinsic_count, beta_int1, fintrinsic1)
#             ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
#                                                                      regular_flag)
#
#
#             err_total = Sinput - fit
#             Smean = np.mean(Sinput)
#             errmean = np.mean(err_total)
#
#             R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
#             R2avg = np.mean(R2list)
#             R2avg_record += [R2avg]
#             if len(R2avg_record) > 10:
#                 # R2avg_slope = (R2avg_record[-1] - R2avg_record[-5])/5.0
#                 N = 5
#                 y = np.array(R2avg_record[-N:])
#                 x = np.array(range(N))
#                 R2avg_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )
#                 if R2avg_slope < converging_slope_limit[1]:
#                     converging = False
#
#             R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)
#
#             results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
#
#             ssqchange = ssqd - ssqd_original
#
#             if verbose:
#                 print('SAPM  {} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} error {:.2f} error2 {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(
#                         nperson,iter, alpha, ssqd, error, error2, ssqchange, 100. * ssqd / ssqd_starting0, R2avg, R2total))
#             ssqd_old = copy.deepcopy(ssqd)
#             # now repeat it ...
#
#         if save_test_record and nperson == test_person:
#             test_record.append({'stage': 2, 'R2avg_record': R2avg_record, 'stage1_base':stage2_start})
#
#         # fit the results now to determine output signaling from each region
#         Mconn[ctarget, csource] = copy.deepcopy(betavals)
#         Minput[dtarget, dsource] = copy.deepcopy(deltavals)
#         # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings, Minput,
#         #                                     Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1,
#         #                                     ncomponents_to_fit)
#
#         fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                     vintrinsic_count, beta_int1, fintrinsic1)
#         W = []
#         loadings_fit = []
#         loadings = []
#         components = []
#
#         Sconn = Meigv @ Mintrinsic    # signalling over each connection
#
#         entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
#                  'fit':fit, 'loadings_fit':loadings_fit, 'W':W, 'loadings':loadings, 'components':components,
#                  'R2total':R2total, 'R2avg':R2avg, 'Mintrinsic':Mintrinsic, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
#                  'Meigv':Meigv, 'betavals':betavals, 'deltavals':deltavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
#                  'fintrinsic_base':fintrinsic_base, 'Sinput_original':Sinput_original, 'DBname':DBname, 'DBnum':DBnum}
#
#         SAPMresults.append(copy.deepcopy(entry))
#
#         stoptime = time.ctime()
#
#         if save_test_record and nperson == test_person:
#             np.save(test_record_name,test_record)
#
#     np.save(SAPMresultsname, SAPMresults)
#
#     if verbose:
#         print('finished SAPM at {}'.format(time.ctime()))
#         print('     started at {}'.format(starttime))
#         print('     results written to {}'.format(SAPMresultsname))
#     return SAPMresultsname



#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model1_V4(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [],
                    betascale = 0.1, alphascale = 0.01, Lweight = 0.01, nitermax_stage3 = 1200,
                    nitermax_stage2 = 300, nitermax_stage1 = 100, nsteps_stage2 = 4, nsteps_stage1 = 30,
                    levelthreshold = [1e-4, 1e-5, 1e-6], verbose = True, silentrunning = False, run_whole_group = False,
                    resumerun = False):

# this version fits to principal components of Sinput
#     save_test_record = False
    p,f = os.path.split(SAPMresultsname)
    f1,e = os.path.splitext(f)
    test_record_name = os.path.join(p,'gradient_descent_record.npy')
    betavals_savename = os.path.join(p,'betavals_' + f1[:20] + '.npy')
    test_record = []
    test_person = 0

    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    converging_slope_limit = levelthreshold
    initial_alpha = alphascale
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

    clusterlist = cnums_to_clusterlist(cnums,nclusterlist)

    nregion, ntotal = np.shape(Minput)
    regular_flag = 1-latent_flag   # flag where connections are not latent

    ntime, NP = np.shape(tplist_full)
    Nintrinsics = vintrinsic_count + fintrinsic_count

    ncomponents_to_fit = copy.deepcopy(nregions)
#---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 5.0
    epochnum = 0
    SAPMresults = []
    first_pass_results = []
    second_pass_results = []
    beta_init_record = []

    # -----------option for fitting to data from all participants at the same time----------
    if run_whole_group:
        NP_to_run = 1
    else:
        NP_to_run = copy.deepcopy(NP)
    #---------------------------------------------------------------------------------------
    #---------option for resuming previously interrupted run -------------------------------
    if resumerun:
        SAPMresults = list(np.load(SAPMresultsname, allow_pickle = True))
        NPstart = len(SAPMresults)
    else:
        NPstart = 0

    #---------------------------------------------------------------------------------------

    # initialize beta values-----------------------------------
    nbeta = len(csource)
    # half_nsteps1 = np.floor(nsteps_stage1/2.0).astype(int)
    if isinstance(betascale,str):
        b = np.load(betascale,allow_pickle=True).flat[0]
        beta_initial1 = copy.deepcopy(b['beta_initial'])
        # beta_temp = np.random.randn(half_nsteps1, nbeta)
        beta_initial_original = np.random.randn(nsteps_stage1, nbeta)
        # beta_initial_original = np.concatenate((beta_temp, -beta_temp), axis=0)
        betanorm = np.sqrt(np.sum(beta_initial_original ** 2, axis=1))
        betanorm = np.repeat(betanorm[:, np.newaxis], nbeta, axis=1)
        beta_range_default = 1.0
        beta_initial_original = beta_range_default * beta_initial_original / betanorm  # normalize the magnitude

    else:
        # beta_initial_original = betascale*np.random.randn(nsteps_stage1,nbeta)

        # beta_temp = np.random.randn(half_nsteps1, nbeta)
        beta_initial_original = np.random.randn(nsteps_stage1, nbeta)
        # beta_initial_original = np.concatenate((beta_temp, -beta_temp), axis=0)
        betanorm = np.sqrt(np.sum(beta_initial_original ** 2, axis=1))
        betanorm = np.repeat(betanorm[:, np.newaxis], nbeta, axis=1)
        beta_initial_original = betascale * beta_initial_original / betanorm  # normalize the magnitude

        # balance the starting points for gradient-decent to avoid the abyss
        # nsteps_half = np.floor(nsteps_stage1/2).astype(int)
        # for hh in range(nsteps_half):
        #     beta_initial_original[2*hh+1,:] = -1.0*beta_initial_original[2*hh,:]   # ensure that starting points are balanced

    # initialize deltavals
    # ndelta = len(dtarget)
    # delta_initial = np.ones(ndelta)
    # deltascale = np.std(Sinput,axis=1)
    # meanscale = np.mean(deltascale)

    stage2_monitoring_progress = []
    for nperson in range(NPstart, NP_to_run):
        if not silentrunning:
            if verbose:
                print('starting person {} at {}'.format(nperson,time.ctime()))
            else:
                # print('starting person {} at {}'.format(nperson,time.ctime()), end = '\r')
                print('.', end = '')

        if run_whole_group:
            tp = []
            for pcounter in range(NP):
                tp += tplist_full[epochnum][pcounter]['tp']
            nruns = np.sum(nruns_per_person)
        else:
            tp = tplist_full[epochnum][nperson]['tp']
            nruns = nruns_per_person[nperson]

        tsize_total = len(tp)

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
        # nr = np.shape(Sinput)[0]
        # pca = sklearn.decomposition.PCA()
        # pca.fit(Sinput)
        # components = pca.components_
        # loadings = pca.transform(Sinput)
        # mu2 = np.mean(Sinput, axis=0)
        # loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
        # components = np.concatenate((mu2[np.newaxis, :], components), axis=0)
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
            fintrinsic1 = np.array(list(ftemp) * nruns)
            # beta_int1 = 1.0

            try:
                Nfintrinsic = len(fintrinsic_region)
                Sint = np.mean(Sinput[fintrinsic_region, :], axis=0)
            except:
                Nfintrinsic = 1
                Sint = Sinput[fintrinsic_region, :]

            if np.var(ftemp) > 1.0e-3:
                G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
                b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
                beta_int1 = b[0]
            else:
                beta_int1 = 0.0

        else:
            beta_int1 = 0.0
            fintrinsic1 = []

        lastgood_beta_int1 = copy.deepcopy(beta_int1)
        #
        # # initialize beta values-----------------------------------
        nbeta = len(csource)
        if isinstance(betascale,str):
            b = np.load(betascale,allow_pickle=True).flat[0]
            beta_initial1 = copy.deepcopy(b['beta_initial'])
            if np.ndim(beta_initial1) > 1:
                if np.ndim(beta_initial1) > 2:  # beta_initial1 had to have been saved with multiple estimates x nbeta x NP
                    nest,nbetaest,npersonest = np.shape(beta_initial1)
                    beta_initial = copy.deepcopy(beta_initial_original)
                    beta_initial[:nest,:] = copy.deepcopy(beta_initial1[:,:,nperson])
                else:  # beta_initial1 had to have been saved as size nbeta x NP
                    beta_initial = copy.deepcopy(beta_initial_original)
                    beta_initial[0,:] = copy.deepcopy(beta_initial1[:,nperson])

            else: # beta_initial1 had to have been saved as size nbeta
                beta_initial = copy.deepcopy(beta_initial_original)
                beta_initial[0,:] = copy.deepcopy(beta_initial1)
        else:
            beta_initial = copy.deepcopy(beta_initial_original)

        # initialize deltavals
        ndelta = len(dtarget)
        delta_initial = np.ones(ndelta)
        deltascale = np.std(Sinput,axis=1)
        meanscale = np.mean(deltascale)

        # initialize
        results_record = []
        ssqd_record = []

        # stage 1 - test the initial betaval settings
        stage1_ssqd = np.zeros(nsteps_stage1)
        stage1_slope = np.zeros(nsteps_stage1)
        stage1_r2final = np.zeros(nsteps_stage1)
        stage1_ssqd_slope = np.zeros(nsteps_stage1)
        stage1_ssqd_final = np.zeros(nsteps_stage1)
        stage1_results = []
        for ns in range(nsteps_stage1):
            ssqd_record_stage1 = []
            beta_init_record.append({'beta_initial':beta_initial[ns,:]})

            # initalize Sconn
            betavals = copy.deepcopy(beta_initial[ns,:]) # initialize beta values
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

            fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                        vintrinsic_count, beta_int1, fintrinsic1)
            ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                     regular_flag)
            # add cost for correlated latents
            # lcc = np.corrcoef(Mintrinsic)
            # latent_cost = np.sum(
            #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
            # ssqd += 0.001*latent_cost

            ssqd_starting = copy.deepcopy(ssqd)
            ssqd_starting0 = copy.deepcopy(ssqd)
            ssqd_old = copy.deepcopy(ssqd)
            ssqd_record += [ssqd]

            iter = 0
            converging = True
            dssq_record = np.ones(3)
            dssq_count = 0
            sequence_count = 0
            R2avg_record = []
            R2avg_slope = 0.
            ssqd_slope = 0.

            while alpha > alpha_limit and iter < nitermax_stage1 and converging:
                iter += 1

                betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
                                                    update_betavals_V4(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
                                                    ctarget, csource, dtarget, dsource, dval,fintrinsic_count,
                                                    vintrinsic_count, beta_int1,fintrinsic1, Lweight, regular_flag, alpha,alphabint,
                                                    latent_flag=latent_flag)  # kappavals, ktarget, ksource,

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
                    if sequence_count > 4:
                        alpha = np.min([1.3*alpha, initial_alpha])
                        alphabint = np.min([1.3*alphabint, initial_alpha])
                        sequence_count = 0

                Mconn[ctarget, csource] = copy.deepcopy(betavals)
                Minput[dtarget, dsource] = copy.deepcopy(deltavals)

                fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                            vintrinsic_count, beta_int1, fintrinsic1)
                ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals,
                                                                         deltavals, regular_flag)
                # add cost for correlated latents
                # lcc = np.corrcoef(Mintrinsic)
                # latent_cost = np.sum(
                #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
                # ssqd += 0.001*latent_cost

                ssqd_record_stage1 += [ssqd]

                err_total = Sinput - fit
                Smean = np.mean(Sinput)
                errmean = np.mean(err_total)

                R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
                R2avg = np.mean(R2list)
                R2avg_record += [R2avg]
                if len(R2avg_record) > 10:
                    N = 5
                    y = np.array(R2avg_record[-N:])
                    x = np.array(range(N))
                    R2avg_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )

                    y = np.array(ssqd_record_stage1[-N:])
                    x = np.array(range(N))
                    ssqd_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )
                    if ssqd_slope > -converging_slope_limit[0]:
                        converging = False

                R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

                results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

                ssqchange = ssqd - ssqd_original

                if verbose and not silentrunning:
                    betamag = np.sqrt(np.sum(betavals**2))
                    print('SAPM  {} stage1 pass {} iter {} alpha {:.3e}  error+cost {:.3f} error {:.3f} L1 cost {:.3f} change {:.3e}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}  beta mag. {:.3f}'.format(nperson,
                                    ns, iter, alpha, ssqd, error, error2, ssqchange, 100.*ssqd/ssqd_starting0, R2avg, R2total, betamag))
                ssqd_old = copy.deepcopy(ssqd)
                # now repeat it ...
            stage1_ssqd[ns] = ssqd
            stage1_slope[ns] = R2avg_slope
            stage1_r2final[ns] = R2avg_record[-1]

            stage1_ssqd_slope[ns] = ssqd_slope
            stage1_ssqd_final[ns] = ssqd_record_stage1[-1]
            stage1_results.append({'betavals':betavals, 'deltavals':deltavals})

            # if save_test_record and nperson == test_person:
            #     test_record.append({'stage':1, 'R2avg_record':R2avg_record})


        # get the best nsteps_stage2 betavals from stage1 so far ...
        #--------end of stage 1, finding the trajectories to continue on...
        # ...based on lowest ssqd, highest slope, some combination of these ...
        projected_r2 = stage1_r2final + stage1_slope*50.0   # project 50 iterations forward with the final slope
        projected_ssqd = stage1_ssqd_final + stage1_ssqd_slope*50.0   # project 50 iterations forward with the final slope

        nset = np.ceil(nsteps_stage2/2.0).astype(int)
        xd = np.argsort(-stage1_ssqd_slope)  # want the highest values
        xs1 = xd[-nset:]
        xsearch = np.array([a for a in range(nsteps_stage1) if a not in xs1]).astype(int)
        xs2 = np.argsort(stage1_ssqd[xsearch])  # want the lowest values that are not already included
        x = np.concatenate((xsearch[xs2[:nset]], xs1),axis=0)

        beta_initial2 = np.zeros((nset*2, nbeta))
        delta_initial2 = np.zeros((nset*2, ndelta))
        stage2_start = np.zeros(nset*2)
        for ns in range(len(x)):
            betavals1 = stage1_results[x[ns]]['betavals']
            deltavals1 = stage1_results[x[ns]]['deltavals']
            beta_initial2[ns,:] = copy.deepcopy(betavals1)
            delta_initial2[ns,:] = copy.deepcopy(deltavals1)

            # if save_test_record and nperson == test_person:
            #     stage2_start[ns] = copy.deepcopy(x[ns])

        # test alternatives for latent input DB values - added May 3, 2025
        check_for_alternative_DBvals = True
        if check_for_alternative_DBvals:
            print('checking for alternative optimal betavals ...')
            beta_initial2, delta_initial2 = check_alternative_latent_DBvals(Sinput, Minput, Mconn, Mintrinsic,
                                fintrinsic_count, beta_initial2, delta_initial2, ctarget, csource, dtarget, dsource, silentrunning = silentrunning)
            nsteps_stage2_temp, ndb = np.shape(beta_initial2)
            print('Number of steps for stage 2 is now {}'.format(nsteps_stage2_temp))
        else:
            nsteps_stage2_temp = nsteps_stage2

        # start of stage 2
        # initialize
        results_record = []
        ssqd_record = []

        # stage 2 - test the initial betaval settings
        stage2_ssqd = np.zeros(nsteps_stage2_temp)
        stage2_slope = np.zeros(nsteps_stage2_temp)
        stage2_r2final = np.zeros(nsteps_stage2_temp)
        stage2_ssqd_slope = np.zeros(nsteps_stage2_temp)
        stage2_ssqd_final = np.zeros(nsteps_stage2_temp)
        stage2_results = []
        for ns in range(nsteps_stage2_temp):
            ssqd_record_stage2 = []
            beta_init_record.append({'beta_initial':beta_initial2[ns,:]})

            # initalize Sconn
            betavals = copy.deepcopy(beta_initial2[ns,:]) # initialize beta values at zero
            lastgood_betavals = copy.deepcopy(betavals)
            deltavals = copy.deepcopy(delta_initial2[ns,:])
            lastgood_deltavals = copy.deepcopy(deltavals)

            alphalist = initial_alpha*np.ones(nbeta)
            alphabint = copy.deepcopy(initial_alpha)
            alpha = copy.deepcopy(initial_alpha)
            Lweight = copy.deepcopy(initial_Lweight)
            dval = copy.deepcopy(initial_dval)

            # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
            Mconn[ctarget,csource] = copy.deepcopy(betavals)
            Minput[dtarget, dsource] = copy.deepcopy(deltavals)

            fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                        vintrinsic_count, beta_int1, fintrinsic1)
            ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                     regular_flag)
            # add cost for correlated latents
            # lcc = np.corrcoef(Mintrinsic)
            # latent_cost = np.sum(
            #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
            # ssqd += 0.001*latent_cost

            ssqd_starting = copy.deepcopy(ssqd)
            ssqd_old = copy.deepcopy(ssqd)
            ssqd_record += [ssqd]

            iter = 0
            converging = True
            dssq_record = np.ones(3)
            dssq_count = 0
            sequence_count = 0
            R2avg_record = []
            R2avg_slope = 0.
            ssqd_slope = 0.

            while alpha > alpha_limit and iter < nitermax_stage2 and converging:
                iter += 1

                betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
                                                    update_betavals_V4(Sinput, Minput, Mconn, betavals,
                                                    deltavals, betalimit, ctarget, csource, dtarget, dsource,
                                                    dval,fintrinsic_count,
                                                    vintrinsic_count, beta_int1,fintrinsic1, Lweight, regular_flag, alpha,alphabint,
                                                    latent_flag=latent_flag)  # kappavals, ktarget, ksource,

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
                    if sequence_count > 4:
                        alpha = np.min([1.3*alpha, initial_alpha])
                        alphabint = np.min([1.3*alphabint, initial_alpha])
                        sequence_count = 0

                Mconn[ctarget, csource] = copy.deepcopy(betavals)
                Minput[dtarget, dsource] = copy.deepcopy(deltavals)

                fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                            vintrinsic_count, beta_int1, fintrinsic1)
                ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals,
                                                                         deltavals, regular_flag)
                # add cost for correlated latents
                # lcc = np.corrcoef(Mintrinsic)
                # latent_cost = np.sum(
                #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
                # ssqd += 0.001*latent_cost

                ssqd_record_stage2 += [ssqd]

                err_total = Sinput - fit
                Smean = np.mean(Sinput)
                errmean = np.mean(err_total)

                R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
                R2avg = np.mean(R2list)
                R2avg_record += [R2avg]
                if len(R2avg_record) > 10:
                    N = 5
                    y = np.array(R2avg_record[-N:])
                    x = np.array(range(N))
                    R2avg_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )

                    y = np.array(ssqd_record_stage2[-N:])
                    x = np.array(range(N))
                    ssqd_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )
                    if ssqd_slope > -converging_slope_limit[1]:
                        converging = False

                R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

                results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

                ssqchange = ssqd - ssqd_original

                if verbose and not silentrunning:
                    betamag = np.sqrt(np.sum(betavals**2))
                    print('SAPM  {} stage2 pass {} iter {} alpha {:.3e}  error+cost {:.3f} error {:.3f} L1 cost {:.3f} change {:.3e}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}  beta mag. {:.3f}'.format(nperson,
                                    ns, iter, alpha, ssqd, error, error2, ssqchange, 100.*ssqd/ssqd_starting0, R2avg, R2total, betamag))
                ssqd_old = copy.deepcopy(ssqd)
                # now repeat it ...
            stage2_ssqd[ns] = ssqd
            stage2_slope[ns] = R2avg_slope
            stage2_r2final[ns] = R2avg_record[-1]
            stage2_ssqd_slope[ns] = ssqd_slope
            stage2_ssqd_final[ns] = ssqd_record_stage2[-1]
            stage2_results.append({'betavals':betavals, 'deltavals':deltavals, 'ssqd':ssqd, 'R2avg':R2avg_record})

            # if save_test_record and nperson == test_person:
            #     test_record.append({'stage':2, 'R2avg_record':R2avg_record, 'stage2_results':stage2_results})

        # end of stage2
        # get the best betavals from stage2 so far ...
        x = np.argmin(stage2_ssqd)
        betavals = stage2_results[x]['betavals']
        deltavals = stage2_results[x]['deltavals']

        stage2_monitoring_progress.append({'results':stage2_results})

        # if save_test_record and nperson == test_person:
        #     stage3_start = copy.deepcopy(x)

        # stage 3
        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        if verbose and not silentrunning: print('starting stage 3 ....')

        lastgood_betavals = copy.deepcopy(betavals)
        alpha = copy.deepcopy(initial_alpha)
        alphabint = copy.deepcopy(initial_alpha)
        Lweight = copy.deepcopy(initial_Lweight)
        dval = copy.deepcopy(initial_dval)

        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqd += 0.001*latent_cost

        ssqd_starting = copy.deepcopy(ssqd)
        ssqd_old = copy.deepcopy(ssqd)
        ssqd_record += [ssqd]

        iter = 0
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        sequence_count = 0
        R2avg_record = []
        ssqd_record_stage3 = []
        R2avg_slope = 0.
        ssqd_slope = 0.

        while alpha > alpha_limit and iter < nitermax_stage3 and converging:
            iter += 1

            betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
                update_betavals_V4(Sinput, Minput, Mconn, betavals, deltavals, betalimit, ctarget, csource, dtarget, dsource,
                                   dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag,
                                   alpha, alphabint, latent_flag=latent_flag)  # kappavals, ktarget, ksource,

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
                if sequence_count > 4:
                    alpha = np.min([1.3*alpha, initial_alpha])
                    alphabint = np.min([1.3*alphabint, initial_alpha])
                    sequence_count = 0

            Mconn[ctarget, csource] = copy.deepcopy(betavals)
            Minput[dtarget, dsource] = copy.deepcopy(deltavals)

            fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                        vintrinsic_count, beta_int1, fintrinsic1)
            ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                     regular_flag)
            # add cost for correlated latents
            # lcc = np.corrcoef(Mintrinsic)
            # latent_cost = np.sum(
            #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
            # ssqd += 0.001*latent_cost

            ssqd_record_stage3 += [ssqd]

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)

            R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
            R2avg = np.mean(R2list)
            R2avg_record += [R2avg]
            if len(R2avg_record) > 10:
                N = 5
                y = np.array(R2avg_record[-N:])
                x = np.array(range(N))
                R2avg_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )

                y = np.array(ssqd_record_stage3[-N:])
                x = np.array(range(N))
                ssqd_slope = (N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x*x) - np.sum(x)*np.sum(x) )
                if ssqd_slope > -converging_slope_limit[2]:
                    converging = False

            R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

            results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

            ssqchange = ssqd - ssqd_original

            if verbose and not silentrunning:
                betamag = np.sqrt(np.sum(betavals**2))
                print('SAPM  {} stage 3  beta vals:  iter {} alpha {:.3e}  error+cost {:.3f} error {:.3f} L1 cost {:.3f} change {:.3e}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}  beta mag. {:.3f}'.format(
                        nperson,iter, alpha, ssqd, error, error2, ssqchange, 100. * ssqd / ssqd_starting0, R2avg, R2total, betamag))
            ssqd_old = copy.deepcopy(ssqd)
            # now repeat it ...

        # if save_test_record and nperson == test_person:
        #     test_record.append({'stage': 3, 'R2avg_record': R2avg_record, 'stage3_base':stage3_start})

        # fit the results now to determine output signaling from each region
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)

        W = []
        loadings_fit = []

        # loadings = []
        # components = []

        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'fit':fit, 'loadings_fit':loadings_fit, 'W':W, 'loadings':loadings, 'components':components,
                 'R2total':R2total, 'R2avg':R2avg, 'Mintrinsic':Mintrinsic, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
                 'Meigv':Meigv, 'betavals':betavals, 'deltavals':deltavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
                 'cnums':cnums, 'fintrinsic_base':fintrinsic_base, 'Sinput_original':Sinput_original, 'DBname':DBname, 'DBnum':DBnum}

        SAPMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()

        # if save_test_record and nperson == test_person:
        #     np.save(test_record_name,test_record)

        # change Dec 2024 - save results after each individual data set
        np.save(SAPMresultsname, SAPMresults)

    # save betavals for use as starting point for other fits
    nb = len(betavals)
    betarecord = np.zeros((nb,NP_to_run))
    for nperson in range(NP_to_run):
        betarecord[:,nperson] = SAPMresults[nperson]['betavals']
    betainit = np.mean(betarecord, axis = 1)
    b = {'beta_initial':betainit}
    np.save(betavals_savename, b)

    # write out stage2 results for checking on performance
    p,f1 = os.path.split(SAPMresultsname)
    f,e = os.path.splitext(f1)
    stage2name = os.path.join(p,f[:15] + '_stage2.npy')
    np.save(stage2name, stage2_monitoring_progress)

    if verbose and not silentrunning:
        print('record of average betavals saved to {}'.format(betavals_savename))
        print('finished SAPM at {}'.format(time.ctime()))
        print('     started at {}'.format(starttime))
        print('     results written to {}'.format(SAPMresultsname))
    return SAPMresultsname



#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model1_V5(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [],
                    betascale = 0.1, alphascale = 0.01, Lweight = 0.01, nitermax_stage3 = 1200,
                    nitermax_stage2 = 300, nitermax_stage1 = 100, nsteps_stage2 = 4, nsteps_stage1 = 30,
                    levelthreshold = [1e-4, 1e-5, 1e-6], verbose = True, silentrunning = False, run_whole_group = False,
                    resumerun = False):

    p,f = os.path.split(SAPMresultsname)
    f1,e = os.path.splitext(f)
    test_record_name = os.path.join(p,'gradient_descent_record.npy')
    betavals_savename = os.path.join(p,'betavals_' + f1[:20] + '.npy')
    test_record = []
    test_person = 0

    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    converging_slope_limit = levelthreshold
    initial_alpha = alphascale
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

    clusterlist = cnums_to_clusterlist(cnums,nclusterlist)

    nregion, ntotal = np.shape(Minput)
    regular_flag = 1-latent_flag   # flag where connections are not latent

    ntime, NP = np.shape(tplist_full)
    Nintrinsics = vintrinsic_count + fintrinsic_count

    ncomponents_to_fit = copy.deepcopy(nregions)
    #---------------------------------------------------------------------------------------------------------
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 5.0
    epochnum = 0
    SAPMresults = []
    first_pass_results = []
    second_pass_results = []
    beta_init_record = []

    # -----------option for fitting to data from all participants at the same time----------
    if run_whole_group:
        NP_to_run = 1
    else:
        NP_to_run = copy.deepcopy(NP)
    #---------------------------------------------------------------------------------------
    #---------option for resuming previously interrupted run -------------------------------
    if resumerun:
        SAPMresults = list(np.load(SAPMresultsname, allow_pickle = True))
        NPstart = len(SAPMresults)
    else:
        NPstart = 0
    #---------------------------------------------------------------------------------------

    # initialize beta values-----------------------------------
    nbeta = len(csource)
    # half_nsteps1 = np.floor(nsteps_stage1/2.0).astype(int)
    if isinstance(betascale,str):
        b = np.load(betascale,allow_pickle=True).flat[0]
        beta_initial1 = copy.deepcopy(b['beta_initial'])
        beta_initial_original = np.random.randn(nsteps_stage1, nbeta)
        betanorm = np.sqrt(np.sum(beta_initial_original ** 2, axis=1))
        betanorm = np.repeat(betanorm[:, np.newaxis], nbeta, axis=1)
        beta_range_default = 1.0
        beta_initial_original = beta_range_default * beta_initial_original / betanorm  # normalize the magnitude

    else:
        beta_initial_original = np.random.randn(nsteps_stage1, nbeta)
        betanorm = np.sqrt(np.sum(beta_initial_original ** 2, axis=1))
        betanorm = np.repeat(betanorm[:, np.newaxis], nbeta, axis=1)
        beta_initial_original = betascale * beta_initial_original / betanorm  # normalize the magnitude

    stage2_monitoring_progress = []
    for nperson in range(NPstart, NP_to_run):
        if not silentrunning:
            if verbose:
                print('starting person {} at {}'.format(nperson,time.ctime()))
            else:
                print('.', end = '')

        if run_whole_group:
            tp = []
            for pcounter in range(NP):
                tp += tplist_full[epochnum][pcounter]['tp']
            nruns = np.sum(nruns_per_person)
        else:
            tp = tplist_full[epochnum][nperson]['tp']
            nruns = nruns_per_person[nperson]

        tsize_total = len(tp)

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
        # nr = np.shape(Sinput)[0]
        # pca = sklearn.decomposition.PCA()
        # pca.fit(Sinput)
        # components = pca.components_
        # loadings = pca.transform(Sinput)
        # mu2 = np.mean(Sinput, axis=0)
        # loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
        # components = np.concatenate((mu2[np.newaxis, :], components), axis=0)
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
            fintrinsic1 = np.array(list(ftemp) * nruns)

            try:
                Nfintrinsic = len(fintrinsic_region)
                Sint = np.mean(Sinput[fintrinsic_region, :], axis=0)
            except:
                Nfintrinsic = 1
                Sint = Sinput[fintrinsic_region, :]

            if np.var(ftemp) > 1.0e-3:
                G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
                b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
                beta_int = b[0]*np.ones(nsteps_stage1)
            else:
                beta_int = np.zeros(nsteps_stage1)

        else:
            beta_int = np.zeros(nsteps_stage1)
            fintrinsic1 = []

        lastgood_beta_int1 = copy.deepcopy(beta_int[0])
        #
        # # initialize beta values-----------------------------------
        nbeta = len(csource)
        if isinstance(betascale,str):
            b = np.load(betascale,allow_pickle=True).flat[0]
            beta_initial1 = copy.deepcopy(b['beta_initial'])
            if np.ndim(beta_initial1) > 1:
                if np.ndim(beta_initial1) > 2:  # beta_initial1 had to have been saved with multiple estimates x nbeta x NP
                    nest,nbetaest,npersonest = np.shape(beta_initial1)
                    beta_initial = copy.deepcopy(beta_initial_original)
                    beta_initial[:nest,:] = copy.deepcopy(beta_initial1[:,:,nperson])
                else:  # beta_initial1 had to have been saved as size nbeta x NP
                    beta_initial = copy.deepcopy(beta_initial_original)
                    beta_initial[0,:] = copy.deepcopy(beta_initial1[:,nperson])

            else: # beta_initial1 had to have been saved as size nbeta
                beta_initial = copy.deepcopy(beta_initial_original)
                beta_initial[0,:] = copy.deepcopy(beta_initial1)
        else:
            beta_initial = copy.deepcopy(beta_initial_original)

        # initialize deltavals
        ndelta = len(dtarget)
        delta_initial = np.ones((nsteps_stage1, ndelta))
        deltascale = np.std(Sinput,axis=1)
        meanscale = np.mean(deltascale)

        # run stage 1
        results_record, stage1_ssqd, stage1_slope, stage1_r2final, stage1_ssqd_slope, stage1_ssqd_final, stage1_results = \
                    run_sapm_stage_V2(nperson,1, nsteps_stage1, beta_initial, delta_initial, initial_alpha, initial_Lweight,
                            Sinput, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, beta_int, fintrinsic1,
                            fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage1,
                                   converging_slope_limit[0], verbose = verbose, silentrunning = silentrunning)

        # get the best nsteps_stage2 betavals from stage1 so far ...
        #--------end of stage 1, finding the trajectories to continue on...
        # ...based on lowest ssqd, highest slope, some combination of these ...
        projected_r2 = stage1_r2final + stage1_slope*50.0   # project 50 iterations forward with the final slope
        projected_ssqd = stage1_ssqd_final + stage1_ssqd_slope*50.0   # project 50 iterations forward with the final slope

        nset = np.ceil(nsteps_stage2/2.0).astype(int)
        xd = np.argsort(-stage1_ssqd_slope)  # want the highest values
        xs1 = xd[-nset:]
        xsearch = np.array([a for a in range(nsteps_stage1) if a not in xs1]).astype(int)
        xs2 = np.argsort(stage1_ssqd[xsearch])  # want the lowest values that are not already included
        x = np.concatenate((xsearch[xs2[:nset]], xs1),axis=0)

        beta_initial2 = np.zeros((nset*2, nbeta))
        delta_initial2 = np.zeros((nset*2, ndelta))
        beta_int = np.zeros(nset*2)
        stage2_start = np.zeros(nset*2)
        for ns in range(len(x)):
            betavals1 = stage1_results[x[ns]]['betavals']
            deltavals1 = stage1_results[x[ns]]['deltavals']
            beta_int1 = stage1_results[x[ns]]['beta_int1']
            beta_initial2[ns,:] = copy.deepcopy(betavals1)
            delta_initial2[ns,:] = copy.deepcopy(deltavals1)
            beta_int[ns] = copy.deepcopy(beta_int1)

        # test alternatives for latent input DB values - added May 3, 2025
        check_for_alternative_DBvals = True
        if check_for_alternative_DBvals:
            if not silentrunning:
                print('checking for alternative optimal betavals ...')
            Mintrinsic = copy.deepcopy(stage1_results[-1]['Mintrinsic'])
            beta_initial2, delta_initial2 = check_alternative_latent_DBvals(Sinput, Minput, Mconn, Mintrinsic,
                                fintrinsic_count, beta_initial2, delta_initial2, ctarget, csource, dtarget, dsource, silentrunning = silentrunning)
            nsteps_stage2_temp, ndb = np.shape(beta_initial2)
            if not silentrunning:
                print('Number of steps for stage 2 is now {}'.format(nsteps_stage2_temp))
        else:
            nsteps_stage2_temp = nsteps_stage2

        # run stage 2---------------------------------------------------
        results_record, stage2_ssqd, stage2_slope, stage2_r2final, stage2_ssqd_slope, stage2_ssqd_final, stage2_results = \
                    run_sapm_stage_V2(nperson,2, nsteps_stage2, beta_initial2, delta_initial2, initial_alpha, initial_Lweight,
                            Sinput, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, beta_int, fintrinsic1,
                            fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage2,
                                   converging_slope_limit[1], verbose = verbose, silentrunning = silentrunning)

        # end of stage2
        # get the best betavals from stage2 so far ...
        x = np.argmin(stage2_ssqd)
        betavals = stage2_results[x]['betavals']
        deltavals = stage2_results[x]['deltavals']
        beta_int = stage2_results[x]['beta_int1']

        stage2_monitoring_progress.append({'results':stage2_results})

        # run stage 3-------------------------------------------------------------
        results_record, stage3_ssqd, stage3_slope, stage3_r2final, stage3_ssqd_slope, stage3_ssqd_final, stage3_results = \
                    run_sapm_stage_V2(nperson,3, 1, betavals[np.newaxis,:], deltavals[np.newaxis,:], initial_alpha, initial_Lweight,
                            Sinput, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, [beta_int], fintrinsic1,
                            fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage3,
                                   converging_slope_limit[2], verbose = verbose, silentrunning = silentrunning)

        # fit the results now to determine output signaling from each region
        betavals = copy.deepcopy(stage3_results[-1]['betavals'])
        deltavals = copy.deepcopy(stage3_results[-1]['deltavals'])
        Mintrinsic = copy.deepcopy(stage3_results[-1]['Mintrinsic'])

        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)

        R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / (np.sum(Sinput ** 2, axis=1) + 1.0e-6)
        R2avg = np.mean(R2list)
        R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

        # W = []
        # loadings_fit = []
        # loadings = []
        # components = []
        # 'loadings_fit': loadings_fit, 'W': W, 'loadings': loadings, 'components': components

        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'fit':fit,
                 'R2total':R2total, 'R2avg':R2avg, 'Mintrinsic':Mintrinsic, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
                 'Meigv':Meigv, 'betavals':betavals, 'deltavals':deltavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
                 'cnums':cnums, 'fintrinsic_base':fintrinsic_base, 'Sinput_original':Sinput_original, 'DBname':DBname, 'DBnum':DBnum}

        SAPMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()
        np.save(SAPMresultsname, SAPMresults)

    # save betavals for use as starting point for other fits
    nb = len(betavals)
    betarecord = np.zeros((nb,NP_to_run))
    for nperson in range(NP_to_run):
        betarecord[:,nperson] = SAPMresults[nperson]['betavals']
    betainit = np.mean(betarecord, axis = 1)
    b = {'beta_initial':betainit}
    np.save(betavals_savename, b)

    # write out stage2 results for checking on performance
    p,f1 = os.path.split(SAPMresultsname)
    f,e = os.path.splitext(f1)
    stage2name = os.path.join(p,f[:15] + '_stage2.npy')
    np.save(stage2name, stage2_monitoring_progress)

    if verbose and not silentrunning:
        print('record of average betavals saved to {}'.format(betavals_savename))
        print('finished SAPM at {}'.format(time.ctime()))
        print('     started at {}'.format(starttime))
        print('     results written to {}'.format(SAPMresultsname))
    return SAPMresultsname



#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model1_pca(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [],
                    betascale = 0.1, alphascale = 0.01, Lweight = 0.01, nitermax_stage3 = 1200,
                    nitermax_stage2 = 300, nitermax_stage1 = 100, nsteps_stage2 = 4, nsteps_stage1 = 30,
                    levelthreshold = [1e-4, 1e-5, 1e-6], npc = 20, verbose = True, silentrunning = False, run_whole_group = False,
                    resumerun = False):

# this version fits to principal components of Sinput
#     save_test_record = False
    p,f = os.path.split(SAPMresultsname)
    f1,e = os.path.splitext(f)
    test_record_name = os.path.join(p,'gradient_descent_record.npy')
    betavals_savename = os.path.join(p,'betavals_' + f1[:20] + '.npy')
    test_record = []
    test_person = 0

    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    converging_slope_limit = levelthreshold
    initial_alpha = alphascale
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

    clusterlist = cnums_to_clusterlist(cnums,nclusterlist)

    nregion, ntotal = np.shape(Minput)
    regular_flag = 1-latent_flag   # flag where connections are not latent

    ntime, NP = np.shape(tplist_full)
    Nintrinsics = vintrinsic_count + fintrinsic_count

    ncomponents_to_fit = copy.deepcopy(nregions)
    #---------------------------------------------------------------------------------------------------------
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 5.0
    epochnum = 0
    SAPMresults = []
    first_pass_results = []
    second_pass_results = []
    beta_init_record = []

    # -----------option for fitting to data from all participants at the same time----------
    if run_whole_group:
        NP_to_run = 1
    else:
        NP_to_run = copy.deepcopy(NP)
    #---------------------------------------------------------------------------------------
    #---------option for resuming previously interrupted run -------------------------------
    if resumerun:
        SAPMresults = list(np.load(SAPMresultsname, allow_pickle = True))
        NPstart = len(SAPMresults)
    else:
        NPstart = 0
    #---------------------------------------------------------------------------------------

    # initialize beta values-----------------------------------
    nbeta = len(csource)
    # half_nsteps1 = np.floor(nsteps_stage1/2.0).astype(int)
    if isinstance(betascale,str):
        b = np.load(betascale,allow_pickle=True).flat[0]
        beta_initial1 = copy.deepcopy(b['beta_initial'])
        beta_initial_original = np.random.randn(nsteps_stage1, nbeta)
        betanorm = np.sqrt(np.sum(beta_initial_original ** 2, axis=1))
        betanorm = np.repeat(betanorm[:, np.newaxis], nbeta, axis=1)
        beta_range_default = 1.0
        beta_initial_original = beta_range_default * beta_initial_original / betanorm  # normalize the magnitude

    else:
        beta_initial_original = np.random.randn(nsteps_stage1, nbeta)
        betanorm = np.sqrt(np.sum(beta_initial_original ** 2, axis=1))
        betanorm = np.repeat(betanorm[:, np.newaxis], nbeta, axis=1)
        beta_initial_original = betascale * beta_initial_original / betanorm  # normalize the magnitude

    stage2_monitoring_progress = []
    for nperson in range(NPstart, NP_to_run):
        if not silentrunning:
            if verbose:
                print('starting person {} at {}'.format(nperson,time.ctime()))
            else:
                print('.', end = '')

        if run_whole_group:
            tp = []
            for pcounter in range(NP):
                tp += tplist_full[epochnum][pcounter]['tp']
            nruns = np.sum(nruns_per_person)
        else:
            tp = tplist_full[epochnum][nperson]['tp']
            nruns = nruns_per_person[nperson]

        tsize_total = len(tp)

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
            fintrinsic1 = np.array(list(ftemp) * nruns)

            try:
                Nfintrinsic = len(fintrinsic_region)
                Sint = np.mean(Sinput[fintrinsic_region, :], axis=0)
            except:
                Nfintrinsic = 1
                Sint = Sinput[fintrinsic_region, :]

            if np.var(ftemp) > 1.0e-3:
                G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
                b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
                beta_int = b[0]*np.ones(nsteps_stage1)
            else:
                beta_int = np.zeros(nsteps_stage1)

        else:
            beta_int = np.zeros(nsteps_stage1)
            fintrinsic1 = []

        lastgood_beta_int1 = copy.deepcopy(beta_int[0])
        #
        # # initialize beta values-----------------------------------
        nbeta = len(csource)
        if isinstance(betascale,str):
            b = np.load(betascale,allow_pickle=True).flat[0]
            beta_initial1 = copy.deepcopy(b['beta_initial'])
            if np.ndim(beta_initial1) > 1:
                if np.ndim(beta_initial1) > 2:  # beta_initial1 had to have been saved with multiple estimates x nbeta x NP
                    nest,nbetaest,npersonest = np.shape(beta_initial1)
                    beta_initial = copy.deepcopy(beta_initial_original)
                    beta_initial[:nest,:] = copy.deepcopy(beta_initial1[:,:,nperson])
                else:  # beta_initial1 had to have been saved as size nbeta x NP
                    beta_initial = copy.deepcopy(beta_initial_original)
                    beta_initial[0,:] = copy.deepcopy(beta_initial1[:,nperson])

            else: # beta_initial1 had to have been saved as size nbeta
                beta_initial = copy.deepcopy(beta_initial_original)
                beta_initial[0,:] = copy.deepcopy(beta_initial1)
        else:
            beta_initial = copy.deepcopy(beta_initial_original)

        #---------------------------------------------------------------------
        # get principal components of Sinput----------------------------------
        components, loadings = get_data_components(Sinput, fixed_component=fintrinsic1)
        nr,ncomp = np.shape(loadings)
        # normalize the loadings instead of the components
        for nn in range(ncomp):
            maxload = np.max(np.abs(loadings[:, nn]))
            loadings[:, nn] /= maxload
            components[nn, :] *= maxload
        if fintrinsic_count > 0:
            fintrinsic1 = copy.deepcopy(components[1, :])

        if npc < ncomp:
            ncomp = copy.deepcopy(npc)
            loadings = loadings[:,:ncomp]
            components = components[:ncomp,:]

        if fintrinsic_count > 0:
            fintrinsic_pc = np.zeros(ncomp)
            fintrinsic_pc[1] = 1.0
        else:
            fintrinsic_pc = []
        #---------------------------------------------------------------------
        #---------------------------------------------------------------------


        # initialize deltavals
        ndelta = len(dtarget)
        delta_initial = np.ones((nsteps_stage1, ndelta))
        deltascale = np.std(Sinput,axis=1)
        meanscale = np.mean(deltascale)

        # run stage 1
        results_record, stage1_ssqd, stage1_slope, stage1_r2final, stage1_ssqd_slope, stage1_ssqd_final, stage1_results = \
                    run_sapm_stage_V2(nperson,1, nsteps_stage1, beta_initial, delta_initial, initial_alpha, initial_Lweight,
                            loadings, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, beta_int, fintrinsic_pc,
                            fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage1,
                                   converging_slope_limit[0], verbose = verbose, silentrunning = silentrunning)

        # get the best nsteps_stage2 betavals from stage1 so far ...
        #--------end of stage 1, finding the trajectories to continue on...
        # ...based on lowest ssqd, highest slope, some combination of these ...
        projected_r2 = stage1_r2final + stage1_slope*50.0   # project 50 iterations forward with the final slope
        projected_ssqd = stage1_ssqd_final + stage1_ssqd_slope*50.0   # project 50 iterations forward with the final slope

        nset = np.ceil(nsteps_stage2/2.0).astype(int)
        xd = np.argsort(-stage1_ssqd_slope)  # want the highest values
        xs1 = xd[-nset:]
        xsearch = np.array([a for a in range(nsteps_stage1) if a not in xs1]).astype(int)
        xs2 = np.argsort(stage1_ssqd[xsearch])  # want the lowest values that are not already included
        x = np.concatenate((xsearch[xs2[:nset]], xs1),axis=0)

        beta_initial2 = np.zeros((nset*2, nbeta))
        delta_initial2 = np.zeros((nset*2, ndelta))
        beta_int = np.zeros(nset*2)
        stage2_start = np.zeros(nset*2)
        for ns in range(len(x)):
            betavals1 = stage1_results[x[ns]]['betavals']
            deltavals1 = stage1_results[x[ns]]['deltavals']
            beta_int1 = stage1_results[x[ns]]['beta_int1']
            beta_initial2[ns,:] = copy.deepcopy(betavals1)
            delta_initial2[ns,:] = copy.deepcopy(deltavals1)
            beta_int[ns] = copy.deepcopy(beta_int1)

        # test alternatives for latent input DB values - added May 3, 2025
        check_for_alternative_DBvals = True
        if check_for_alternative_DBvals:
            print('checking for alternative optimal betavals ...')
            Mintrinsic = copy.deepcopy(stage1_results[-1]['Mintrinsic'])
            beta_initial2, delta_initial2 = check_alternative_latent_DBvals(loadings, Minput, Mconn, Mintrinsic,
                                fintrinsic_count, beta_initial2, delta_initial2, ctarget, csource, dtarget, dsource, silentrunning = silentrunning)
            nsteps_stage2_temp, ndb = np.shape(beta_initial2)
            print('Number of steps for stage 2 is now {}'.format(nsteps_stage2_temp))
        else:
            nsteps_stage2_temp = nsteps_stage2

        # run stage 2---------------------------------------------------
        results_record, stage2_ssqd, stage2_slope, stage2_r2final, stage2_ssqd_slope, stage2_ssqd_final, stage2_results = \
                    run_sapm_stage_V2(nperson,2, nsteps_stage2, beta_initial2, delta_initial2, initial_alpha, initial_Lweight,
                            loadings, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, beta_int, fintrinsic_pc,
                            fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage2,
                                   converging_slope_limit[1], verbose = verbose, silentrunning = silentrunning)

        # end of stage2
        # get the best betavals from stage2 so far ...
        x = np.argmin(stage2_ssqd)
        betavals = stage2_results[x]['betavals']
        deltavals = stage2_results[x]['deltavals']
        beta_int = stage2_results[x]['beta_int1']

        stage2_monitoring_progress.append({'results':stage2_results})

        # run stage 3-------------------------------------------------------------
        results_record, stage3_ssqd, stage3_slope, stage3_r2final, stage3_ssqd_slope, stage3_ssqd_final, stage3_results = \
                    run_sapm_stage_V2(nperson,3, 1, betavals[np.newaxis,:], deltavals[np.newaxis,:], initial_alpha, initial_Lweight,
                            loadings, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, [beta_int], fintrinsic_pc,
                            fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage3,
                                   converging_slope_limit[2], verbose = verbose, silentrunning = silentrunning)

        # fit the results now to determine output signaling from each region
        betavals = copy.deepcopy(stage3_results[-1]['betavals'])
        deltavals = copy.deepcopy(stage3_results[-1]['deltavals'])
        Mintrinsic_pc = copy.deepcopy(stage3_results[-1]['Mintrinsic'])
        beta_int1 = copy.deepcopy(stage3_results[-1]['beta_int1'])

        pc_R2avg = copy.deepcopy(stage3_results[-1]['R2avg'])
        pc_R2total = copy.deepcopy(stage3_results[-1]['R2total'])

        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)

        #--------final fit to pcamethod results--------------
        # Nintrinsic = vintrinsic_count + fintrinsic_count
        # e, v = np.linalg.eig(Mconn)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
        # Meigv = np.real(v[:, -Nintrinsic:])
        # for aa in range(Nintrinsic):
        #     Meigv[:, aa] = Meigv[:, aa] / Meigv[(-Nintrinsic + aa), aa]
        # Mintrinsic = Mintrinsic_pc @ components
        #
        # fit = Minput @ Meigv @ Mintrinsic
        # err = np.sum((Sinput-fit)**2)
        #---------------------------------------------------


        R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
        R2avg = np.mean(R2list)
        R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

        # W = []
        # loadings_fit = []

        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'fit':fit, 'loadings':loadings, 'components':components,
                 'R2total':R2total, 'R2avg':R2avg, 'pc_R2total':pc_R2total, 'pc_R2avg':pc_R2avg, 'Mintrinsic':Mintrinsic,
                 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count, 'Meigv':Meigv,
                 'betavals':betavals, 'deltavals':deltavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
                 'cnums':cnums, 'fintrinsic_base':fintrinsic_base, 'Sinput_original':Sinput_original,
                 'DBname':DBname, 'DBnum':DBnum}

        SAPMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()
        np.save(SAPMresultsname, SAPMresults)

    # save betavals for use as starting point for other fits
    nb = len(betavals)
    betarecord = np.zeros((nb,NP_to_run))
    for nperson in range(NP_to_run):
        betarecord[:,nperson] = SAPMresults[nperson]['betavals']
    betainit = np.mean(betarecord, axis = 1)
    b = {'beta_initial':betainit}
    np.save(betavals_savename, b)

    # write out stage2 results for checking on performance
    p,f1 = os.path.split(SAPMresultsname)
    f,e = os.path.splitext(f1)
    stage2name = os.path.join(p,f[:15] + '_stage2.npy')
    np.save(stage2name, stage2_monitoring_progress)

    if verbose and not silentrunning:
        print('record of average betavals saved to {}'.format(betavals_savename))
        print('finished SAPM at {}'.format(time.ctime()))
        print('     started at {}'.format(starttime))
        print('     results written to {}'.format(SAPMresultsname))
    return SAPMresultsname



def get_data_components(Sinput, fixed_component = []):
    # first remove fixed component if it is provided
    nr,nt = np.shape(Sinput)

    d = np.array(np.shape(fixed_component))
    fixed_part = False
    if len(d) == 1:
        if d[0] == nt:
            fixed_part = True
            fixed_component = fixed_component[np.newaxis,:]
    if (len(d) == 2):
        if (d[0] <= nr) and (d[1] == nt):
            fixed_part = True
        if (d[1] <= nr) and (d[0] == nt):
            fixed_part = True
            fixed_component = fixed_component.T

    # remove fixed_component
    # Sinput = fixed_weight @ fixed_component + remainder
    if fixed_part:
        favg = np.mean(fixed_component, axis = 1)
        fixed_component -= np.repeat(favg[:,np.newaxis],nt, axis = 1)
        fixed_weight = Sinput @ fixed_component.T @ np.linalg.inv(fixed_component @ fixed_component.T)
        Sinput2 = Sinput - fixed_weight @ fixed_component
    else:
        Sinput2 = copy.deepcopy(Sinput)

    # get principal components of Sinput--------------------------
    nr = np.shape(Sinput)[0]
    pca = sklearn.decomposition.PCA()
    pca.fit(Sinput2)
    components = pca.components_
    loadings = pca.transform(Sinput2)
    mu2 = np.mean(Sinput2, axis=0)
    if fixed_part:
        loadings = np.concatenate((np.ones((nr, 1)), fixed_weight, loadings), axis=1)
        components = np.concatenate((mu2[np.newaxis, :], fixed_component, components), axis=0)
    else:
        loadings = np.concatenate((np.ones((nr, 1)), loadings), axis=1)
        components = np.concatenate((mu2[np.newaxis, :], components), axis=0)
    return components, loadings



def run_sapm_stage(nperson, stagenum, nsteps_stage, beta_initial, delta_initial, initial_alpha, initial_Lweight,
                   Sinput, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, beta_int, fintrinsic1,
                   fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage,
                   converging_slope_limit, verbose = False, silentrunning = True):
    # initialize
    results_record = []
    ssqd_record = []
    beta_init_record = []

    stage_ssqd = np.zeros(nsteps_stage)
    stage_slope = np.zeros(nsteps_stage)
    stage_r2final = np.zeros(nsteps_stage)
    stage_ssqd_slope = np.zeros(nsteps_stage)
    stage_ssqd_final = np.zeros(nsteps_stage)
    stage_results = []
    betalimit = 5.0

    for ns in range(nsteps_stage):
        ssqd_record_stage = []
        beta_init_record.append({'beta_initial': beta_initial[ns, :]})
        beta_int1 = copy.deepcopy(beta_int[ns])
        lastgood_beta_int1 = copy.deepcopy(beta_int1)

        # initalize Sconn
        betavals = copy.deepcopy(beta_initial[ns, :])  # initialize beta values
        nbeta = len(betavals)
        lastgood_betavals = copy.deepcopy(betavals)
        deltavals = copy.deepcopy(delta_initial[ns, :])
        lastgood_deltavals = copy.deepcopy(deltavals)

        alphalist = initial_alpha * np.ones(nbeta)
        alphabint = copy.deepcopy(initial_alpha)
        alpha = copy.deepcopy(initial_alpha)
        Lweight = copy.deepcopy(initial_Lweight)
        dval = copy.deepcopy(initial_dval)

        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqd += 0.001*latent_cost

        ssqd_starting = copy.deepcopy(ssqd)
        ssqd_starting0 = copy.deepcopy(ssqd)
        ssqd_old = copy.deepcopy(ssqd)
        ssqd_record += [ssqd]

        iter = 0
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        sequence_count = 0
        R2avg_record = []
        R2avg_slope = 0.
        ssqd_slope = 0.

        while alpha > alpha_limit and iter < nitermax_stage and converging:
            iter += 1

            betavals, deltavals, beta_int1, fit, dssq_db, dssq_dd, dssq_dbeta1, ssqd_original, ssqd, alpha, alphabint = \
                update_betavals_V4(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
                                   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
                                   vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
                                   latent_flag=latent_flag)  # kappavals, ktarget, ksource,

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
                if sequence_count > 4:
                    alpha = np.min([1.3 * alpha, initial_alpha])
                    alphabint = np.min([1.3 * alphabint, initial_alpha])
                    sequence_count = 0

            Mconn[ctarget, csource] = copy.deepcopy(betavals)
            Minput[dtarget, dsource] = copy.deepcopy(deltavals)

            fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                        vintrinsic_count, beta_int1, fintrinsic1)
            ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals,
                                                                     deltavals, regular_flag)
            # add cost for correlated latents
            # lcc = np.corrcoef(Mintrinsic)
            # latent_cost = np.sum(
            #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
            # ssqd += 0.001*latent_cost

            ssqd_record_stage += [ssqd]

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)

            R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
            R2avg = np.mean(R2list)
            R2avg_record += [R2avg]
            if len(R2avg_record) > 10:
                N = 5
                y = np.array(R2avg_record[-N:])
                x = np.array(range(N))
                R2avg_slope = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x * x) - np.sum(x) * np.sum(x))

                y = np.array(ssqd_record_stage[-N:])
                x = np.array(range(N))
                ssqd_slope = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x * x) - np.sum(x) * np.sum(x))
                if ssqd_slope > -converging_slope_limit:
                    converging = False

            R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

            results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

            ssqchange = ssqd - ssqd_original

            if verbose and not silentrunning:
                betamag = np.sqrt(np.sum(betavals ** 2))
                print('SAPM  {} stage{} pass {} iter {} alpha {:.3e}  error+cost {:.3f} error {:.3f} L1 cost {:.3f} change {:.3e}  R2 avg {:.3f}  R2 total {:.3f}  beta mag. {:.3f}'.format(
                        nperson, stagenum,
                        ns, iter, alpha, ssqd, error, error2, ssqchange, R2avg, R2total,
                        betamag))
            ssqd_old = copy.deepcopy(ssqd)
            # now repeat it ...
        stage_ssqd[ns] = ssqd
        stage_slope[ns] = R2avg_slope
        stage_r2final[ns] = R2avg_record[-1]

        stage_ssqd_slope[ns] = ssqd_slope
        stage_ssqd_final[ns] = ssqd_record_stage[-1]
        stage_results.append({'betavals': betavals, 'deltavals': deltavals, 'Mintrinsic':Mintrinsic,
                              'beta_int1':beta_int1, 'R2avg':R2avg, 'R2total':R2total})

    return results_record, stage_ssqd, stage_slope, stage_r2final, stage_ssqd_slope, stage_ssqd_final, stage_results




def run_sapm_stage_V2(nperson, stagenum, nsteps_stage, beta_initial, delta_initial, initial_alpha, initial_Lweight,
                   Sinput, Mconn, Minput, ctarget, csource, dtarget, dsource, initial_dval, beta_int, fintrinsic1,
                   fintrinsic_count, vintrinsic_count, latent_flag, regular_flag, alpha_limit, nitermax_stage,
                   converging_slope_limit, verbose = False, silentrunning = True):
    # this version updates the DB and D values separately

    # initialize
    results_record = []
    ssqd_record = []
    beta_init_record = []

    stage_ssqd = np.zeros(nsteps_stage)
    stage_slope = np.zeros(nsteps_stage)
    stage_r2final = np.zeros(nsteps_stage)
    stage_ssqd_slope = np.zeros(nsteps_stage)
    stage_ssqd_final = np.zeros(nsteps_stage)
    stage_results = []
    betalimit = 5.0

    for ns in range(nsteps_stage):
        ssqd_record_stage = []
        beta_init_record.append({'beta_initial': beta_initial[ns, :]})
        beta_int1 = copy.deepcopy(beta_int[ns])
        lastgood_beta_int1 = copy.deepcopy(beta_int1)

        # initalize Sconn
        betavals = copy.deepcopy(beta_initial[ns, :])  # initialize beta values
        nbeta = len(betavals)
        lastgood_betavals = copy.deepcopy(betavals)
        deltavals = copy.deepcopy(delta_initial[ns, :])
        lastgood_deltavals = copy.deepcopy(deltavals)

        alphalist = initial_alpha * np.ones(nbeta)
        alphabint = copy.deepcopy(initial_alpha)
        alphad = copy.deepcopy(initial_alpha)
        alpha = copy.deepcopy(initial_alpha)
        Lweight = copy.deepcopy(initial_Lweight)
        dval = copy.deepcopy(initial_dval)

        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        Minput[dtarget, dsource] = copy.deepcopy(deltavals)

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqd += 0.001*latent_cost

        # pre-run prep to get D vals matched with starting DB vals
        iter = 0
        dsequence_count = 0
        if verbose and not silentrunning:
            print('running pre-estimates of D values', end = "")
        while iter < 20:
            iter += 1
            if verbose and not silentrunning:
                print('.', end = "")
            # update D vals and beta_int1
            betavals, deltavals, beta_int1, fit, dssq_dd, dssq_dbeta1, ssqd_original, ssqd = \
                update_Dvals_V1(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
                                   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
                                   vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
                                   latent_flag=latent_flag)  # kappavals, ktarget, ksource,

            if ssqd > ssqd_original:
                alphad *= 0.5
                alphabint *= 0.5
                deltavals = copy.deepcopy(lastgood_deltavals)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                dsequence_count = 0
            else:
                lastgood_deltavals = copy.deepcopy(deltavals)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)
                dsequence_count += 1
                if dsequence_count > 4:
                    alphad = np.min([1.3 * alphad, initial_alpha])
                    alphabint = np.min([1.3 * alphabint, initial_alpha])
                    dsequence_count = 0


        if verbose and not silentrunning:
            print('done')
        alphabint = copy.deepcopy(initial_alpha)
        alphad = copy.deepcopy(initial_alpha)
        # end of pre-run prep -----------------------------------

        ssqd_starting = copy.deepcopy(ssqd)
        ssqd_starting0 = copy.deepcopy(ssqd)
        ssqd_old = copy.deepcopy(ssqd)
        ssqd_record += [ssqd]

        iter = 0
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        sequence_count = 0
        dsequence_count = 0
        R2avg_record = []
        R2avg_slope = 0.
        ssqd_slope = 0.

        while alpha > alpha_limit and iter < nitermax_stage and converging:
            iter += 1

            # update D vals and beta_int1
            betavals, deltavals, beta_int1, fit, dssq_dd, dssq_dbeta1, ssqd_original, ssqd = \
                update_Dvals_V1(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
                                   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
                                   vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
                                   latent_flag=latent_flag)  # kappavals, ktarget, ksource,

            if ssqd > ssqd_original:
                alphad *= 0.5
                alphabint *= 0.5
                deltavals = copy.deepcopy(lastgood_deltavals)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                dsequence_count = 0
            else:
                lastgood_deltavals = copy.deepcopy(deltavals)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)
                dsequence_count += 1
                if dsequence_count > 4:
                    alphad = np.min([1.3 * alphad, initial_alpha])
                    alphabint = np.min([1.3 * alphabint, initial_alpha])
                    dsequence_count = 0


            # update D vals and beta_int1, again
            betavals, deltavals, beta_int1, fit, dssq_dd, dssq_dbeta1, ssqd_original, ssqd = \
                update_Dvals_V1(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
                                   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
                                   vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
                                   latent_flag=latent_flag)  # kappavals, ktarget, ksource,

            if ssqd > ssqd_original:
                alphad *= 0.5
                alphabint *= 0.5
                deltavals = copy.deepcopy(lastgood_deltavals)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                dsequence_count = 0
            else:
                lastgood_deltavals = copy.deepcopy(deltavals)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)
                dsequence_count += 1
                if dsequence_count > 4:
                    alphad = np.min([1.3 * alphad, initial_alpha])
                    alphabint = np.min([1.3 * alphabint, initial_alpha])
                    bsequence_count = 0


            # update DB vals
            betavals, deltavals, beta_int1, fit, dssq_db, ssqd_original, ssqd = \
                update_DBvals_V1(Sinput, Minput, Mconn, betavals, deltavals, betalimit,
                                   ctarget, csource, dtarget, dsource, dval, fintrinsic_count,
                                   vintrinsic_count, beta_int1, fintrinsic1, Lweight, regular_flag, alpha, alphabint,
                                   latent_flag=latent_flag)


            if ssqd > ssqd_original:
                alpha *= 0.5
                betavals = copy.deepcopy(lastgood_betavals)  # no improvement, so don't update
                sequence_count = 0
            else:
                lastgood_betavals = copy.deepcopy(betavals)
                sequence_count += 1
                if sequence_count > 4:
                    alpha = np.min([1.3 * alpha, initial_alpha])
                    sequence_count = 0

            Mconn[ctarget, csource] = copy.deepcopy(betavals)
            Minput[dtarget, dsource] = copy.deepcopy(deltavals)

            fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                        vintrinsic_count, beta_int1, fintrinsic1)
            ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals,
                                                                     deltavals, regular_flag)
            # add cost for correlated latents
            # lcc = np.corrcoef(Mintrinsic)
            # latent_cost = np.sum(
            #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
            # ssqd += 0.001*latent_cost

            ssqd_record_stage += [ssqd]

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)

            R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / (np.sum(Sinput ** 2, axis=1) + 1.0e-6)
            R2avg = np.mean(R2list)
            R2avg_record += [R2avg]
            if len(R2avg_record) > 10:
                N = 5
                y = np.array(R2avg_record[-N:])
                x = np.array(range(N))
                R2avg_slope = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x * x) - np.sum(x) * np.sum(x))

                y = np.array(ssqd_record_stage[-N:])
                x = np.array(range(N))
                ssqd_slope = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x * x) - np.sum(x) * np.sum(x))
                if ssqd_slope > -converging_slope_limit:
                    converging = False

            R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

            results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

            ssqchange = ssqd - ssqd_original

            if verbose and not silentrunning:
                betamag = np.sqrt(np.sum(betavals ** 2))
                print('SAPM  {} stage{} pass {} iter {} alpha {:.3e}  alphad {:.3e} error+cost {:.3f} error {:.3f} L1 cost {:.3f} change {:.3e}  R2 avg {:.3f}  R2 total {:.3f}  beta mag. {:.3f}'.format(
                        nperson, stagenum,
                        ns, iter, alpha, alphad, ssqd, error, error2, ssqchange, R2avg, R2total,
                        betamag))
            ssqd_old = copy.deepcopy(ssqd)
            # now repeat it ...
        stage_ssqd[ns] = ssqd
        stage_slope[ns] = R2avg_slope
        stage_r2final[ns] = R2avg_record[-1]

        stage_ssqd_slope[ns] = ssqd_slope
        stage_ssqd_final[ns] = ssqd_record_stage[-1]
        stage_results.append({'betavals': betavals, 'deltavals': deltavals, 'Mintrinsic':Mintrinsic,
                              'beta_int1':beta_int1, 'R2avg':R2avg, 'R2total':R2total})

    return results_record, stage_ssqd, stage_slope, stage_r2final, stage_ssqd_slope, stage_ssqd_final, stage_results



#-------------------------------initialize betavals--------------------------------
def betaval_init_shotgun(Lweight, csource, ctarget, Sinput, Minput, Mconn, components, loadings, fintrinsic_count,
                                vintrinsic_count, deltavals, beta_int1, fintrinsic1, ncomponents_to_fit, nreps = 100000):
    search_record = []
    betascale = 0.2
    for rr in range(nreps):
        # initialize beta values at random values-----------------------------------
        betavals = betascale * np.random.randn(len(csource))  # initialize beta values at zero
        Mconn[ctarget, csource] = betavals

        # fit, loadings_fit, W, Mintrinsic, Meigv, err = network_eigenvector_method_V3(Sinput, components, loadings,
        #                                                 Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1,
        #                                                 fintrinsic1, ncomponents_to_fit)
        #
        # ssqd, error, error2, costfactor = sapm_error_function_V3(Sinput, Mconn, fit, loadings, loadings_fit, Lweight,
        #                                                          betavals, deltavals, regular_flag)  # , deltavals, beta_int1, Minput, Mintrinsic, Meigv

        fit, Mintrinsic, Meigv, err = network_eigenvector_method_V4(Sinput, Minput, Mconn, fintrinsic_count,
                                                                    vintrinsic_count, beta_int1, fintrinsic1)
        ssqd, error, error2, costfactor = sapm_error_function_V4(Sinput, Mconn, fit, Lweight, betavals, deltavals,
                                                                 regular_flag)
        # add cost for correlated latents
        # lcc = np.corrcoef(Mintrinsic)
        # latent_cost = np.sum(
        #     np.triu(np.abs(np.corrcoef(Mintrinsic[fintrinsic_count:, fintrinsic_count:])))) - vintrinsic_count
        # ssqd += 0.001*latent_cost

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
                        clusterdataname, samplesplit, samplestart=0, initial_clusters=[], timepoint='all', epoch='all',
                        betascale=0.1, alphascale = 0.01, Lweight = 0.01, levelthreshold = [1e-4, 1e-5, 1e-6],
                        leveliter = [100, 250, 1200], leveltrials = [30, 4, 1], run_whole_group = True):

    # initial_clusters has been changed to be the same format as cnums - Dec 2024
    # search is based on the clusters listed in initial_clusters now

    overall_start_time_text = time.ctime()
    overall_start_time = time.time()

    if not os.path.exists(outputdir): os.mkdir(outputdir)

    # load some data, setup some parameters...
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    nclusterlist = np.array([nclusterdict[x]['nclusters'] for x in range(len(nclusterdict))])
    cluster_name = [nclusterdict[x]['name'] for x in range(len(nclusterdict))]
    not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
    nclusterlist = nclusterlist[not_latent]
    full_rnum_base = [np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]
    namelist = [cluster_name[x] for x in not_latent]
    namelist += ['Rtotal']
    namelist += ['R ' + cluster_name[x] for x in not_latent]

    nregions = len(nclusterlist)

    print('best cluster search:  preparing data ...')
    filter_tcdata = False
    normalizevar = False
    subsample = [samplesplit,samplestart]  # [2,0] use every 2nd data set, starting with samplestart
    prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                  run_whole_group=False, normalizevar=normalizevar, filter_tcdata = filter_tcdata,
                                  subsample = subsample)

    print('best cluster search:  loading parameters ...')
    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]

    # check for bad data
    tcdata = copy.deepcopy(SAPMparams['tcdata_centered'])  # data for all regions/clusters concatenated along time dimension for all runs
    nclusters_total, tsize_total = np.shape(tcdata)
    # need to get principal components for each region to model the clusters as a continuum

    # ---moved this to prep_data_sem_physio_model_SO_V2
    # tplist_full = copy.deepcopy(SAPMparams['tplist_full'])
    # tcdata_centered = copy.deepcopy(SAPMparams['tcdata_centered'])
    # tcdata_centered_original = copy.deepcopy(SAPMparams['tcdata_centered_original'])
    # nruns_per_person = copy.deepcopy(SAPMparams['nruns_per_person'])
    # nruns_total = np.sum(nruns_per_person)
    # tsize = copy.deepcopy(SAPMparams['tsize'])
    # nrfull, tsizefull = np.shape(tcdata_centered)
    # good_runs = []
    # bad_runs = []
    # good_tp = []
    # nbad = 0
    # for nn in range(nruns_total):
    #     t1 = nn*tsize
    #     t2 = (nn+1)*tsize
    #     tp = list(range(t1,t2))
    #     check = (np.sum(tcdata_centered[:,tp]**2, axis = 1) == 0.).any()
    #     if check:
    #         nbad += 1
    #         bad_runs += [nn]
    #     else:
    #         good_runs += [nn]
    #         good_tp += [tp]
    # if nbad > 0:
    #     SAPMparams2 = copy.deepcopy(SAPMparams)
    #     tcdata_centered2 = copy.deepcopy(tcdata_centered[:,good_tp])
    #     tcdata_centered_original = copy.deepcopy(tcdata_centered_original[:,good_tp])
    #     DBnum2 = copy.deepcopy(SAPMparams['DBnum'][good_runs])
    #     bad_DBnum = copy.deepcopy(SAPMparams['DBnum'][bad_runs])
    #     nruns_cumulative = np.cumsum(nruns_per_person)
    #     nruns_per_person2 = copy.deepcopy(nruns_per_person)
    #     bad_people = []
    #     for bb in bad_runs:
    #         cc = np.where(nruns_cumulative > bb)[0][0]
    #         nruns_per_person2[cc] -= 1
    #         bad_people += [cc]
    #     bad_record, bad_counts = np.unique(bad_people, return_counts = True)
    #     print('Appears to be bad data in  DBnums {}'.format(bad_DBnum))
    #     for bb in range(len(bad_record)):
    #         print('   person {}  with {} bad runs'.format(bad_record[bb], bad_counts[bb]))

    maxiter = 50

    full_rnum_base = np.array([np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]).astype(int)

    # change format of cnums (and initial_clusters) - Nov 30 2024
    if (len(initial_clusters) > nregions):
        initial_clusters = initial_clusters[:nregions]
    if (len(initial_clusters) < nregions):
        temp_clusters = copy.deepcopy(initial_clusters)
        for x in range(len(initial_clusters),nregions):
            temp_clusters.append({'cnums':list(range(nclusterlist[x]))})
        initial_clusters = copy.deepcopy(temp_clusters)

    n_initial_clusters = [len(initial_clusters[x]['cnums']) for x in range(len(initial_clusters))]
    print('n_initial_clusters = {}'.format(n_initial_clusters))
    single_clusters = np.where(np.array(n_initial_clusters) == 1)[0]
    fixed_clusters = []
    for nn in range(len(initial_clusters)):
        if (n_initial_clusters[nn] == 1):
            if initial_clusters[nn]['cnums'][0] < 0:
                fixed_clusters += [nn]

    cluster_numbers = np.zeros(nregions)
    for nn in range(nregions):
        if nn in fixed_clusters:
            cluster_numbers[nn] = np.abs(initial_clusters[nn]['cnums'])-1
        else:
            cluster_numbers[nn] = np.random.choice(np.abs(initial_clusters[nn]['cnums']))

    cluster_numbers = np.array(cluster_numbers).astype(int)
    print('starting clusters: {}'.format(cluster_numbers))
    lastgood_clusters = copy.deepcopy(cluster_numbers)

    # if only one cluster listed per region, then use this as the starting point to search over all clusters
    print('single_clusters = {}'.format(single_clusters))
    if len(single_clusters) == nregions:
        initial_clusters = []
        n_initial_clusters = np.zeros(len(nclusterlist))
        for nn in range(len(nclusterlist)):
            if nn in fixed_clusters:
                initial_clusters += [{'cnums':[cluster_numbers[nn]]}]
                n_initial_clusters[nn] = 1
            else:
                initial_clusters += [{'cnums':list(range(nclusterlist[nn]))}]
                n_initial_clusters[nn] = nclusterlist[nn]

        print('searching clusters: {}'.format(initial_clusters))

    # gradient descent to find best cluster combination
    outputmessages = []
    messagecount = 0

    iter = 0
    costrecord = []
    print('starting step descent search of clusters at {}'.format(time.ctime()))
    converging = True

    if betascale == 0:
        nitermax = 50
        nitermax_stage1 = 1
        nsteps_stage1 = 10
    else:
        nitermax = leveliter[2]
        nsteps_stage1 = leveltrials[0]
        nitermax_stage1 = leveliter[0]
        nsteps_stage2 = leveltrials[1]
        nitermax_stage2 = leveliter[1]
        nitermax_stage3 = leveliter[2]

    print('best cluster search:  running baseline set of clusters ...')
    cnums = [{'cnums':[x]} for x in cluster_numbers]
    # output = sem_physio_model1_V4(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname,
    #                             fixed_beta_vals=[], betascale=betascale, alphascale = alphascale, Lweight = Lweight,
    #                             nitermax_stage3=nitermax,
    #                             nitermax_stage2=nitermax_stage2, nsteps_stage2=nsteps_stage2,
    #                             nitermax_stage1=nitermax_stage1, nsteps_stage1=nsteps_stage1,
    #                             levelthreshold=levelthreshold, verbose = False, run_whole_group = run_whole_group,
    #                             resumerun = False)

    output = sem_physio_model1_V5(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                                  fixed_beta_vals=[], betascale=betascale, alphascale=alphascale, Lweight=Lweight,
                                  nitermax_stage3=nitermax,
                                  nitermax_stage2=nitermax_stage2, nsteps_stage2=nsteps_stage2,
                                  nitermax_stage1=nitermax_stage1, nsteps_stage1=nsteps_stage1,
                                  levelthreshold=levelthreshold, verbose=False, run_whole_group=run_whole_group,
                                  resumerun=False, silentrunning = True)

    print('\r  finished running baseline clusters....                                         ')

    # now, correct the results for normalizing the variance
    if normalizevar:
        output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose = False)
    else:
        output = copy.deepcopy(SAPMresultsname)

    SAPMresults = np.load(output,allow_pickle=True)

    R2list = np.array([SAPMresults[x]['R2avg'] for x in range(len(SAPMresults))])
    R2list2 = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
    best_R2list = copy.deepcopy(R2list)
    best_R2list2 = copy.deepcopy(R2list2)
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
            cost_values = np.zeros(len(initial_clusters[nnn]['cnums']))
            current_results_record = []
            for xxx in range(len(initial_clusters[nnn]['cnums'])):
                current_results_record.append({'R2list': [], 'R2list2': [], 'region': nnn, 'cluster': xxx, 'ccount': 0})

            print('testing region {}'.format(nnn))
            if nnn in fixed_clusters:
                message = 'cluster for region {} is fixed at {}'.format(nnn,initial_clusters[nnn]['cnums'][0])
                print(message)
                mindex = np.sum(nclusterlist[:nn]) + initial_clusters[nnn]['cnums'][0]
                outputmessages += [message]
                messagecount += 1
            else:
                nclusterstosearch = len(initial_clusters[nnn]['cnums'])
                for ccount, ccc in enumerate(initial_clusters[nnn]['cnums']):
                    test_clusters = copy.deepcopy(cluster_numbers)
                    if test_clusters[nnn] == ccc:   # no change in cluster number from last run
                        cost_values[ccount] = lastcost
                        message = 'region {}  using cluster {}  total of (1-R2 avg) for the group is {:.3f} - current cluster'.format(nnn, ccc,cost_values[ccount])
                        print(message)
                        mindex = np.sum(nclusterlist[:nn]) + ccc
                        outputmessages += [message]
                        messagecount += 1
                    else:
                        test_clusters[nnn] = ccc
                        cnums = [{'cnums': [x]} for x in test_clusters]

                        output = sem_physio_model1_V5(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                                                      fixed_beta_vals=[], betascale=betascale, alphascale=alphascale,
                                                      Lweight=Lweight,
                                                      nitermax_stage3=nitermax,
                                                      nitermax_stage2=nitermax_stage2, nsteps_stage2=nsteps_stage2,
                                                      nitermax_stage1=nitermax_stage1, nsteps_stage1=nsteps_stage1,
                                                      levelthreshold=levelthreshold, verbose=False,
                                                      run_whole_group=run_whole_group,
                                                      resumerun=False, silentrunning = True)

                        if normalizevar:
                            output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose=False)
                        else:
                            output = copy.deepcopy(SAPMresultsname)
                        SAPMresults = np.load(output, allow_pickle=True)

                        # print('size of SAPMresults is {}'.format(np.shape(SAPMresults)))
                        R2list = np.array([SAPMresults[x]['R2avg'] for x in range(len(SAPMresults))])
                        R2list2 = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
                        cost_values[ccount] = np.sum(1 - R2list)

                        entry = {'R2list':R2list, 'R2list2':R2list2, 'region':nnn, 'cluster':ccc, 'ccount':ccount}
                        results_record.append(entry)
                        current_results_record[ccount] = copy.deepcopy(entry)
                        message = 'region {} using cluster {}  total of (1-R2 avg) for the group is {:.3f}   '.format(nnn, ccc,cost_values[ccount])
                        print('\r  region {} using cluster {}  total of (1-R2 avg) for the group is {:.3f}   '.format(nnn, ccc,cost_values[ccount]))
                        mindex = np.sum(nclusterlist[:nn]) + ccc
                        outputmessages += [message]
                        messagecount += 1

                x = np.argmin(cost_values)
                this_cost = cost_values[x]
                delta_cost = this_cost-lastcost
                if this_cost < lastcost:
                    cluster_numbers[nnn] = initial_clusters[nnn]['cnums'][x]
                    nbetterclusters += 1
                    lastcost = copy.deepcopy(this_cost)

                    # best_R2list = copy.deepcopy(results_record[x-nclusterstosearch-1]['R2list'])
                    # best_R2list2 = copy.deepcopy(results_record[x-nclusterstosearch-1]['R2list2'])

                    best_R2list = copy.deepcopy(current_results_record[x]['R2list'])
                    best_R2list2 = copy.deepcopy(current_results_record[x]['R2list2'])
                else:
                    print('no improvement in clusters found ... region {}'.format(nnn))

                message = 'iter {} region {} new cost = {:.3f}  previous cost = {:.3f} starting cost {:.3f}  delta cost = {:.3e} {}'.format(
                    iter, nnn, this_cost, lastcost, basecost, delta_cost, time.ctime())
                print(message)
                outputmessages += [message]
                messagecount += 1

        if nbetterclusters == 0:
            converging = False
            print('no improvement in clusters found in any region ...')

        # peek at results
        print('\nbest cluster set so far is : {}'.format(cluster_numbers))
        print('average R2 across data sets = {:.3f} {} {:.3f}'.format(np.mean(best_R2list),chr(177),np.std(R2list)))
        print('total R2 across data sets = {:.3f} {} {:.3f}'.format(np.mean(best_R2list2),chr(177),np.std(R2list2)))
        print('average R2 range {:.3f} to {:.3f}'.format(np.min(best_R2list),np.max(best_R2list2)))

        R2avg_text = '{:.3f} {} {:.3f}'.format(np.mean(best_R2list),chr(177),np.std(best_R2list))
        R2total_text = '{:.3f} {} {:.3f}'.format(np.mean(best_R2list2),chr(177),np.std(best_R2list2))
        R2range_text = '{:.3f} to {:.3f}'.format(np.min(best_R2list),np.max(best_R2list))
        results_record_summary.append({'best clusters':cluster_numbers, 'R2avg':R2avg_text, 'R2total':R2total_text, 'R2range':R2range_text})

        outputmessages += [R2avg_text]
        outputmessages += [R2total_text]
        outputmessages += [R2range_text]
        messagecount += 3

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

    cluster_numbers_output = [{'cnums':[x]} for x in cluster_numbers]

    return cluster_numbers_output, outputmessages

# # gradient descent method to find best clusters------------------------------------
# def SAPM_cluster_stepsearch_parallel(outputdir, SAPMresultsname, SAPMparametersname, networkfile, regiondataname,
#                         clusterdataname, samplesplit, samplestart=0, initial_clusters=[], timepoint='all', epoch='all', betascale=0.1, Lweight = 1.0):
#
#     overall_start_time_text = time.ctime()
#     overall_start_time = time.time()
#
#     if not os.path.exists(outputdir): os.mkdir(outputdir)
#
#     # load some data, setup some parameters...
#     network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
#     nclusterlist = np.array([nclusterdict[x]['nclusters'] for x in range(len(nclusterdict))])
#     cluster_name = [nclusterdict[x]['name'] for x in range(len(nclusterdict))]
#     not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
#     nclusterlist = nclusterlist[not_latent]
#     full_rnum_base = [np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]
#     namelist = [cluster_name[x] for x in not_latent]
#     namelist += ['Rtotal']
#     namelist += ['R ' + cluster_name[x] for x in not_latent]
#
#     nregions = len(nclusterlist)
#
#     print('best cluster search (parallel):  preparing data ...')
#     prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
#                                   run_whole_group=False, normalizevar=True, filter_tcdata = False)
#
#     print('best cluster search (parallel):  loading parameters ...')
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
#     # need to get principal components for each region to model the clusters as a continuum
#
#     nclusters_total, tsize_total = np.shape(tcdata)
#
#     maxiter = 50
#     subsample = [samplesplit,samplestart]  # [2,0] use every 2nd data set, starting with samplestart
#
#     full_rnum_base = np.array([np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]).astype(int)
#     initial_clusters = np.array(initial_clusters)
#     if (initial_clusters < 0).any():
#         fixed_clusters = np.where(initial_clusters >= 0)[0]
#     else:
#         fixed_clusters = []
#
#     if (len(initial_clusters) != nregions):
#         temp_clusters = -1*np.ones(nregions)
#         temp_clusters[:len(initial_clusters)] = copy.deepcopy(initial_clusters)   # pad list with -1
#         initial_clusters = copy.deepcopy(temp_clusters)
#
#     cluster_numbers = np.zeros(nregions)
#     for nn in range(nregions):
#         if initial_clusters[nn] < 0:
#             cnum = np.random.choice(range(nclusterlist[nn]))
#             cluster_numbers[nn] = copy.deepcopy(cnum)
#         else:
#             cluster_numbers[nn] = copy.deepcopy(initial_clusters[nn])
#     cluster_numbers = np.array(cluster_numbers).astype(int)
#
#     print('starting clusters: {}'.format(cluster_numbers))
#
#     lastgood_clusters = copy.deepcopy(cluster_numbers)
#
#     # gradient descent to find best cluster combination
#     iter = 0
#     costrecord = []
#     print('starting step descent search of clusters (parallel) at {}'.format(time.ctime()))
#     converging = True
#
#     if betascale == 0:
#         nitermax = 50
#         nitermax_stage1 = 1
#         nsteps_stage1 = 10
#     else:
#         nitermax = 100
#         nitermax_stage1 = 30
#         nitermax_stage2 = 50
#         nsteps_stage1 = 20
#
#     # output = sem_physio_model1_V3(cluster_numbers+full_rnum_base, fintrinsic_base, SAPMresultsname, SAPMparametersname,
#     #                               fixed_beta_vals=[], betascale=betascale, Lweight = Lweight, nitermax=nitermax, verbose=False,normalizevar=False,
#     #                               nitermax_stage1=nitermax_stage1, nsteps_stage1=nsteps_stage1, converging_slope_limit = [1e-3,1e-5])
#
#     print('best cluster search (parallel):  running baseline set of clusters ...')
#     output = sem_physio_model1_V4(cluster_numbers+full_rnum_base, fintrinsic_base, SAPMresultsname, SAPMparametersname,
#                                 fixed_beta_vals=[], betascale=betascale, Lweight = Lweight, nitermax_stage3=nitermax,
#                                 nitermax_stage2=nitermax_stage2, nsteps_stage2=2,
#                              nitermax_stage1=nitermax_stage1, nsteps_stage1=nsteps_stage1,
#                              converging_slope_limit=[1e-4, 1e-5, 1e-6], verbose = False)
#
#     # now, correct the results for normalizing the variance
#     output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose = False)
#
#     SAPMresults = np.load(output,allow_pickle=True)
#
#     R2list = np.array([SAPMresults[x]['R2avg'] for x in range(len(SAPMresults))])
#     basecost = np.sum(1 - R2list)
#     lastcost = copy.deepcopy(basecost)
#     results_record = []
#     results_record_summary = []
#     while converging and (iter < maxiter):
#         iter += 1
#         nbetterclusters = 0
#         random_region_order = list(range(nregions))
#         np.random.shuffle(random_region_order)
#         for nnn in random_region_order:
#             cost_values = np.zeros(nclusterlist[nnn])
#             print('testing region {}'.format(nnn))
#             if nnn in fixed_clusters:
#                 print('cluster for region {} is fixed at {}'.format(nnn,cluster_numbers[nnn]))
#             else:
#                 parameters = []
#                 tested_cluster_number = np.zeros(nclusterlist[nnn]-1, dtype = int)
#                 clustercount = 0
#                 for ccc in range(nclusterlist[nnn]):
#                     test_clusters = copy.deepcopy(cluster_numbers)
#                     if test_clusters[nnn] != ccc:   # no change in cluster number from last run
#                         test_clusters[nnn] = ccc
#                         tested_cluster_number[clustercount] = ccc
#                         clustercount += 1
#
#                         params = {'cluster_numbers':test_clusters + full_rnum_base,
#                                     'fintrinsic_base':fintrinsic_base, 'SAPMresultsname':SAPMresultsname,
#                                     'SAPMparametersname':SAPMparametersname, 'fixed_beta_vals':[],
#                                     'betascale':betascale,'Lweight':Lweight, 'normalizevar':False,
#                                     'nitermax_stage3':nitermax,'nitermax_stage2':nitermax_stage2,
#                                     'nitermax_stage1':nitermax_stage1,'nsteps_stage1':nsteps_stage1,
#                                     'nsteps_stage2':2, 'converging_slope_limit':[1e-4, 1e-5, 1e-6] }
#                         parameters.append(params)
#
#                 nprocessors = nclusterlist[nnn]-1
#                 pool = mp.Pool(nprocessors)
#                 searchresults = pool.map(SAPM_parallel_runs, parameters)
#                 pool.close()
#
#                 cost_values = np.zeros(nclusterlist[nnn])
#                 cost_values[cluster_numbers[nnn]] = copy.deepcopy(lastcost)
#                 for ccc in range(nclusterlist[nnn]-1):
#                     R2list = searchresults[ccc]['R2list']
#                     R2list2 = searchresults[ccc]['R2list2']
#                     cost_value = searchresults[ccc]['cost_value']
#                     cost_values[tested_cluster_number[ccc]] = cost_value
#
#                     entry = {'R2list':R2list, 'R2list2':R2list2, 'region':nnn, 'cluster':ccc}
#                     results_record.append(entry)
#                     print('  using cluster {}  total of (1-R2 avg) for the group is {:.3f}'.format(ccc,cost_values[ccc]))
#
#                 x = np.argmin(cost_values)
#                 this_cost = cost_values[x]
#                 delta_cost = this_cost-lastcost
#                 if this_cost < lastcost:
#                     cluster_numbers[nnn] = x
#                     nbetterclusters += 1
#                     lastcost = copy.deepcopy(this_cost)
#                 else:
#                     print('no improvement in clusters found ... region {}'.format(nnn))
#
#                 print('iter {} region {} new cost = {:.3f}  previous cost = {:.3f} starting cost {:.3f}  delta cost = {:.3e} {}'.format(
#                     iter, nnn, this_cost, lastcost, basecost, delta_cost, time.ctime()))
#
#         if nbetterclusters == 0:
#             converging = False
#             print('no improvement in clusters found in any region ...')
#
#         # peek at results
#         print('\nbest cluster set so far is : {}'.format(cluster_numbers))
#         print('average R2 across data sets = {:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list)))
#         print('total R2 across data sets = {:.3f} {} {:.3f}'.format(np.mean(R2list2),chr(177),np.std(R2list2)))
#         print('average R2 range {:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list)))
#
#         R2avg_text = '{:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list))
#         R2total_text = '{:.3f} {} {:.3f}'.format(np.mean(R2list2),chr(177),np.std(R2list2))
#         R2range_text = '{:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list))
#         results_record_summary.append({'best clusters':cluster_numbers, 'R2avg':R2avg_text, 'R2total':R2total_text, 'R2range':R2range_text})
#
#     outputname = os.path.join(outputdir, 'step_descent_record.npy')
#     np.save(outputname, results_record)
#     print('results record written to {}'.format(outputname))
#
#     outputname = os.path.join(outputdir, 'step_descent_record_summary.npy')
#     np.save(outputname, results_record_summary)
#     print('results record summary written to {}'.format(outputname))
#
#     overall_end_time_text = time.ctime()
#     overall_end_time = time.time()
#     dtime = overall_end_time-overall_start_time
#     dtimem = np.floor(dtime/60).astype(int)
#     dtimes = np.round(dtime % 60).astype(int)
#     print('Cluster search started at {}\n             and ended at {}\n     {} minutes {} sec total'.format(overall_start_time_text, overall_end_time_text, dtimem,dtimes))
#     return cluster_numbers
#
#
# def SAPM_parallel_runs(paramslist):
#
#     cluster_numbers = paramslist['cluster_numbers']
#     fintrinsic_base = paramslist['fintrinsic_base']
#     SAPMresultsname = paramslist['SAPMresultsname']
#     SAPMparametersname = paramslist['SAPMparametersname']
#     fixed_beta_vals = paramslist['fixed_beta_vals']
#     betascale = paramslist['betascale']
#     Lweight = paramslist['Lweight']
#     normalizevar = paramslist['normalizevar']
#     nitermax_stage3 = paramslist['nitermax_stage3']
#     nitermax_stage2 = paramslist['nitermax_stage2']
#     nitermax_stage1 = paramslist['nitermax_stage1']
#     nsteps_stage1 = paramslist['nsteps_stage1']
#     nsteps_stage2 = paramslist['nsteps_stage2']
#     converging_slope_limit = paramslist['converging_slope_limit']
#
#     output = sem_physio_model1_V4(cluster_numbers, fintrinsic_base, SAPMresultsname,
#                                   SAPMparametersname, fixed_beta_vals=fixed_beta_vals, betascale=betascale,
#                                   Lweight=Lweight, nitermax_stage3=nitermax_stage3,
#                                   nitermax_stage2=nitermax_stage2, nsteps_stage2=nsteps_stage2,
#                                   nitermax_stage1=nitermax_stage1, nsteps_stage1=nsteps_stage1,
#                                   converging_slope_limit=converging_slope_limit, verbose=False, silentrunning = True)
#
#     output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose=False)
#     SAPMresults = np.load(output, allow_pickle=True)
#
#     # print('size of SAPMresults is {}'.format(np.shape(SAPMresults)))
#     R2list = np.array([SAPMresults[x]['R2avg'] for x in range(len(SAPMresults))])
#     R2list2 = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#     cost_value = np.sum(1 - R2list)
#
#     return {'R2list':R2list, 'R2list2':R2list2, 'cost_value':cost_value}
#


def nperms(n, input_list):
	# find all possible combinations of n values selected out of the input list
	# without duplicates

	nvals = len(input_list)
	permutations = []
	dimspec = [nvals for xx in range(n)]
	grid = np.ones(dimspec)

	coords = np.zeros((n,nvals**n))
	for nn in range(n):
		predims = nn
		postdims = n-nn-1
		d1 = np.tile(np.array(range(nvals)),nvals**postdims)
		d1 = np.repeat(d1,nvals**predims)
		coords[nn,:] = d1

	dcoords = coords[1:,:]-coords[:-1,:]
	keeplist = [xx for xx in range(nvals**n) if np.min(dcoords[:,xx]) > 0]
	perms = np.array(coords[:,keeplist]).astype(int)

	nperms = np.shape(perms)[1]
	output_list = []
	for xx in range(nperms):
		sample = [input_list[c] for c in perms[:,xx]]
		output_list += [sample]

	return np.array(output_list), perms


def cluster_search_pca(regiondataname, networkmodel, initial_clusters = []):
    # regiondata = np.load(regiondataname, allow_pickle=True).flat[0]
    region_data = load_filtered_regiondata(regiondataname, networkfile)

    region_properties = regiondata['region_properties']
    nregions = len(region_properties)
    rnamelist = [region_properties[xx]['rname'] for xx in range(nregions)]
    nruns_per_person = region_properties[0]['nruns_per_person']
    tsize = region_properties[0]['tsize']

    NP = len(nruns_per_person)
    nruns_total = np.sum(nruns_per_person)

    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkmodel)
    nclusterlist = []
    for xx in range(len(nclusterdict)):
        if not 'intrinsic' in nclusterdict[xx]['name']:
            nclusterlist += [nclusterdict[xx]['nclusters']]
    nclusterlist = np.array(nclusterlist)
    nclusterstotal = np.sum(nclusterlist)

    if fintrinsic_count > 0:
        # fintrinsic_base = copy.deepcopy(params['fintrinsic_base'])
        fintrinsic_base -= np.mean(fintrinsic_base)
        fintrinsic_tc = copy.deepcopy(fintrinsic_base)
    else:
        fintrinsic_tc = []

    vdata = []
    for nn in range(NP):
        # data per person
        Sinput = np.zeros((nclusterstotal,tsize*nruns_per_person[nn]))
        t1 = np.sum(nruns_per_person[:nn])*tsize
        t2 = np.sum(nruns_per_person[:(nn+1)])*tsize
        for rr in range(nregions):
            c1 = np.sum(nclusterlist[:rr]).astype(int)
            c2 = np.sum(nclusterlist[:(rr+1)])
            Sinput[c1:c2,:] = region_properties[rr]['tc'][:,t1:t2]

        if fintrinsic_count > 0:
            #  Sinput = f_b @ flatent_tc    - fit fixed latent component if there is one
            flatent_tc = np.repeat(fintrinsic_tc, nruns_per_person[nn], axis=1)
            f_b = Sinput @ flatent_tc.T @ np.linalg.inv(flatent_tc @ flatent_tc.T)
            f_fit = f_b @ flatent_tc
            Sinput_res = Sinput - f_fit   # take out fixed latent component
            var_flatent = np.var(Sinput,axis=1) - np.var(Sinput_res,axis=1)
        else:
            Sinput_res = copy.deepcopy(Sinput)
            var_flatent = np.zeros(nclusterstotal)

        # get principal components and weights for timecourse data in Sinput
        # nregions, tsizefull = np.shape(Sinput)
        # Sin_std = np.repeat(np.std(Sinput, axis=1)[:, np.newaxis], tsizefull, axis=1)
        # Sinput_norm = Sinput / Sin_std
        pca = PCA(n_components=nclusterstotal)
        pca.fit(Sinput_res)
        # S_pca_ = pca.fit(Sinput).transform(Sinput)

        components = pca.components_
        evr = pca.explained_variance_ratio_
        ev = pca.explained_variance_
        # get loadings
        mu = np.mean(Sinput_res, axis=0)   # the average component is separate and messes up everything because
                                            # it might not be linearly independent of other components
        vm = np.var(mu, ddof=1)
        mu = np.repeat(mu[np.newaxis, :], nclusterstotal, axis=0)

        loadings = pca.transform(Sinput_res)
        # fit_check = (loadings @ components) + mu

        # now find which set of clusters can be best explained by only vintrinsic_count PCA components
        vc = np.var(components,axis=1, ddof = 1)
        v_by_component = (loadings**2) * np.repeat(vc[:,np.newaxis],nclusterstotal,axis=1)
        vS = np.var(Sinput_res - mu, axis=1, ddof = 1)
        v_ratio_by_component = v_by_component / np.repeat(vS[:,np.newaxis],nclusterstotal,axis=1)

        if fintrinsic_count > 0:
            v_ratio_flatent = var_flatent / vS
            v_ratio_by_component = np.concatenate((v_ratio_by_component,v_ratio_flatent[:,np.newaxis]),axis=1)

        # v_ratio_by_component is [cluster_number x component]
        # find which combination of cluster numbers gives the highest total for some set of compoents across all people
        vdata.append({'vratio':v_ratio_by_component})
        if nn == 0:
            vdata_group = copy.deepcopy(v_ratio_by_component[:,:,np.newaxis])
        else:
            vdata_group = np.concatenate((vdata_group,v_ratio_by_component[:,:,np.newaxis]),axis=2)

    # find one cluster per region that gives the best overall fit for all people
    # permutations of first Nsearch components
    Nsearch = 8
    cnumset = list(range(Nsearch))
    component_list, perms = nperms(vintrinsic_count+2, cnumset)   # combinations of PCA terms to check
    ncombinations = np.shape(component_list)[0]

    multistep_results = []
    nrepeats = 10
    for rrr in range(nrepeats):
        # gradient-descent type search---------------------------------------
        # initial cluster guess
        offset = np.array([0] + list(np.cumsum(nclusterlist))[:-1])
        cluster_numbers = np.zeros(len(nclusterlist))
        for nn in range(len(nclusterlist)):
            cnum = np.random.choice(range(nclusterlist[nn]))
            cluster_numbers[nn] = copy.deepcopy(cnum)
        cnumlist = (cluster_numbers + offset).astype(int)

        # change format of cnums (and initial_clusters) - Nov 30 2024
        initial_cluster_list = np.array([initial_clusters[x]['cnums'][0] for x in range(len(initial_clusters))])
        if (initial_cluster_list < 0).any():
            fixed_clusters = np.where(initial_cluster_list >= 0)[0]
            cluster_numbers[fixed_clusters] = copy.deepcopy(initial_cluster_list[fixed_clusters])
        else:
            fixed_clusters = []
        print('Search for best clusters based on PCA method ...')
        print('     clusters {} are fixed at {}'.format(fixed_clusters, initial_cluster_list[fixed_clusters]))

        lastavg = 0
        maxiter = 100
        iter = 0
        converging = True
        verbose = False
        while converging and (iter < maxiter):
            iter += 1
            nbetterclusters = 0
            random_region_order = list(range(nregions))
            np.random.shuffle(random_region_order)
            for nnn in random_region_order:
                avg_var_list = np.zeros(nclusterlist[nnn])
                print('testing region {}'.format(nnn))
                if nnn in fixed_clusters:
                    print('cluster for region {} is fixed at {}'.format(nnn, cluster_numbers[nnn]))
                else:
                    for ccc in range(nclusterlist[nnn]):
                        test_clusters = copy.deepcopy(cluster_numbers)
                        if test_clusters[nnn] == ccc:  # no change in cluster number from last run
                            avg_var_list[ccc] = lastavg
                            if verbose:
                                print('  using cluster {}  total of avg. variance explained for the group is {:.3f} - current cluster'.format(ccc, avg_var_list[ccc]))
                        else:
                            test_clusters[nnn] = ccc
                            cnumlist = (test_clusters + offset).astype(int)

                            # find best combination in each person
                            best_var = np.zeros(NP)
                            for pp in range(NP):
                                var_check_list = np.zeros(ncombinations)
                                for nn in range(ncombinations):
                                    complist = component_list[nn, :]
                                    if fintrinsic_count > 0:
                                        complist = np.concatenate((complist, [nclusterstotal]))

                                    vdata_subset = vdata_group[cnumlist, :, pp]
                                    # check = np.mean(np.sum(vdata_group[:,cc,:],axis=1),axis=1)
                                    var_check_list[nn] = np.sum(vdata_subset[:,complist])  # how much variance is accounted for by these components, for each cluster, in each person
                                best_var[pp] = np.max(var_check_list)
                            avg_var_list[ccc] = np.mean(best_var)

                            # entry = {'R2list': R2list, 'R2list2': R2list2, 'region': nnn, 'cluster': ccc}
                            # results_record.append(entry)
                            if verbose:
                                print('  using cluster {}  total of avg. variance explained for the group is {:.3f}'.format(ccc, avg_var_list[ccc]))

                    x = np.argmax(avg_var_list)
                    this_avg = avg_var_list[x]
                    delta_avg = this_avg - lastavg
                    if this_avg > lastavg:
                        cluster_numbers[nnn] = x
                        nbetterclusters += 1
                        lastavg = copy.deepcopy(this_avg)
                    else:
                        if verbose:
                            print('no improvement in clusters found ... region {}'.format(nnn))

                    print('iter {} region {} new avg. variance = {:.3f}  previous avg. variance  = {:.3f}  delta variance = {:.3e} {}'.format(
                            iter, nnn, this_avg, lastavg, delta_avg, time.ctime()))

            if nbetterclusters == 0:
                converging = False
                print('no improvement in clusters found in any region ...')

        multistep_results.append({'cluster_numbers':cluster_numbers, 'lastavg':lastavg})

    # find best of nrepeats
    lastavg_record = [multistep_results[xx]['lastavg'] for xx in range(nrepeats)]
    dd = np.argmax(lastavg_record)
    best_cluster_numbers = multistep_results[dd]['cluster_numbers'].astype(int)

    print('\nbest overall cluster set is : {}'.format(best_cluster_numbers))
    print('\n    overall variance estimate accounted for is {:.3f}'.format(np.max(lastavg_record)))

    best_cluster_numbers_output = [{'cnums':[x]} for x in best_cluster_numbers]

    return best_cluster_numbers_output


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

    Nintrinsics = vintrinsic_count + fintrinsic_count

    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
    NP = len(SAPMresults_load)

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
        # loadings = copy.deepcopy(SAPMresults_load[nperson]['loadings'])
        # components = copy.deepcopy(SAPMresults_load[nperson]['components'])
        # loadings_fit = copy.deepcopy(SAPMresults_load[nperson]['loadings_fit'])
        clusterlist = copy.deepcopy(SAPMresults_load[nperson]['clusterlist'])
        fintrinsic1 = copy.deepcopy(SAPMresults_load[nperson]['fintrinsic1'])

        nr, nr_nL = np.shape(Sinput)
        # for aa in range(nr):
        #     loadings[aa,:] /= (std_scale[clusterlist[aa],nperson] + 1.0e-10)
        #     loadings_fit[aa,:] /= (std_scale[clusterlist[aa],nperson] + 1.0e-10)

        # correct deltavals and betavals
        deltavals = np.zeros(len(csource))
        for aa in range(len(csource)):
            ss = csource[aa]
            tt = ctarget[aa]
            Minput[tt, ss] /= (std_scale[clusterlist[tt], nperson] + 1.0e-10)
            deltavals[aa] = copy.deepcopy(Minput[tt,ss])

        # fit the result ...
        # nr, ncomponents_to_fit = np.shape(loadings_fit)

        fit_original = Minput @ Sconn
        fit_original1 = Minput @ Meigv @ Mintrinsic

        R2list = 1.0 - np.sum((Sinput_original - fit_original) ** 2, axis=1) / (np.sum(Sinput_original ** 2, axis=1) + 1.0e-10)
        R2avg = np.mean(R2list)
        R2total = 1.0 - np.sum((Sinput_original - fit_original) ** 2) / (np.sum(Sinput_original ** 2) + 1.0e-10)

        if verbose:
            print('SAPM {} variance correction:  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson, R2avg, R2total))

        # put back the results
        SAPMresults_load[nperson]['Minput'] = copy.deepcopy(Minput)
        SAPMresults_load[nperson]['R2total'] = copy.deepcopy(R2total)
        SAPMresults_load[nperson]['R2avg'] = copy.deepcopy(R2avg)
        SAPMresults_load[nperson]['deltavals'] = copy.deepcopy(deltavals)
        # SAPMresults_load[nperson]['loadings'] = copy.deepcopy(loadings)
        # SAPMresults_load[nperson]['loadings_fit'] = copy.deepcopy(loadings_fit)

    p,fe = os.path.split(SAPMresultsname)
    f,e = os.path.splitext(fe)
    outputname = os.path.join(p,f+'_corr'+e)
    np.save(outputname, SAPMresults_load)
    return outputname


def sem_physio_correct_for_normalization2(SAPMresultsname, SAPMparametersname, verbose=True):
    p, f = os.path.split(SAPMresultsname)
    if '_corr' in f:
        f = f.replace('_corr', '_original')
    else:
        f2, e = os.path.splitext(f)
        f = f2 + '_original.npy'
    original_name = os.path.join(p, f)

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

    Nintrinsics = vintrinsic_count + fintrinsic_count

    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
    NP = len(SAPMresults_load)
    new_SAPMresults = copy.deepcopy(SAPMresults_load)

    # original results
    for nperson in range(NP):
        Sinput = copy.deepcopy(SAPMresults_load[nperson]['Sinput'])
        Sinput_original = copy.deepcopy(SAPMresults_load[nperson]['Sinput_original'])
        Sconn = copy.deepcopy(SAPMresults_load[nperson]['Sconn'])
        Minput = copy.deepcopy(SAPMresults_load[nperson]['Minput'])
        Mconn = copy.deepcopy(SAPMresults_load[nperson]['Mconn'])
        betavals = copy.deepcopy(SAPMresults_load[nperson]['betavals'])
        Meigv = copy.deepcopy(SAPMresults_load[nperson]['Meigv'])
        Mintrinsic = copy.deepcopy(SAPMresults_load[nperson]['Mintrinsic'])
        cnums = copy.deepcopy(SAPMresults_load[nperson]['cnums'])
        clusterlist = copy.deepcopy(SAPMresults_load[nperson]['clusterlist'])
        beta_int1 = copy.deepcopy(SAPMresults_load[nperson]['beta_int1'])

        nr, tsize_total = np.shape(Sinput)

        # find region with fixed intrinsic input
        if fintrinsic_count > 0:
            c = np.where(np.abs(Minput[:,-Nintrinsics]) > 0)[0]

        W1 = std_scale[clusterlist, nperson]
        W = np.eye(nr + Nintrinsics)
        Wi = np.eye(nr + Nintrinsics)
        for ww in range(nr):
            W[ww, ww] = W1[ww]
            Wi[ww, ww] = 1.0 / W1[ww]

        beta_int_new = beta_int1 * Wi[c,c]
        Minput_original = W[:nr,:nr] @ Minput

        Minput_full = np.zeros((nr + Nintrinsics, nr + Nintrinsics))
        Minput_full[:nr, :] = copy.deepcopy(Minput_original)
        for xx in range(Nintrinsics):
            Minput_full[-xx - 1, -xx - 1] = 1.0

        G = np.linalg.inv(Minput_full.T @ Minput_full) @ Minput_full.T @ Wi
        Gi = np.linalg.inv(G)

        Sconn_original = G @ Sconn

        # new eigenvectors etc
        e, v = np.linalg.eig(Mconn)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
        v_original = G @ v
        E = np.diag(e)
        Mconn_original_rec = v_original @ E @ np.linalg.inv(v_original)
        betavals_original_rec = Mconn_original_rec[ctarget, csource]

        Meigv_original_rec = np.real(v_original[:, -Nintrinsics:])
        for aa in range(Nintrinsics):
            Meigv_original_rec[:, aa] = Meigv_original_rec[:, aa] / Meigv_original_rec[(-Nintrinsics + aa), aa]


        # other method for checking
        # Sinput_original_check = Wi[:nr, :nr] @ Sinput
        # Sconn_original = G @ Sconn
        # Mconn_original = G @ Mconn @ Gi
        # betavals_original = Mconn_original[ctarget, csource]

        # e, v = np.linalg.eig(Mconn)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
        # v_original = G @ v
        # E = np.diag(e)
        # Mconn_original_rec = np.real(v_original @ E @ np.linalg.inv(v_original))
        # betavals_original_rec = Mconn_original_rec[ctarget, csource]

        # e, v = np.linalg.eig(Mconn_original)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
        # Meigv_original = np.real(v[:, -Nintrinsics:])
        # for aa in range(Nintrinsics):
        #     Meigv_original[:, aa] = Meigv_original[:, aa] / Meigv_original[(-Nintrinsics + aa), aa]

        # Meigv_original_check = G @ Meigv
        # for aa in range(Nintrinsics):
        #     Meigv_original_check[:, aa] = Meigv_original_check[:, aa] / Meigv_original_check[(-Nintrinsics + aa), aa]

        # Mconn_original @ Meigv_original[:,0] - Meigv_original[:,0]
        # Mconn_original @ Meigv_original_check[:,0] - Meigv_original_check[:,0]

        Mintrinsic_original_rec = copy.deepcopy(Mintrinsic)
        nr, tsize_total = np.shape(Sinput)
        if fintrinsic_count > 0:
            fintrinsic1 = beta_int_new * Mintrinsic[0, :] / beta_int1
            f1 = fintrinsic1[np.newaxis, :]

            # fit the fixed intrinsic, remove it, and then fit the variable intrinsics to the remainder
            new_Mintrinsic = np.zeros((Nintrinsics, tsize_total))
            Mint_fixed = Mintrinsic[0, :][np.newaxis, :]
            partial_fit = (Minput_full[:nr,:] @ Meigv_original_rec[:, 0])[:, np.newaxis] @ Mint_fixed  # is this right?

            residual = Sinput_original - partial_fit
            M1r = Minput_full[:nr,:] @ Meigv_original_rec[:, 1:]

            Mint_variable = np.linalg.inv(M1r.T @ M1r) @ M1r.T @ residual

            Mintrinsic_original_rec[0, :] = copy.deepcopy(Mint_fixed)
            Mintrinsic_original_rec[1:, :] = copy.deepcopy(Mint_variable)
        else:
            M1r = Minput_full[:nr,:] @ Meigv_original_rec
            Mintrinsic_original_rec = np.linalg.inv(M1r.T @ M1r) @ M1r.T @ Sinput_original

        # Sconn_original_check = Meigv_original_check @ Mintrinsic_original_check
        # Sinput_original_check = Minput_full[:nr,:] @ Sconn_original_check

        fit = Minput @ Sconn
        fit2 = Minput_full[:nr,:] @ Sconn_original

        # now work backwards because this was based on Min_original = Min
        # Sinput_original = Minput_original @ Sconn_original
        # need to preserve the zeros and scaling

        new_SAPMresults[nperson]['Sconn'] = copy.deepcopy(Sconn_original)
        new_SAPMresults[nperson]['Mconn'] = copy.deepcopy(Mconn_original_rec)
        new_SAPMresults[nperson]['Meigv'] = copy.deepcopy(Meigv_original_rec)
        new_SAPMresults[nperson]['Mintrinsic'] = copy.deepcopy(Mintrinsic_original_rec)
        new_SAPMresults[nperson]['betavals'] = copy.deepcopy(betavals_original_rec)


    np.save(original_name, new_SAPMresults)
    return original_name


# main program
def SAPMrun_V2(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkfile, timepoint,
            epoch, betascale = 0.1, Lweight = 0.01, alphascale = 0.01, leveltrials = [30, 4, 1], leveliter = [100, 250, 1200],
               levelthreshold = [1e-4, 1e-5, 1e-6], reload_existing = False, fully_connected = False, run_whole_group = False,
               resumerun = False, verbose = True, silentrunning = False, normalizevar = False):

    # load some data, setup some parameters...
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    nclusterlist = np.array([nclusterdict[x]['nclusters'] for x in range(len(nclusterdict))])
    cluster_name = [nclusterdict[x]['name'] for x in range(len(nclusterdict))]
    not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
    nclusterlist = nclusterlist[not_latent]
    full_rnum_base = [np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]
    namelist = [cluster_name[x] for x in not_latent]
    namelist += ['Rtotal']
    namelist += ['R ' + cluster_name[x] for x in not_latent]

    # starting values
    cnums_original = copy.deepcopy(cnums)
    excelsheetname = 'clusters'

    # run the analysis with SAPM
    # clusterlist = np.array(cnums) + full_rnum_base
    # change format of cnums - Nov 30 2024
    # clusterlist = [(cnums[x]['cnums'][0] + full_rnum_base[x]) for x in range(len(cnums))]

    if resumerun:
        print('Resuming previous run, using previous parameters file ...')
    else:
        if fully_connected:
            # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
            for x in range(len(cnums)):
                if np.min(cnums[x]['cnums']) < 0:
                    cnums[x]['cnums'] = list(range(nclusterlist[x]))
            # cnums = [{'cnums':list(range(nclusterlist[x]))} for x in range(len(nclusterlist))]
            prep_data_sem_physio_model_SO_FC(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                      cnums = cnums, run_whole_group=False, normalizevar=normalizevar, filter_tcdata = True)
            # use all clusters for fully-connected model
            print('nclusterlist = {}'.format(nclusterlist))
            # print('cnums = {}'.format(cnums))
        else:
            print('networkfile = {}'.format(networkfile))
            print('regiondataname = {}'.format(regiondataname))
            print('clusterdataname = {}'.format(clusterdataname))
            print('SAPMparametersname = {}'.format(SAPMparametersname))
            print('timepoint = {}'.format(timepoint))
            print('epoch = {}'.format(epoch))
            prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                      run_whole_group=False, normalizevar=normalizevar, filter_tcdata = False)

            print('nclusterlist = {}'.format(nclusterlist))
            # print('cnums = {}'.format(cnums))

    pcamethod = False
    if pcamethod:
        npc_components = len(nclusterlist) + 1
        output = sem_physio_model1_pca(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                                       fixed_beta_vals=[], betascale=betascale, alphascale=alphascale, Lweight=Lweight,
                                       nitermax_stage3=leveliter[2], nitermax_stage2=leveliter[1],
                                       nitermax_stage1=leveliter[0], nsteps_stage2=leveltrials[1],
                                       nsteps_stage1=leveltrials[0], levelthreshold=levelthreshold,
                                       run_whole_group=run_whole_group,
                                       resumerun=resumerun, npc=npc_components, verbose=verbose,
                                       silentrunning=silentrunning)
    else:
        output = sem_physio_model1_V5(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                                      fixed_beta_vals=[], betascale=betascale, alphascale = alphascale, Lweight=Lweight,
                                      nitermax_stage3 = leveliter[2], nitermax_stage2 = leveliter[1],
                                      nitermax_stage1 = leveliter[0], nsteps_stage2 = leveltrials[1],
                                      nsteps_stage1 = leveltrials[0], levelthreshold=levelthreshold, run_whole_group = run_whole_group,
                                      resumerun = resumerun, verbose = verbose, silentrunning = silentrunning)

    # now, correct the results for normalizing the variance
    if normalizevar:
        output = sem_physio_correct_for_normalization(SAPMresultsname, SAPMparametersname, verbose = True)
        # output2 = sem_physio_correct_for_normalization2(SAPMresultsname, SAPMparametersname, verbose = True)

    SAPMresults = np.load(output, allow_pickle=True)
    NP = len(SAPMresults)
    print('NP = {}'.format(NP))
    R2list =np.zeros(len(SAPMresults))
    R2list2 =np.zeros(len(SAPMresults))
    if pcamethod:
        pc_R2list =np.zeros(len(SAPMresults))
        pc_R2list2 =np.zeros(len(SAPMresults))

    for nperson in range(NP):
        R2list[nperson] = SAPMresults[nperson]['R2avg']
        R2list2[nperson] = SAPMresults[nperson]['R2total']
        # the next lines only when sem_physio_model1_pca is used
    if pcamethod:
        for nperson in range(NP):
            pc_R2list[nperson] = SAPMresults[nperson]['pc_R2avg']
            pc_R2list2[nperson] = SAPMresults[nperson]['pc_R2total']
    print('SAPM parameters computed for {} data sets'.format(NP))
    print('R2 values averaged {:.3f} {} {:.3f}'.format(np.mean(R2list),chr(177),np.std(R2list)))
    print('average R2 values ranged from {:.3f} to {:.3f}'.format(np.min(R2list),np.max(R2list)))
    print('Total R2 values were {:.3f} {} {:.3f}'.format(np.mean(R2list2),chr(177),np.std(R2list2)))
    print('Total R2 values ranged from {:.3f} to {:.3f}'.format(np.min(R2list2),np.max(R2list2)))
    if pcamethod:
        print('Total pc R2 values were {:.3f} {} {:.3f}'.format(np.mean(pc_R2list2),chr(177),np.std(pc_R2list2)))
        print('Total pc R2 values ranged from {:.3f} to {:.3f}'.format(np.min(pc_R2list2),np.max(pc_R2list2)))


def SAPMprune_fc_network(cnums, regiondataname, clusterdataname, SAPMresultsname_previous, SAPMparametersname_previous, networkfile,
            nametag = '', timepoint = 'all', epoch = 'all', Lweight = 0.01, alphascale = 0.01, leveltrials = [30, 4, 1],
            leveliter = [100, 250, 1200], levelthreshold = [1e-4, 1e-5, 1e-6], run_whole_group = False, verbose = True,
            silentrunning = False):

    SAPMresults_previous = np.load(SAPMresultsname_previous, allow_pickle=True)
    SAPMparams_previous = np.load(SAPMparametersname_previous, allow_pickle=True).flat[0]
    cnums_previous = copy.deepcopy(SAPMparams_previous['cnums'])

    p,f1 = os.path.split(SAPMresultsname_previous)
    f,e = os.path.splitext(f1)
    index = SAPMresultsname_previous.find('_results')

    SAPMresultsname = SAPMresultsname_previous[:index] + nametag + '_pruned_results.npy'
    SAPMparametersname = SAPMresultsname_previous[:index] + nametag + '_pruned_params.npy'
    newbetainitname = SAPMresultsname_previous[:index] + nametag + '_betainit.npy'
    print('name of data file containing initial beta values is set to {}'.format(newbetainitname))

    # re-run a prior analysis with some regions removed ...
    fully_connected = True

    # map results from previous clusters to new clusters ----------------------------------
    #--------------------------------------------------------------------------------------
    # need to find each connectivity value in the pruned model within the previous model to get the values
    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    # nclusterlist refers to the number of clusters in the data, not in the network model
    nclusterlist = np.array([nclusterdict[x]['nclusters'] for x in range(len(nclusterdict))])
    Nintrinsic = fintrinsic_count + vintrinsic_count

    # create names for the previous regions/clusters
    # rnum_previous = [x for x in range(len(cnums_previous)) for x2 in range(len(cnums_previous[x]['cnums']))]
    # region_name_list_previous = []
    # nregions = len(rnamelist)
    # for x in range(len(cnums_previous)):
    #     if rnum_previous[x] >= nregions:
    #         lnum = rnum_previous[x] - nregions
    #         region_name = 'latent{}'.format(lnum)
    #     else:
    #         region_name = '{}_{}'.format(rnamelist[rnum_previous[x]],cnums_previous[x])
    #     region_name_list_previous += [region_name]

    # regiondata = np.load(regiondataname, allow_pickle=True).flat[0]
    region_data = load_filtered_regiondata(regiondataname, networkfile)

    region_properties = regiondata['region_properties']
    rnamelist = [region_properties[xx]['rname'] for xx in range(len(region_properties))]

    region_name_list_previous = []
    nregions = len(rnamelist)
    for x in range(len(cnums_previous)):
        for x2 in range(len(cnums_previous[x]['cnums'])):
            region_name = '{}_{}'.format(rnamelist[x],cnums_previous[x]['cnums'][x2])
            region_name_list_previous += [region_name]
    for x in range(Nintrinsic):
        region_name = 'latent{}'.format(x)
        region_name_list_previous += [region_name]

    region_name_list = []
    for x in range(len(cnums)):
        for x2 in range(len(cnums[x]['cnums'])):
            region_name = '{}_{}'.format(rnamelist[x],cnums[x]['cnums'][x2])
            region_name_list += [region_name]
    for x in range(Nintrinsic):
        region_name = 'latent{}'.format(x)
        region_name_list += [region_name]

    # map old network to new network
    mapping = []
    previous_num_list = []
    for x in range(len(region_name_list)):
        oldnum = region_name_list_previous.index(region_name_list[x])
        mapping.append({'newnum':x, 'oldnum':oldnum})
        previous_num_list += [oldnum]

    beta_list_previous = copy.deepcopy(SAPMparams_previous['beta_list'])
    beta_list_new = []
    beta_list_count = 0
    map_previous_to_new = []
    for x in range(len(beta_list_previous)):
        pair = copy.deepcopy(beta_list_previous[x]['pair'])
        try:
            cc1 = previous_num_list.index(pair[0])
            cc2 = previous_num_list.index(pair[1])
            name = '{}_{}'.format(cc1,cc2)
            beta_list_new.append({'name':name, 'number':beta_list_count,'pair':[cc1,cc2], 'previous_index':x})
            beta_list_count += 1
            map_previous_to_new += [x]
        except:
            print('{}-{} not in new network'.format(region_name_list_previous[pair[1]], region_name_list_previous[pair[0]]))

    NP = len(SAPMresults_previous)
    nbeta = len(beta_list_new)
    betavals_init = np.zeros((nbeta,NP))
    for nn in range(NP):
        betavals_previous = copy.deepcopy(SAPMresults_previous[nn]['betavals'])
        betavals_init[:,nn] = betavals_previous[map_previous_to_new]

    betavals = {'beta_initial':betavals_init}
    # starting DB, D values saved in file passed in as "betascale"
    np.save(newbetainitname, betavals)
    betascale = copy.deepcopy(newbetainitname)

    # load some data, setup some parameters...
    # here, nclusterlist refers to the number of clusters in the data, not that are used in the network model
    # network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count, fintrinsic_base = load_network_model_w_intrinsics(networkfile)
    full_rnum_base = [np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]

    # starting values
    cnums_original = copy.deepcopy(cnums)
    excelsheetname = 'clusters'

    # run the analysis with SAPM
    # prep the data
    prep_data_sem_physio_model_SO_FC(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                              cnums = cnums, run_whole_group=False, normalizevar=False, filter_tcdata = False)
    # print('cnums = {}'.format(cnums))

    output = sem_physio_model1_V4(cnums, fintrinsic_base, SAPMresultsname, SAPMparametersname,
                                  fixed_beta_vals=[], betascale=newbetainitname, alphascale = alphascale, Lweight=Lweight,
                                  nitermax_stage3 = leveliter[2], nitermax_stage2 = leveliter[1],
                                  nitermax_stage1 = leveliter[0], nsteps_stage2 = leveltrials[1],
                                  nsteps_stage1 = leveltrials[0], levelthreshold=levelthreshold, run_whole_group = run_whole_group,
                                  resumerun = False, verbose = verbose, silentrunning = silentrunning)

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
                               fully_connected = False, cnums = []):

    # print('plot_region_inputs_average ...')
    # print('target = {}'.format(target))

    if isinstance(TargetCanvas,str):
        display_in_GUI = False
    else:
        display_in_GUI = True

    Zthresh = stats.norm.ppf(1-np.array([1.0, 0.05,0.01,0.001]))
    symbollist = [' ','*', chr(8868),chr(8903)]

    nclusters_to_use, nclusters_total = np.shape(Minput)

    # if fully_connected:
    #     nclusterstotal = int(np.sum(nclusterlist))
    #     nclusteroffset = np.array([0] + list(np.cumsum(nclusterlist))).astype(int)

    nregions = len(rnamelist)
    rnamelist_full = copy.deepcopy(rnamelist)
    if fully_connected:
        rnamelist_temp = []
        for x in range(len(cnums)):
            for x2 in cnums[x]['cnums']:
                rnamelist_temp += ['{}{}'.format(rnamelist[x],x2)]
        rnamelist_full = copy.deepcopy(rnamelist_temp)

    Nlatent = nclusters_total - nclusters_to_use
    for nn in range(Nlatent): rnamelist_full += ['latent{}'.format(nn)]

    regionnum = rnamelist_full.index(target)  # input a region

    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
        print('yrange set to {} to {}'.format(ymin,ymax))
    else:
        setylim = False

    # if fully_connected:
    #     tt_base = [x for x in range(nregions) if target >= nclusteroffset[x]][-1]
    #     ctarget = target - nclusteroffset[tt_base]
    #     rtarget = '{}{}'.format(rnamelist[tt_base],ctarget)
    # else:
    #     rtarget = rnamelist.index(target)

    m = Minput[regionnum, :]
    sources = np.where(m != 0)[0]
    # rsources = [beta_list[ss]['pair'][0] for ss in sources]
    nsources = len(sources)
    checkdims = np.shape(Sinput_avg)
    if np.ndim(Sinput_avg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    # get beta values from Mconn
    if fully_connected:
        textlist = []
        for ss in sources:
            beta = Mconn_avg[regionnum, ss]
            valtext = '{:.3f} '.format(beta)
            text = '{} {}'.format(rnamelist_full[ss], valtext)
            # if ss >= nclusterstotal:
            #     text = 'int{} {}'.format(ss - nclusterstotal, valtext)
            # else:
            #     ts_base = [x for x in range(nregions) if ss >= nclusteroffset[x]][-1]
            #     csource = ss - nclusteroffset[ts_base]
            #     rsource = '{}{}'.format(rnamelist[ts_base], csource)
            #     text = '{} {}'.format(rsource, valtext)
            textlist += [text]
    else:
        textlist = []
        for ss in sources:
            beta = Mconn_avg[regionnum, ss]
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
    tc1 = Sinput_avg[regionnum,:]
    tc1p = Sinput_sem[regionnum,:]
    tc1f = fit_avg[regionnum,:]
    tc1fp = fit_sem[regionnum,:]

    y1 = list(tc1f+tc1fp)
    y2 = list(tc1f-tc1fp)
    yy = y1 + y2[::-1]
    axs[0,1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
    axs[0,1].plot(x, tc1f, '-xr', linewidth=1, markersize=4)
    axs[0,1].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
    axs[0,1].plot(x, tc1f+tc1fp, color = (1,0,0), linestyle = '-', linewidth = 0.5)
    axs[0,1].set_title('target input {}'.format(rnamelist_full[regionnum]))
    # ymax = np.max(np.abs(yy))

    # if not fully_connected:
    tc1 = Sconn_avg[regionnum,:]
    tc1p = Sconn_sem[regionnum,:]

    y1 = list(tc1+tc1p)
    y2 = list(tc1-tc1p)
    yy = y1 + y2[::-1]
    axs[1,1].plot(x, tc1, '-ob', linewidth=1, markersize=4)
    axs[1,1].fill(xx,yy, facecolor=(0,0,1), edgecolor='None', alpha = 0.2)
    axs[1,1].set_title('target output {}'.format(rnamelist_full[regionnum]))

    for ss in range(nsources):
        tc1 = Sconn_avg[sources[ss], :]
        tc1p = Sconn_sem[sources[ss], :]
        y1 = list(tc1 + tc1p)
        y2 = list(tc1 - tc1p)
        yy = y1 + y2[::-1]
        axs[ss,0].plot(x, tc1, '-xr')
        axs[ss,0].fill(xx,yy, facecolor=(1,0,0), edgecolor='None', alpha = 0.2)
        axs[ss,0].plot(x, tc1+tc1p, color = (1,0,0), linestyle = '-', linewidth = 0.5)
        if fully_connected:
            if sources[ss] >= nclusters_to_use:
                axs[ss, 0].set_title('source latent {}'.format(sources[ss]-nclusters_to_use))
            else:
                axs[ss,0].set_title('source output {}'.format(rnamelist_full[sources[ss]]))
            axs[ss,0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                    horizontalalignment='left', verticalalignment='bottom', fontsize=10)
        else:
            if sources[ss] >= nregions:
                axs[ss, 0].set_title('source latent {}'.format(sources[ss]-nregions))
            else:
                axs[ss,0].set_title('source output {}'.format(rnamelist_full[sources[ss]]))
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
                                  fully_connected = False, cnums = []):
    # print('plot_region_inputs_regression ...')
    # print('target = {}'.format(target))
    #
    # print('betanamelist = {}'.format(betanamelist))
    # print('beta_list = {}'.format(beta_list))
    # print('rnamelist = {}'.format(rnamelist))
    # print('size of Minput = {}'.format(np.shape(Minput)))
    # print('size of Sinput_reg = {}'.format(np.shape(Sinput_reg)))
    # print('size of Sconn_reg = {}'.format(np.shape(Sconn_reg)))

    nclusters_to_use, nclusters_total = np.shape(Minput)

    if isinstance(TargetCanvas,str):
        display_in_GUI = False
    else:
        display_in_GUI = True

    Zthresh = stats.norm.ppf(1-np.array([1.0, 0.05,0.01,0.001]))
    symbollist = [' ','*', chr(8868),chr(8903)]

    nregions = len(rnamelist)
    rnamelist_full = copy.deepcopy(rnamelist)
    if fully_connected:
        rnamelist_temp = []
        for x in range(len(cnums)):
            for x2 in cnums[x]['cnums']:
                rnamelist_temp += ['{}{}'.format(rnamelist[x],x2)]
        rnamelist_full = copy.deepcopy(rnamelist_temp)

    Nlatent = nclusters_total - nclusters_to_use
    for nn in range(Nlatent): rnamelist_full += ['latent{}'.format(nn)]

    regionnum = rnamelist_full.index(target)   # input a region

    if len(yrange) > 0:
        setylim = True
        ymin = yrange[0]
        ymax = yrange[1]
    else:
        setylim = False

    # if fully_connected:
    #     tt_base = [x for x in range(nregions) if target >= nclusteroffset[x]][-1]
    #     ctarget = target - nclusteroffset[tt_base]
    #     rtarget = '{}{}'.format(rnamelist[tt_base],ctarget)
    # else:
    #     rtarget = rnamelist[target]

    m = Minput[regionnum, :]
    sources = np.where(m != 0)[0]
    nsources = len(sources)
    # rsources = [beta_list[ss]['pair'][0] for ss in sources]
    checkdims = np.shape(Sinput_reg)
    if np.ndim(Sinput_reg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    if fully_connected:
        # textlist = []
        textlist = [rnamelist_full[ss] for ss in sources]
        # for ss in sources:
        #     if ss >= nclusters_to_use:
        #         text = 'int{}'.format(ss - nclusters_to_use)
        #     else:
        #         # ts_base = [x for x in range(nregions) if ss >= nclusteroffset[x]][-1]
        #         # csource = ss - nclusteroffset[ts_base]
        #         # text = '{}{}'.format(rnamelist_full[ts_base], csource)
        #         text = rnamelist_full[ss]
        #     textlist += [text]

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
    tc1 = Sinput_reg[regionnum,:,0]
    tc1p = Sinput_reg[regionnum,:,1]
    tc1f = fit_reg[regionnum,:,0]
    tc1fp = fit_reg[regionnum,:,1]

    Z1 = Sinput_reg[regionnum,:,3]
    Z1f = fit_reg[regionnum,:,3]

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
    axs[1,1].set_title('target input {}'.format(rnamelist_full[regionnum]))

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

        # print('\n\nsources = {}'.format(sources))
        # print('rsources = {}\n\n'.format(rsources))

        if fully_connected:
            if sources[ss] >= nclusters_to_use:
                axs[ss, 0].set_title('source latent {}'.format(sources[ss] - nclusters_to_use))
            else:
                axs[ss,0].set_title('source output {}'.format(rnamelist_full[sources[ss]]))
            axs[ss,0].annotate(textlist[ss], xy=(.025, .025), xycoords='axes  fraction',
                    horizontalalignment='left', verticalalignment='bottom', fontsize=10)
        else:
            if sources[ss] >= nregions:
                axs[ss, 0].set_title('source latent {}'.format(sources[ss] - nregions))
            else:
                axs[ss, 0].set_title('source output {}'.format(rnamelist_full[sources[ss]]))
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


def plot_region_fits(window, regionlist, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist, outputdir, yrange = [], TargetCanvas = 'none', color1 = 'b', color2 = 'r'):  # display_in_GUI = False

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

    print('size of Sinput_avg = {}'.format(np.shape(Sinput_avg)))
    print('size of fit_avg = {}'.format(np.shape(fit_avg)))
    print('rnamelist = {}'.format(rnamelist))
    print('regionlist = {}'.format(regionlist))
    print('ndisplay = {}'.format(ndisplay))

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
            axs[nn].errorbar(t, tc1, tc1_sem, marker = 'o', markerfacecolor = color1, markeredgecolor = color1, linestyle = '-', color = color1, linewidth=1, markersize=4)
        else:
            axs[nn].plot(t, tc1, marker = 'o', markerfacecolor = color1, markeredgecolor = color1, linestyle = '-', color = color1, linewidth=1, markersize=4)

        if len(fit_avg) > 0:
            if len(fit_sem) > 0:
                tcf1_sem = fit_sem[regionlist[nn], :]
                axs[nn].errorbar(t, tcf1, tcf1_sem, marker = 'x', markerfacecolor = color2, markeredgecolor = color2, linestyle = '-', color = color2, linewidth=1, markersize=4)
            else:
                axs[nn].plot(t, tcf1, marker = 'x', markerfacecolor = color2, markeredgecolor = color2, linestyle = '-', color = color2, linewidth=1, markersize=4)

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
                        sigflag = [], fully_connected = False, cnums = []):
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

    if fully_connected:
        nclusters_used = [len(cnums[x]['cnums']) for x in range(len(cnums))]
        nclusterstotal = int(np.sum(nclusters_used))
        nclusteroffset = np.array([0] + list(np.cumsum(nclusters_used))).astype(int)
        labeltext_record = []
        valuetext_record = []
        Ttext_record = []
        T_record = []
        reftext_record = []
        for n1 in range(len(beta_list)):
            tname = betanamelist[n1]
            tpair = beta_list[n1]['pair']
            if tpair[0] >= nclusterstotal:
                ts = 'int{}'.format(tpair[0]-nclusterstotal)
            else:
                t1 = copy.deepcopy(tpair[0])
                ts_base = [x for x in range(nregions) if t1 >= nclusteroffset[x]][-1]
                cs = t1 - nclusteroffset[ts_base]
                sname_base = copy.deepcopy(rnamelist[ts_base])
                if len(sname_base) > 4:  sname_base = sname_base[:4]
                ts = '{}{}'.format(sname_base,cnums[ts_base]['cnums'][cs])

            t2 = copy.deepcopy(tpair[1])
            tt_base = [x for x in range(nregions) if t2 >= nclusteroffset[x]][-1]
            ct = t2 - nclusteroffset[tt_base]
            tname_base = copy.deepcopy(rnamelist[tt_base])
            if len(tname_base) > 4:  tname_base = tname_base[:4]
            tt = '{}{}'.format(tname_base, cnums[tt_base]['cnums'][ct])

            #----------temp-------------------------------
            labeltext = '{}-{}'.format(ts, tt)
            print('n1 = {}   labeltext = {}'.format(n1,labeltext))
            #----------end of temp------------------------


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
                      sigflag = [], fully_connected = False, cnums = []):

    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mslope)

    if np.size(sigflag) == 0:
        sigflag = np.zeros(np.shape(Mslope))

    labeltext_record = []
    inttext_record = []
    slopetext_record = []
    R2text_record = []
    R2_record = []

    if fully_connected:
        nclusters_used = [len(cnums[x]['cnums']) for x in range(len(cnums))]
        nclusterstotal = int(np.sum(nclusters_used))
        nclusteroffset = np.array([0] + list(np.cumsum(nclusters_used))).astype(int)

    for n1 in range(nr1):
        if fully_connected:
            if n1 >= nclusterstotal:
                tt = 'int{}'.format(n1-nclusterstotal)
            else:
                tt_base = [x for x in range(nregions) if n1 >= nclusteroffset[x]][-1]
                ct = n1 - nclusteroffset[tt_base]
                tname_base = copy.deepcopy(rnamelist[tt_base])
                if len(tname_base) > 4:  tname_base = tname_base[:4]
                tt = '{}{}'.format(tname_base,cnums[tt_base]['cnums'][ct])
            showval = False

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
                if fully_connected:
                    if n2 >= nclusterstotal:
                        ss = 'int{}'.format(n2 - nclusterstotal)
                    else:
                        ss_base = [x for x in range(nregions) if n2 >= nclusteroffset[x]][-1]
                        cs = n2 - nclusteroffset[ss_base]
                        sname_base = copy.deepcopy(rnamelist[ss_base])
                        if len(sname_base) > 4:  sname_base = sname_base[:4]
                        ss = '{}{}'.format(sname_base, cnums[ss_base]['cnums'][cs])
                else:
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


def display_anatomical_cluster(clusterdataname, networkfile, targetnum, targetcluster, orientation = 'axial', regioncolor = [0,1,1], templatename = 'ccbs', write_output = False):
    # get the voxel coordinates for the target region
    clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]
    # cluster_properties = clusterdata['cluster_properties']
    cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)

    template_img = clusterdata['template_img']
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
    if hasattr(targetcluster, '__len__'):
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

    # # load template
    # if templatename.lower() == 'brain':
    #     resolution = 2
    # else:
    #     resolution = 1
    # template_img, regionmap_img, template_affine, anatlabels, wmmap, roi_map, gmwm_img = load_templates.load_template_and_masks(
    #     templatename, resolution)

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



def betavalue_labels_SO(csource, ctarget, rnamelist):
    # labels for network models when each region only has one output (but it can be to multiple regions)

    labeltext_record = []
    nregions = len(rnamelist)
    nbeta = len(csource)
    for nn in range(nbeta):
        tt = ctarget[nn]
        ss = csource[nn]

        targetname = rnamelist[tt]
        if ss >= nregions:
            sourcename = 'latent{}'.format(ss-nregions)
        else:
            sourcename = rnamelist[ss]
        labeltext = '{}-{}'.format(sourcename, targetname)

        labeltext_record += [labeltext]

    return labeltext_record



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



def display_SAPM_results(window, outputnametag, covariates, covnametag, outputtype, outputdir, SAPMparametersname, SAPMresultsname,
                         group, target = '', pthresh = 0.05, SAPMstatsfile = '', setylimits = [], TargetCanvas = [],
                         display_in_GUI = False, fully_connected = False,
                         SRresultsname2 = '', SRparametersname2 = '', covariates2 = []):

    # options of results to display:
    # 1) average input time-courses compared with model input
    # 2) modelled input signaling with corresponding source time-courses (outputs from source regions)
    # 3) t-test comparisons between groups, or w.r.t. zero (outputs to excel files)
    # 4) regression with continuous covariate
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram', 'Paired_diff']

    displayrecordname = os.path.join(outputdir, 'SAPM_FC_output_params.npy')
    if os.path.isfile(displayrecordname):
        ddata = np.load(displayrecordname, allow_pickle=True).flat[0]
        displayclusternumber = copy.deepcopy(ddata['clusternumber'])
    else:
        ddata = {'clusternumber':0}
        np.save(displayrecordname, ddata)

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

    nregions = len(rnamelist)
    # load the SEM results
    SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
    try:
        cnums = copy.deepcopy(SAPMresults_load[0]['cnums'])
    except:
        cnums = [{'cnums':[]} for x in range(nregions)]

    nclustersused = np.sum(np.array([len(cnums[x]['cnums']) for x in range(len(cnums))])).astype(int)
    print('nclustersused = {}'.format(nclustersused) )

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

        keylist = SAPMresults_load[0].keys()
        if 'cnums' in keylist:
            cnums = copy.deepcopy(SAPMresults_load[0]['cnums'])
        else:
            cnums = []
        # print('cnums = {}'.format(cnums))

        if (NP == 1) & (len(nruns_per_person) > 1):
            nruns = np.sum(nruns_per_person)   # analysis of whole group together
        else:
            nruns = nruns_per_person[nperson]

        if fintrinsic_count > 0:
            fintrinsic1 = np.array(list(ftemp) * nruns)

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
            # p -= np.mean(p)   # this shifts the intercept value to the mean covariate value
            G = np.concatenate((np.ones((1, len(g1))),p), axis=0) # put the intercept term first
        else:
            continuouscov = False
    else:
        continuouscov = False

    #-------------------------------------------------------------------------------------
    # significance of average Mconn values -----------------------------------------------
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram', 'Paired_diff']
    if NP > 1:
        DB_avg = np.mean(DBrecord[:, :, g1], axis=2)
        DB_sem = np.std(DBrecord[:, :, g1], axis=2) / np.sqrt(len(g1))

        D_avg = np.mean(Drecord[:, :, g1], axis=2)
        D_sem = np.std(Drecord[:, :, g1], axis=2) / np.sqrt(len(g1))

        B_avg = np.mean(Brecord[:, :, g1], axis=2)
        B_sem = np.std(Brecord[:, :, g1], axis=2) / np.sqrt(len(g1))
    else:
        DB_avg = copy.deepcopy(DBrecord[:,:,0])
        DB_sem = 1e-5*np.ones(np.shape(DB_avg))

        D_avg = copy.deepcopy(Drecord[:,:,0])
        D_sem = 1e-5*np.ones(np.shape(D_avg))

        B_avg = copy.deepcopy(Brecord[:,:,0])
        B_sem = 1e-5*np.ones(np.shape(B_avg))


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

            if (NP2 == 1) & (len(nruns_per_person2) > 1):
                nruns = np.sum(nruns_per_person2)  # analysis of whole group together
            else:
                nruns = nruns_per_person2[nperson]

            if fintrinsic_count > 0:
                fintrinsic1 = np.array(list(ftemp) * nruns)

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
        if NP2 > 1:
            DB_avg2 = np.mean(DBrecord2, axis=2)
            DB_sem2 = np.std(DBrecord2, axis=2) / np.sqrt(NP2)

            D_avg2 = np.mean(Drecord2, axis=2)
            D_sem2 = np.std(Drecord2, axis=2) / np.sqrt(NP2)

            B_avg2 = np.mean(Brecord2, axis=2)
            B_sem2 = np.std(Brecord2, axis=2) / np.sqrt(NP2)
        else:
            DB_avg2 = copy.deepcopy(DBrecord2[:,:,0])
            DB_sem2 = 1e-5*np.ones(np.shape(DB_avg2))

            D_avg2 = copy.deepcopy(Drecord2[:,:,0])
            D_sem2 = 1e-5*np.ones(np.shape(D_avg2))

            B_avg2 = copy.deepcopy(Brecord2[:,:,0])
            B_sem2 = 1e-5*np.ones(np.shape(B_avg2))


    if outputtype == 'B_Significance':
        # significant B values-------------------------------------
        descriptor = outputnametag + '_Bsig'
        print('\n\nAverage B values')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(B_avg, B_sem, NP, betanamelist, rnamelist,
                                                beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                fully_connected=fully_connected, cnums = cnums)

        pthresh_list = ['{:.3e}'.format(pthresh)]*len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)]*len(Ttext)

        Rtextlist = [' ']*10
        Rvallist = [0]*10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        # si2 = np.where(T < 1e10)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si], 'B': np.array(valuetext)[si], 'T': np.array(Ttext)[si],
                       'T thresh': np.array(Tthresh_list)[si], 'p thresh': np.array(pthresh_list)[si], 'stat ref': np.array(reftext)[si]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Bsig')

        # significant D values-------------------------------------
        descriptor = outputnametag + '_Dsig'
        print('\n\nAverage D values')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(D_avg, D_sem, NP, betanamelist, rnamelist,
                                                beta_list, format='f', pthresh=pthresh, statsref = '', fully_connected=fully_connected, cnums = cnums)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        # si2 = np.where(T < 1e3)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si], 'D': np.array(valuetext)[si],
                       'T': np.array(Ttext)[si],
                       'T thresh': np.array(Tthresh_list)[si], 'p thresh': np.array(pthresh_list)[si], 'stat ref': np.array(reftext)[si]}
        df = pd.DataFrame(textoutputs)

        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='Dsig')


        # significant DB values-------------------------------------
        descriptor = outputnametag + '_DBsig'
        print('\n\nAverage DB values')
        labeltext, valuetext, Ttext, T, Tthresh, reftext = write_Mconn_values2(DB_avg, DB_sem, NP, betanamelist, rnamelist,
                                                beta_list, format='f', pthresh=pthresh, statsref = DBref_mean, fully_connected=fully_connected, cnums = cnums)

        pthresh_list = ['{:.3e}'.format(pthresh)] * len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)] * len(Ttext)

        Rtextlist = [' '] * 10
        Rvallist = [0] * 10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        # si2 = np.where(T < 1e3)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si], 'DB': np.array(valuetext)[si],
                       'T': np.array(Ttext)[si],
                       'T thresh': np.array(Tthresh_list)[si], 'p thresh': np.array(pthresh_list)[si], 'stat ref': np.array(reftext)[si]}
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
        if len(covnametag) > 0:
            cname = '_' + covnametag
        else:
            cname = ''
        descriptor = outputnametag + cname + '_Breg'
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
                                            R2thresh=R2thresh, fully_connected = fully_connected, cnums = cnums)

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
        if len(covnametag) > 0:
            cname = '_' + covnametag
        else:
            cname = ''
        descriptor = outputnametag + cname + '_Dreg'
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
                                            beta_list, format='f', R2thresh=R2thresh, fully_connected=fully_connected, cnums = cnums)

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
        if len(covnametag) > 0:
            cname = '_' + covnametag
        else:
            cname = ''
        descriptor = outputnametag + cname + '_DBreg'
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
                                            beta_list, format='f', R2thresh=R2thresh, fully_connected=fully_connected, cnums = cnums)

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

        if fully_connected:
            nclusterstotal = int(np.sum(nclusterlist))
            nclusteroffset = np.array([0] + list(np.cumsum(nclusterlist))).astype(int)

            ddata = np.load(displayrecordname, allow_pickle=True).flat[0]
            print('displayclusternumber = {}'.format(displayclusternumber))
            if regionnum[0] >= nregions: # latent input
                displayclusternumber = 0
                ddata = {'clusternumber': displayclusternumber}  # only increment the number if the results are displayed
                np.save(displayrecordname, ddata)
            else:
                if display_in_GUI:
                    displayclusternumber += 1
                    print('nclusterlist = {}'.format(nclusterlist))
                    print('regionnum = {}'.format(regionnum[0]))

                    if displayclusternumber >= len(cnums[regionnum[0]]['cnums']): displayclusternumber = 0
                    ddata = {'clusternumber': displayclusternumber}  # only increment the number if the results are displayed
                    np.save(displayrecordname, ddata)
                # update target name
                displayclusternumber = copy.deepcopy(ddata['clusternumber'])
                actualclusternumber = cnums[regionnum[0]]['cnums'][displayclusternumber]
                target = '{}{}'.format(target, actualclusternumber)

            # rnamelist_full = []
            # for ccc in range(nclusterstotal):
            #     rr = [x for x in range(nregions) if ccc >= nclusteroffset[x]][-1]
            #     cr = ccc - nclusteroffset[rr]
            #     rnamelist_full += ['{}{}'.format(rnamelist[rr], cr)]

            rnamelist_full = []
            for rr in range(len(cnums)):
                for cr in cnums[rr]['cnums']:
                    rnamelist_full += ['{}{}'.format(rnamelist[rr], cr)]

            if fintrinsic_count > 0: rnamelist_full += ['latent0']
            for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count + nn)]

        regionnum = [rnamelist_full.index(target) ]   # input a region
        nametag = target + '_' + outputnametag   # create name for saving figure
        print('Plotting Sinput data for region {}, number {}'.format(target, regionnum))
        print('rnamelist_full = {}'.format(rnamelist_full))

        if len(setylimits) > 0:
            ylim = setylimits[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []

        if fully_connected:
            print('nclusterstotal = {}   nclustersused = {}'.format(nclusterstotal, nclustersused))
            if regionnum[0] >= nclustersused:   # latent input
                latentnum = regionnum[0] - nclustersused
                svgname, Rtext, Rvals = plot_region_fits(window, [latentnum], nametag, Mintrinsic_avg, Mintrinsic_sem, [], [], rnamelist_full[nclustersused:], outputdir, yrangethis, TargetCanvas) # display_in_GUI
            else:
                svgname, Rtext, Rvals = plot_region_fits(window, regionnum, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist_full, outputdir, yrangethis, TargetCanvas) # display_in_GUI
        else:
            if regionnum[0] >= nregions:   # latent input
                latentnum = regionnum[0] - nregions
                svgname, Rtext, Rvals = plot_region_fits(window, [latentnum], nametag, Mintrinsic_avg, Mintrinsic_sem, [], [], rnamelist_full[nregions:], outputdir, yrangethis, TargetCanvas) # display_in_GUI
            else:
                svgname, Rtext, Rvals = plot_region_fits(window, regionnum, nametag, Sinput_avg, Sinput_sem, fit_avg, fit_sem, rnamelist_full, outputdir, yrangethis, TargetCanvas) # display_in_GUI

        outputname = svgname

        print('finished generating results for Plot_BOLDModel...')
        return outputname


    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_Output', 'Plot_SourceModel', 'DrawSAPMdiagram']
    if outputtype == 'Plot_Output':
        print('generating results for Plot_Output...')
        descriptor = outputnametag + '_Output'

        nregions = len(rnamelist)
        rnamelist_full = copy.deepcopy(rnamelist)
        if fintrinsic_count > 0: rnamelist_full += ['latent0']
        for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count + nn)]

        regionnum = [rnamelist_full.index(target) ]   # input a region
        nametag = rnamelist_full[regionnum[0]] + '_' + outputnametag   # create name for saving figure
        nametag1 = target + '_' + descriptor
        print('Plotting Sinput data for region {}, number {}'.format(target, regionnum))

        if len(setylimits) > 0:
            ylim = setylimits[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []
        if regionnum[0] >= nregions:   # latent input
            latentnum = regionnum[0] - nregions
            svgname, Rtext, Rvals = plot_region_fits(window, [latentnum], nametag1, Mintrinsic_avg, Mintrinsic_sem, [], [], rnamelist_full[nregions:], outputdir, yrangethis, TargetCanvas, color1 = 'r') # display_in_GUI
        else:
            if fully_connected:
                ddata = np.load(displayrecordname, allow_pickle=True).flat[0]
                displayclusternumber = copy.deepcopy(ddata['clusternumber'])
                actualclusternumber = cnums[regionnum[0]]['cnums'][displayclusternumber]

                nclusteroffset = np.array([0] + list(np.cumsum(nclusterlist))).astype(int)
                regionnum_fc = displayclusternumber + nclusteroffset[regionnum[0]]
                nametag_fc = '{}{}'.format(rnamelist_full[regionnum[0]],actualclusternumber) + '_' + outputnametag
                svgname, Rtext, Rvals = plot_region_fits(window, [regionnum_fc], nametag_fc, Sinput_avg, Sinput_sem, [], [], rnamelist, outputdir, yrangethis, TargetCanvas, color1 = 'r') # display_in_GUI
            else:
                svgname, Rtext, Rvals = plot_region_fits(window, regionnum, nametag1, Sconn_avg, Sconn_sem, [], [], rnamelist, outputdir, yrangethis, TargetCanvas, color1 = 'r') # display_in_GUI
        outputname = svgname

        print('finished generating results for Plot_Output...')
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

        if len(setylimits) > 0:
            ylim = setylimits[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []

        # nregions = len(rnamelist)
        # rnamelist_full = copy.deepcopy(rnamelist)
        # if fintrinsic_count > 0: rnamelist_full += ['latent0']
        # for nn in range(vintrinsic_count): rnamelist_full += ['latent{}'.format(fintrinsic_count + nn)]
        # regionnum = rnamelist_full.index(target)   # input a region
        if fully_connected:
            regionnum = rnamelist.index(target)
            ddata = np.load(displayrecordname, allow_pickle=True).flat[0]
            displayclusternumber = copy.deepcopy(ddata['clusternumber'])
            actualclusternumber = cnums[regionnum]['cnums'][displayclusternumber]
            target = '{}{}'.format(target, actualclusternumber)

        nametag1 = target + '_' + descriptor
        if continuouscov:
            outputname = plot_region_inputs_regression(window, target, nametag1, Minput, Sinput_reg, fit_reg, Sconn_reg,
                                beta_list, rnamelist, betanamelist, DB_avg, outputdir, yrangethis, TargetCanvas, fully_connected, cnums = cnums)

        outputname = plot_region_inputs_average(window, target, nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                                   Sconn_sem, beta_list, rnamelist, betanamelist, DB_avg, outputdir, yrangethis, TargetCanvas, fully_connected, cnums = cnums)

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
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(B_avg, B_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(B_avg2, B_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

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
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(D_avg, D_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(D_avg2, D_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

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
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(DB_avg, DB_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(DB_avg2, DB_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

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
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(B_avg, B_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(B_avg2, B_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

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
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(D_avg, D_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(D_avg2, D_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = '',
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)
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
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

        labeltext1, valuetext1, Ttext1, T1, Tthresh1, reftext1 = write_Mconn_values2(DB_avg, DB_sem, NP, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)
        labeltext2, valuetext2, Ttext2, T2, Tthresh2, reftext2 = write_Mconn_values2(DB_avg2, DB_sem2, NP2, betanamelist, rnamelist,
                                                                      beta_list, format='f', pthresh=pthresh, statsref = DBref_mean,
                                                                      sigflag = sigflag, fully_connected=fully_connected, cnums = cnums)

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
                                            beta_list, format='f', R2thresh=R2thresh, fully_connected=fully_connected, cnums = cnums)

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
                                beta_list, format='f', R2thresh=R2thresh,  fully_connected=fully_connected, cnums = cnums)

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
                                    beta_list, format='f', R2thresh=R2thresh,  fully_connected=fully_connected, cnums = cnums)

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


def display_anatomical_slices(clusterdataname, networkfile, regionname, clusternum, templatename):
    orientation = 'axial'
    regioncolor = [1,1,0]

    # get the connection and region information
    clusterdata = np.load(clusterdataname, allow_pickle=True).flat[0]
    # cluster_properties = clusterdata['cluster_properties']
    cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)

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

    # change format of cnums - Nov 30 2024
    if type(cnums[0]) == dict:
        cnumslist = [cnums[x]['cnums'][0] for x in range(len(cnums))]
    else:
        cnumslist = copy.deepcopy(cnums)

    # add ellipses and labels
    for nn in range(len(regions)):
        ellipse = mpatches.Ellipse(regions[nn]['pos'],ovalsize[0],ovalsize[1], alpha = 0.3)
        ax.add_patch(ellipse)
        if nn < len(cnumslist):
            ax.annotate('{}{}'.format(regions[nn]['name'],cnumslist[nn]),regions[nn]['pos']+regions[nn]['labeloffset'])
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

    # change format of cnums - Nov 30 2024
    if type(cnums[0]) == dict:
        cnumslist = [cnums[x]['cnums'][0] for x in range(len(cnums))]
    else:
        cnumslist = copy.deepcopy(cnums)

    # add ellipses and labels
    for nn in range(len(regions)):
        ellipse = mpatches.Ellipse(regions[nn]['pos'],ovalsize[0],ovalsize[1], alpha = 0.3)
        ax.add_patch(ellipse)
        if nn < len(cnumslist):
            ax.annotate('{}{}'.format(regions[nn]['name'],cnumslist[nn]),regions[nn]['pos']+regions[nn]['labeloffset'])
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


def generate_null_data_set(regiondataname, networkfile, covariatesname, npeople=0, variable_variance = False):
    # r = np.load(regiondataname, allow_pickle=True).flat[0]
    r = load_filtered_regiondata(regiondataname, networkfile)

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

        # if normalize_var:
        #     nr, tsize = np.shape(Sconn)
        #     tcdata_std = np.zeros(nr)
        #     # # normalize the data to have the same variance, for each person
        #     for nn in range(nr):
        #         tcdata_std[nn] = np.std(Sconn[nn, :])
        #         Sconn[nn, :] /= tcdata_std[nn]

        for cc in range(nclusters):
            for tt in range(nruns_total):
                t1 = tt*tsize
                t2 = (tt+1)*tsize
                tc_run = np.random.normal(0,1,tsize)
                if variable_variance:
                    std_original = np.std(tc[cc,t1:t2])
                    tc_run *= std_original/np.std(tc_run)  # vary the standard deviation randomly across runs/clusters
                else:
                    tc_run /= np.std(tc_run) # make sure the standard deviation = 1 for every run
                tc_run -= np.mean(tc_run)
                new_tc[cc,t1:t2] = copy.deepcopy(tc_run)

        region_properties[nn]['tc'] = copy.deepcopy(new_tc)
        region_properties[nn]['tc_sem'] = copy.deepcopy(new_tc_sem)
        region_properties[nn]['tc_original'] = copy.deepcopy(new_tc)
        region_properties[nn]['tc_sem_original'] = copy.deepcopy(new_tc_sem)
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
    # r = np.load(regiondataname, allow_pickle=True).flat[0]
    r = load_filtered_regiondata(regiondataname, networkfile)

    region_properties = r['region_properties']
    DBname = r['DBname']
    DBnum = r['DBnum']

    p,f = os.path.split(regiondataname)
    paramsname = os.path.join(p,'simparams.npy')
    prep_data_sem_physio_model_SO_V2(networkfile, regiondataname, clusterdataname, paramsname, timepoint, epoch,
                                     run_whole_group=False, normalizevar=False, filter_tcdata=False)

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



def run_null_test_on_network(nsims, networkmodel, cnums, regiondataname, clusterdataname, timepoint = 'all', epoch = 'all',
                             betascale = 0.1, Lweight = 0.001, alphascale = 0.1, levelthreshold = [1e-5, 1e-6, 1e-6],
                             leveliter = [100, 250, 1200], leveltrials = [30, 4, 1], run_whole_group = False, resumerun = False):
    resultsdir, networkfilename = os.path.split(networkmodel)
    networkbasename, ext = os.path.splitext(networkfilename)

    covariatesname = []
    null_regiondataname, null_covariates = generate_null_data_set(regiondataname, networkmodel, covariatesname, npeople=nsims, variable_variance = False)

    SAPMresultsname = os.path.join(resultsdir,'null_results.npy')
    SAPMparametersname = os.path.join(resultsdir,'null_params.npy')

    if run_whole_group:

        # p, f = os.path.split(SAPMresultsname)
        # f1, e = os.path.splitext(f)
        # test_record_name = os.path.join(p, 'gradient_descent_record.npy')
        # betavals_savename = os.path.join(p, 'betavals_' + f1[:20] + '.npy')

        # first run the whole group and then run the individuals using the group results to guide the choice of beta_init
        print('...running SAPM with null data on the entire group')
        SAPMrun_V2(cnums, null_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
                    epoch, betascale = betascale, Lweight = Lweight, alphascale = alphascale, leveltrials=leveltrials,
                   leveliter=leveliter, levelthreshold = levelthreshold, reload_existing = False, fully_connected = False,
                   run_whole_group = True, verbose = True, silentrunning = False, resumerun = resumerun)
    else:
        print('...running SAPM with null data on each individual')
        SAPMrun_V2(cnums, null_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
                    epoch, betascale = betascale, Lweight = Lweight, alphascale = alphascale, leveltrials=leveltrials,
                   leveliter=leveliter, levelthreshold = levelthreshold, reload_existing = False, fully_connected = False,
                   run_whole_group = False, verbose = True, silentrunning = False, resumerun = resumerun)

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
    if run_whole_group:
        betavals = results[0]['betavals'][:,np.newaxis]
    else:
        betavals = np.zeros((ncon,nsims))
        for nn in range(nsims): betavals[:,nn] = results[nn]['betavals']

    bstats = []
    bstats_values = []
    for nn in range(ncon):
        conname = '{}-{}'.format(rnamelist_full[csource[nn]], rnamelist_full[ctarget[nn]])
        b = copy.deepcopy(betavals[nn,:])
        if run_whole_group:
            entry = {'name':conname, 'mean':b[0], 'std':0., 'skewness':0., 'kurtosis':0.}
        else:
            entry = {'name':conname, 'mean':np.mean(b), 'std':np.std(b), 'skewness':scipy.stats.skew(b), 'kurtosis':scipy.stats.kurtosis(b)}
        values_entry = {'name':conname, 'b':b}
        bstats.append(entry)
        bstats_values.append(values_entry)

    npyname = os.path.join(resultsdir, networkbasename + '_bstats.npy')
    np.save(npyname,bstats)
    npynamev = os.path.join(resultsdir, networkbasename + '_bstats_values.npy')
    np.save(npynamev,bstats_values)

    try:
        if run_whole_group:
            xlname = os.path.join(resultsdir, networkbasename + '_bstats_group.xlsx')
        else:
            xlname = os.path.join(resultsdir, networkbasename + '_bstats.xlsx')
        df = pd.DataFrame(bstats)
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='B stats')
    except:
        dateflag = time.ctime()
        dateflag = dateflag.replace(':','')
        dateflag = dateflag.replace(' ','')
        if run_whole_group:
            xlname = os.path.join(resultsdir, networkbasename + '_bstats_group ' + dateflag + '.xlsx')
        else:
            xlname = os.path.join(resultsdir, networkbasename + '_bstats ' + dateflag + '.xlsx')
        df = pd.DataFrame(bstats)
        with pd.ExcelWriter(xlname) as writer:
            df.to_excel(writer, sheet_name='B stats')

    return xlname


def display_null_test_distributions(bstats_values_name):
    bstatsvalues = np.load(bstats_values_name, allow_pickle=True)
    wn = 179
    plt.close(wn)
    fig = plt.figure(wn)
    nv = len(bstats)
    for nn in range(nv):
        b = copy.deepcopy(bstatsvalues[nn]['b'])
        plt.violinplot(b, positions=[nn])
        plt.plot(nn*np.ones(len(b)),b,'ok')
    plt.plot([0,nv],[0,0],'-k')



def run_sim_test_on_network(nsims, networkmodel, cnums, regiondataname, clusterdataname, timepoint = 'all', epoch = 'all',
                            betascale = 0.1, Lweight = 1.0, alphascale = 0.01, leveltrials=[30, 4, 1],
                            leveliter=[100, 250, 1200], levelthreshold = [1e-4, 1e-5, 1e-6], run_whole_group = False,
                            resumerun = False):

    resultsdir, networkfilename = os.path.split(networkmodel)
    networkbasename, ext = os.path.splitext(networkfilename)

    covariatesname = []
    sim_regiondataname, sim_covariates = generate_simulated_data_set(regiondataname, covariatesname, networkmodel, clusterdataname, npeople=nsims,
                                variable_variance=False, timepoint='all', epoch='all')

    SAPMresultsname = os.path.join(resultsdir,'sim_results.npy')
    SAPMparametersname = os.path.join(resultsdir,'sim_params.npy')

    if run_whole_group:
        # first run the whole group and then run the individuals using the group resutls to guide the choice of beta_init
        SAPMrun_V2(cnums, sim_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
                    epoch, betascale = betascale, Lweight = Lweight, alphascale = alphascale, leveltrials=leveltrials,
                   leveliter=leveliter, levelthreshold = levelthreshold, reload_existing = False, fully_connected = False,
                   run_whole_group = True, resumerun = resumerun)

        betavals_savename = os.path.join(resultsdir, 'betavals_save_record.npy')
        SAPMrun_V2(cnums, sim_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
                    epoch, betascale = betavals_savename, Lweight = Lweight, alphascale = alphascale, leveltrials=leveltrials,
                   leveliter=leveliter, levelthreshold = levelthreshold, reload_existing = False, fully_connected = False,
                   run_whole_group = False, verbose = False, silentrunning = False, resumerun = resumerun)
    else:
        SAPMrun_V2(cnums, sim_regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkmodel, timepoint,
                    epoch, betascale = betascale, Lweight = Lweight, alphascale = alphascale, leveltrials=leveltrials,
                   leveliter=leveliter, levelthreshold = levelthreshold, reload_existing = False, fully_connected = False,
                   run_whole_group = False, verbose = False, silentrunning = False, resumerun = resumerun)


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


def cnums_to_clusterlist(cnums, nclusterlist):
    full_rnum_base = np.array([np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]).astype(int)
    clusterlist = np.array([(cnums[x]['cnums'][x2] + full_rnum_base[x]) for x in range(len(cnums)) for x2 in range(len(cnums[x]['cnums']))]).astype(int)
    return clusterlist


def check_alternative_latent_DBvals(Sinput, Minput, Mconn, Mintrinsic, fintrinsic_count, betavals, deltavals, ctarget, csource, dtarget, dsource, silentrunning = False):
    nr, tsize_total = np.shape(Sinput)
    c_latent = np.where(csource >= nr)[0]
    ns,ndb = np.shape(betavals)  # betavals here is a list of options for starting values for betavals
    Nintrinsics,tsize_total = np.shape(Mintrinsic)
    new_betavals = copy.deepcopy(betavals)
    new_deltavals = copy.deepcopy(deltavals)

    Ncombos = 2**Nintrinsics   # number of options to try

    vsize = 2*np.ones(Nintrinsics)
    R2avg_final = np.zeros(ns)
    for ss in range(ns):

        # find one best alternative for each initial set of betavals
        R2avg_list = []
        betavals_record = []
        for nnn in range(Ncombos):
            betavals2 = copy.deepcopy(betavals[ss, :])
            deltavals2 = copy.deepcopy(deltavals[ss, :])

            negflag = ind2sub_ndims(vsize, nnn)
            for mm in range(Nintrinsics):
                if negflag[mm] > 0:
                    # reverse a DB value for a latent input
                    betavals2[c_latent[mm]] *= -1.

            Mconn2 = copy.deepcopy(Mconn)
            Mconn2[ctarget,csource] = copy.deepcopy(betavals2)
            Minput2 = copy.deepcopy(Minput)
            Minput2[dtarget,dsource] = copy.deepcopy(deltavals2)

            e, v = np.linalg.eig(Mconn2)  # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
            Meigv2 = np.real(v[:, -Nintrinsics:])

            for aa in range(Nintrinsics):
                Meigv2[:,aa] = Meigv2[:,aa]/Meigv2[(-Nintrinsics+aa),aa]

            if fintrinsic_count > 0:
                Mintrinsic2 = np.zeros((Nintrinsics, tsize_total))

                # Mint_variable = np.linalg.inv(M1[:,1:].T @ M1[:,1:]) @ M1[:,1:].T @ Sinput
                Mint_fixed = copy.deepcopy(Mintrinsic[0,:])[np.newaxis,:]
                partial_fit = (Minput2 @ Meigv2[:,0])[:,np.newaxis] @ Mint_fixed    # is this right?

                residual = Sinput-partial_fit
                M1r = Minput2 @ Meigv2[:,1:]

                Mint_variable = np.linalg.inv(M1r.T @ M1r) @ M1r.T @ residual

                Mintrinsic2[0,:] = copy.deepcopy(Mint_fixed)
                Mintrinsic2[1:,:] = copy.deepcopy(Mint_variable)

                fit2 = Minput2 @ Meigv2 @ Mintrinsic2
            else:
                M1 = Minput2 @ Meigv2
                Mintrinsic2 = np.linalg.inv(M1.T @ M1) @ M1.T @ Sinput
                fit2 = Minput2 @ Meigv2 @ Mintrinsic2

            R2_2 = 1.0 - np.sqrt(np.sum(Sinput - fit2) ** 2) / np.sqrt(np.sum(Sinput ** 2))
            R2avg_2 = np.mean(R2_2)

            R2avg_list += [R2avg_2]
            betavals_record.append({'betavals':betavals2})

        # find the best results
        cbest = np.argmax(R2avg_list)
        # R2avg_final[ss] = np.max(R2avg_list)
        if cbest == 0:
            if not silentrunning:
                print('betaval set {}  .... kept the original set'.format(ss))
        else:
            negflag = ind2sub_ndims(vsize, cbest)
            if not silentrunning:
                print('betaval set {}  .... added new set.  Latent change flags: {}'.format(ss, negflag))
            added_betavals = copy.deepcopy(betavals_record[cbest]['betavals'])[np.newaxis,:]
            new_betavals = np.concatenate((new_betavals, added_betavals), axis = 0)   # add the new set
            new_deltavals = np.concatenate((new_deltavals, deltavals[ss,:][np.newaxis,:]), axis = 0)   # add the new set

    return new_betavals, new_deltavals


def network_model_check(networkmodelname):
    output_message = []

    network, nclusterdict, sem_region_list, fintrinsic_count, vintrinsic_count, \
         fintrinsic_base = load_network_model_w_intrinsics(networkmodelname)

    Nintrinsic = fintrinsic_count + vintrinsic_count
    nregions = len(sem_region_list) - Nintrinsic

    nclusters_to_use = list(np.ones(nregions).astype(int))
    nclusters_to_use_total = copy.deepcopy(nregions)

    Minput = np.zeros((nregions, nregions + Nintrinsic))  # mixing of connections to model the inputs to each region
    for nn in range(len(network)):
        target = network[nn]['targetnum']
        sources = network[nn]['sourcenums']

        target_fc = np.sum(nclusters_to_use[:target]).astype(int)
        nclusters_target_fc = nclusters_to_use[target]

        for ttfc in range(target_fc, (target_fc + nclusters_target_fc)):
            for mm in range(len(sources)):
                source = sources[mm]
                if source >= nregions:  # latent input
                    source_fc = nclusters_to_use_total + (source - nregions)
                    nclusters_source_fc = 1
                else:
                    source_fc = np.sum(nclusters_to_use[:source]).astype(int)
                    nclusters_source_fc = int(nclusters_to_use[source])

                for ssfc in range(source_fc, (source_fc + nclusters_source_fc)):
                    Minput[ttfc, ssfc] = 1

    # check network model
    det0 = np.linalg.det(Minput.T @ Minput)
    if det0 == 0.:
        status = 'Failure'
        print('network model is bad.')
        output_message += ['network model is bad.']

        detlist = np.zeros(nregions + Nintrinsic)
        for nn in range(nregions + Nintrinsic):
            ii = list(range(nregions + Nintrinsic))
            ii.remove(nn)
            detlist[nn] = np.linalg.det(Minput[:, ii].T @ Minput[:, ii])
            dcheck = np.where(np.abs(detlist) > 0)[0]

        print('problem regions appear to be numbers:  {}'.format(dcheck))

        output_message += ['problem regions appear to be numbers:  {}'.format(dcheck)]

        print('Minput matrix: ')
        M = copy.deepcopy(Minput)
        # M = Minput.T @ Minput
        nr, nlt = np.shape(M)
        for n1 in range(nr):
            if M[n1, 0] == 0:
                text = '  0  '
            else:
                text = '{:.2f} '.format(M[n1, 0])
            for n2 in range(1, nlt):
                if M[n1, n2] == 0:
                    text += '  0  '
                else:
                    text += '{:.2f} '.format(M[n1, n2])
            print(text)

    else:
        print('network model is good.')
        status = 'Success'
        output_message += ['network model is good.']

    return status, output_message


#
#
# if __name__ == '__main__':
#     main()
#
