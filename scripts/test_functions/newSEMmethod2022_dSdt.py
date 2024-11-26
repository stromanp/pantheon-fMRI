# test_CCrecord_display
# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv')

# test_connectivity_with_derivatives.py

import numpy as np
import matplotlib.pyplot as plt
import py2ndlevelanalysis
import copy
import pyclustering
import pydisplay
import time
import pysem
from sklearn.cluster import KMeans
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d

# pack the values
# def pack_values(betavals, Soutput, intrinsic2):
#     vals = []
#     vals = list(betavals.flatten())
#     vals += list(Soutput.flatten())
#     vals += list(intrinsic2.flatten())
#     return vals
#
#
# def unpack_values(vals, size1, size2, size3):
#     nvals1 = np.prod(size1)
#     nvals2 = np.prod(size2)
#     nvals3 = np.prod(size3)
#     s = 0
#     betavals = np.reshape(vals[:nvals1],size1)
#     s += nvals1
#     Soutput = np.reshape(vals[s:s+nvals2],size2)
#     s += nvals2
#     intrinsic2 = np.reshape(vals[s:s+nvals3],size3)
#     return betavals, Soutput, intrinsic2


# main function
outputdir = r'D:/threat_safety_python/SEMresults'
networkfile = r'D:/threat_safety_python/network_model_with_intrinsics2.xlsx'
network, ncluster_list, sem_region_list = pyclustering.load_network_model(networkfile)

# load data
DBname = r'D:/threat_safety_python/threat_safety_database.xlsx'
xls = pd.ExcelFile(DBname, engine='openpyxl')
df1 = pd.read_excel(xls, 'paradigm1_BOLD')
del df1['Unnamed: 0']  # get rid of the unwanted header column
fields = list(df1.keys())
paradigm = df1['paradigms_BOLD']
timevals = df1['time']
paradigm_centered = paradigm - np.mean(paradigm)
dparadigm = np.zeros(len(paradigm))
dparadigm[1:] = np.diff(paradigm_centered)

region_data_name1 = r'D:/threat_safety_python/SEMresults/threat_safety_regiondata_allthreat55.npy'
clustername = r'D:/threat_safety_python/SEMresults/threat_safety_clusterdata.npy'

region_data1 = np.load(region_data_name1, allow_pickle=True).flat[0]
region_properties = region_data1['region_properties']

cluster_data = np.load(clustername, allow_pickle=True).flat[0]
cluster_properties = cluster_data['cluster_properties']

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


# setup index lists---------------------------------------------------------------------------
# timepoints for full runs----------------------------------------------
tplist_full = []
dtplist_full = []
et1 = 0
et2 = tsize
dtsize = tsize-1  # for using deriviation of tc wrt time
tplist1 = []
dtplist1 = []
nclusterstotal,tsizetotal = np.shape(tcdata)
tcdata_centered = copy.deepcopy(tcdata)
dtcdata_centered = np.zeros((nclusterstotal,nruns_total*tsize))
for nn in range(NP):
    r1 = sum(nruns_per_person[:nn])
    r2 = sum(nruns_per_person[:(nn + 1)])
    tp = []  # initialize list
    dtp = []  # initialize list
    tpoints = []
    dtpoints = []
    for ee2 in range(r1, r2):
        tp = list(range((ee2 * tsize), (ee2 * tsize + tsize)))
        # dtp = list(range((ee2 * dtsize), (ee2 * dtsize + dtsize)))
        tpoints = tpoints + tp  # concatenate lists
        # dtpoints = dtpoints + dtp  # concatenate lists
        temp = np.mean(tcdata[:, tp],axis=1)
        temp_mean = np.repeat(temp[:, np.newaxis], tsize, axis=1)
        tcdata_centered[:, tp] = tcdata[:, tp] - temp_mean   # center each epoch, in each person
        dtcdata_centered[:, tp[1:]] = np.diff(tcdata[:, tp])   # 1st dervitive of timecourse wrt time (estimated)
    tplist1.append({'tp': tpoints})
    # dtplist1.append({'tp': dtpoints})
tplist_full.append(tplist1)
# dtplist_full.append(dtplist1)


# setup matrices for solving network equation
Nintrinsic = 2
nregions = len(rnamelist)
m = Nintrinsic + nregions
Minput = np.zeros((m,m))
Moutput = np.zeros((m,m))
for nn in range(nregions):
    a = network[nn]['targetnum']
    b = network[nn]['sourcenums']
    Minput[a,b] = 1
    Moutput[a,b] = 1

for nn in range(Nintrinsic):
    Minput[nn+nregions,nn+nregions] = 1
    Moutput[nn+nregions,nn+nregions] = 1

# keep a record of matrix indices that need to be estimated---------------
ctarget0,csource0 = np.where(Minput > 0)
exclude = np.where( (ctarget0 >= nregions) & (csource0 >= nregions))[0]   # don't scale intrinsics at the output
keep = np.setdiff1d(list(range(len(ctarget0))),exclude)
ctarget = ctarget0[keep]
csource = csource0[keep]

nbeta = len(ctarget)  # the number of beta values to be estimated

timepoint = 0
nperson = 0    # select one person (for testing)
tp = tplist_full[timepoint][nperson]['tp']
tsize_total = len(tp)

# get tc data for each region/cluster
clusterlist = [1,9,11,15,23,26,31,36,43,45]
Sinput = []
for cval in clusterlist:
    tc1 = dtcdata_centered[cval, tp]
    Sinput.append(tc1)
# Sinput is size:  nregions x tsize_total

beta_int1 = 0.1    # start the magnitude of intrinsic1 at a small value
intrinsic1 = np.array(list(dparadigm) * nruns_per_person[nperson])
intrinsic2 = 0.01*np.random.randn(tsize_total)    # initialize unknown intrinsic with small random values

#  Sinput = Minput @ Moutput @ Soutput    --> solve for Soutput (including intrinsic2)
Soutput = 0.01*np.random.randn(nregions,tsize_total)   # initialize Soutput with small random values
betavals = np.zeros(nbeta)  # initialize beta values at zero

# on each iteration....
# determine values of:   betavals, Soutput, intrinsic2
alpha = 1e-3
Lweight = 1.0e-10
dval = 0.05

Sinput_full = np.concatenate((Sinput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
Soutput_full = np.concatenate((Soutput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
Moutput[ctarget,csource] = betavals

# starting point
fit = Minput @ Moutput @ Soutput_full
err = Sinput_full[:nregions,:] - fit[:nregions,:]
cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(betavals)) + np.sum(np.abs(Soutput)) + np.sum(np.abs(intrinsic2))
ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization
ssqd_starting = ssqd

nitermax = 100
tol = 1.0e-6
dval_limit = 1.0e-4

dssqd = 1.
iter = 0
while dval > dval_limit  and iter < nitermax:
    iter += 1

    # gradients for betavals
    othercost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(Soutput)) + np.sum(np.abs(intrinsic2))   # part that doesn't change here
    dssq_db = np.zeros(np.shape(betavals))
    for nn in range(nbeta):
        b = copy.deepcopy(betavals)
        b[nn] += dval
        Moutput[ctarget, csource] = b

        fit = Minput @ Moutput @ Soutput_full
        err = Sinput_full[:nregions,:] - fit[:nregions,:]
        cost = np.sum(np.abs(b)) + othercost
        ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        dssq_db[nn] = (ssqdp - ssqd) / dval

    # betavals -= alpha * dssq_db

    # gradients for Soutput
    # initialize
    Sinput_full = np.concatenate((Sinput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Soutput_full = np.concatenate((Soutput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Moutput[ctarget,csource] = betavals

    othercost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(betavals)) + np.sum(np.abs(intrinsic2))  # part that doesn't change here
    dssq_dS = np.zeros(np.shape(Soutput))
    nS = np.size(Soutput)
    nregions,tsize_full = np.shape(Soutput)
    for nn in range(nS):
        S = copy.deepcopy(Soutput)
        m,n = np.unravel_index(nn,(nregions,tsize_full))
        S[m,n] += dval
        S_full = np.concatenate((S, beta_int1*intrinsic1[np.newaxis, :], intrinsic2[np.newaxis, :]), axis=0)

        fit = Minput @ Moutput @ S_full
        err = Sinput_full[:nregions,:] - fit[:nregions,:]
        cost = np.sum(np.abs(S)) + othercost
        ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        dssq_dS[m,n] = (ssqdp - ssqd) / dval

    # Soutput -= alpha * dssq_dS

    # gradients for intrinsic2
    # initialize
    Sinput_full = np.concatenate((Sinput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Soutput_full = np.concatenate((Soutput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Moutput[ctarget,csource] = betavals

    othercost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(betavals)) + np.sum(np.abs(Soutput))  # part that doesn't change here
    dssq_dI = np.zeros(np.shape(intrinsic2))
    nI = np.size(intrinsic2)
    for nn in range(nI):
        II = copy.deepcopy(intrinsic2)
        II[nn] += dval
        S_full = np.concatenate((Soutput, beta_int1*intrinsic1[np.newaxis, :], II[np.newaxis, :]), axis=0)
        Sin_full = np.concatenate((Sinput, beta_int1*intrinsic1[np.newaxis, :], II[np.newaxis, :]), axis=0)

        fit = Minput @ Moutput @ S_full
        err = Sin_full[:nregions,:] - fit[:nregions,:]
        cost = np.sum(np.abs(II)) + othercost
        ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        dssq_dI[nn] = (ssqdp - ssqd) / dval

    # gradient for beta_int1
    # initialize
    Sinput_full = np.concatenate((Sinput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Soutput_full = np.concatenate((Soutput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Moutput[ctarget,csource] = betavals

    dint = copy.deepcopy(beta_int1)
    dint += dval
    S_full = np.concatenate((Soutput, dint*intrinsic1[np.newaxis, :], intrinsic2[np.newaxis,:]), axis=0)
    Sin_full = np.concatenate((Sinput, dint*intrinsic1[np.newaxis, :], intrinsic2[np.newaxis,:]), axis=0)

    fit = Minput @ Moutput @ S_full
    err = Sin_full[:nregions,:] - fit[:nregions,:]
    cost = np.sum(np.abs(dint)) + np.sum(np.abs(betavals)) + np.sum(np.abs(Soutput))  + np.sum(np.abs(intrinsic2))
    ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
    dssq_dbeta1 = (ssqdp - ssqd) / dval


    # apply the changes
    betavals -= alpha * dssq_db
    Soutput -= alpha * dssq_dS
    intrinsic2 -= alpha * dssq_dI
    beta_int1 -= alpha * dssq_dbeta1

    Sinput_full = np.concatenate((Sinput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Soutput_full = np.concatenate((Soutput,beta_int1*intrinsic1[np.newaxis,:],intrinsic2[np.newaxis,:]),axis=0)
    Moutput[ctarget,csource] = betavals

    fit = Minput @ Moutput @ Soutput_full
    err = Sinput_full[:nregions,:] - fit[:nregions,:]
    cost = np.sum(np.abs(betavals)) + np.sum(np.abs(Soutput)) + np.sum(np.abs(intrinsic2))
    ssqd_new = np.sum(err ** 2) + Lweight * cost  # L1 regularization

    if ssqd_new > ssqd:
        dval *= 0.5
        # revert back to last good values
        betavals = copy.deepcopy(lastgood_betavals)
        Soutput = copy.deepcopy(lastgood_Soutput)
        intrinsic2 = copy.deepcopy(lastgood_intrinsic2)
        beta_int1 = copy.deepcopy(lastgood_beta_int1)
    else:
        # save the good values
        lastgood_betavals = copy.deepcopy(betavals)
        lastgood_Soutput = copy.deepcopy(Soutput)
        lastgood_intrinsic2 = copy.deepcopy(intrinsic2)
        lastgood_beta_int1 = copy.deepcopy(beta_int1)

        dssqd = ssqd - ssqd_new
        ssqd = ssqd_new
    print('iter {} delta ssq {:.4f}  relative: {:.1f} percent'.format(iter,-dssqd,100.0*ssqd/ssqd_starting))


# show results
betavals = copy.deepcopy(lastgood_betavals)
Soutput = copy.deepcopy(lastgood_Soutput)
intrinsic2 = copy.deepcopy(lastgood_intrinsic2)
beta_int1 = copy.deepcopy(lastgood_beta_int1)

Sinput_full = np.concatenate((Sinput, intrinsic1[np.newaxis, :], intrinsic2[np.newaxis, :]), axis=0)
Soutput_full = np.concatenate((Soutput, intrinsic1[np.newaxis, :], intrinsic2[np.newaxis, :]), axis=0)
Moutput[ctarget, csource] = betavals

fit = Minput @ Moutput @ Soutput_full
err = Sinput_full[:nregions, :] - fit[:nregions, :]

regionnum = 6
plt.close(24)
fig = plt.figure(24)
plt.plot(range(tsize_full),Sinput_full[regionnum,:],'-ob')
plt.plot(range(tsize_full),fit[regionnum,:],'-xr')

plt.close(25)
fig = plt.figure(25)
plt.plot(range(tsize_full),Sinput_full[regionnum,:],'-ob')
plt.plot(range(tsize_full),Soutput_full[regionnum,:],'-xr')

# convert from dS/dt to S
nruns = nruns_per_person[nperson]
TCinput_full = np.zeros((nregions+Nintrinsic, tsize*nruns))
TCoutput_full = np.zeros((nregions+Nintrinsic, tsize*nruns))
for rr in range(nregions+Nintrinsic):
    dtcinfull = Sinput_full[rr,:]
    dtcoutfull = Soutput_full[rr,:]

    temp = np.reshape(dtcinfull, (nruns, tsize))
    tcinfull = np.zeros((nruns,tsize))
    tcinfull = np.cumsum(temp,axis=1)
    TCinput_full[rr,:] = np.reshape(tcinfull,nruns*tsize)

    temp = np.reshape(dtcoutfull, (nruns, tsize))
    tcoutfull = np.zeros((nruns,tsize))
    tcoutfull = np.cumsum(temp,axis=1)
    TCoutput_full[rr,:] = np.reshape(tcoutfull,nruns*tsize)
