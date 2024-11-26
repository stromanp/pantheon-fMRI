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
from sklearn.decomposition import PCA
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
from mpl_toolkits import mplot3d


def sub2ind(vsize, indices):
    # give all the combinations for the values that are allowed to vary
    ndims = len(vsize)
    w = 0
    for nn in range(ndims):
        w += indices[nn]*np.prod(vsize[:nn])
    return w


def all_flat_indices_nfixed(vsize, fixedindices, fixedvals):
    # give all the combinations for the values that are allowed to vary
    ndims = len(vsize)
    vsize2 = copy.deepcopy(vsize)
    vsize2[fixedindices] = 1
    nc = np.prod(vsize2)
    w = np.zeros(nc).astype(int)
    # convert to original vsize indices
    for nn in range(nc):
        # x = np.array(np.unravel_index(nn,vsize2))
        x = ind2sub_ndims(vsize2, nn)
        x[fixedindices] = fixedvals
        w[nn] = sub2ind(vsize, x)
    return w


def load_network_model_w_intrinsics(networkmodel):
    xls = pd.ExcelFile(networkmodel, engine = 'openpyxl')
    dnet = pd.read_excel(xls, 'connections')
    dnet.pop('Unnamed: 0')   # remove this blank field from the beginning
    dnclusters = pd.read_excel(xls, 'nclusters')

    vintrinsic_count = 0
    fintrinsic_count = 0

    nregions = len(dnclusters)
    ntargets, ncols = dnet.shape
    nsources_max = ncols-1

    sem_region_list = []
    ncluster_list = []
    for nn in range(nregions):
        sem_region_list.append(dnclusters.loc[nn,'name'])
        cname = dnclusters.loc[nn,'name']
        if 'vintrinsic' in cname:  vintrinsic_count += 1
        if 'fintrinsic' in cname:  fintrinsic_count += 1
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

    return network, ncluster_list, sem_region_list, fintrinsic_count, vintrinsic_count


def gradients_in_vintrinsics(Sinput, Soutput, fintrinsic1, vintrinsics, beta_int1,
                             Minput, Moutput, dvali, fintrinsic_count, vintrinsic_count):
    nregions, tsize_full = np.shape(Sinput)
    nv,nt = np.shape(vintrinsics)
    nI = nv*nt
    dssq_dI = np.zeros((nv,nt))

    II = copy.deepcopy(vintrinsics)
    Sinput_full = np.array(Sinput)
    Soutput_full = np.array(Soutput)
    if fintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Soutput_full = np.concatenate((Soutput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
        Soutput_full = np.concatenate((Soutput_full, vintrinsics), axis=0)

    fit, Soutput_full = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)
    Soutput = Soutput_full[:nregions,:]

    err = Sinput_full[:nregions, :] - fit[:nregions, :]
    cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(II)) + np.sum(np.abs(betavals))
    ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

    for nn in range(nI):
        II = copy.deepcopy(vintrinsics)
        aa,bb = np.unravel_index(nn, (nv,nt))
        II[aa,bb] += dvali

        Sin_full = np.array(Sinput)
        S_full = np.array(Soutput)
        if fintrinsic_count > 0:
            Sin_full = np.concatenate((Sin_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            S_full = np.concatenate((S_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Sin_full = np.concatenate((Sin_full, II), axis=0)
        S_full = np.concatenate((S_full, II), axis=0)

        fit, S_full = network_eigenvalue_method(S_full, Minput, Moutput, nregions)

        err = Sin_full[:nregions, :] - fit[:nregions, :]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(II)) + np.sum(np.abs(betavals))
        ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        dssq_dI[aa,bb] = (ssqdp - ssqd) / dvali

    return dssq_dI, ssqd


def gradients_in_beta1(Sinput, Soutput, fintrinsic1, vintrinsics, beta_int1, Minput, Moutput,
                       dval, fintrinsic_count, vintrinsic_count):
    dint = copy.deepcopy(beta_int1)

    Sin_full = np.array(Sinput)
    S_full = np.array(Soutput)
    if fintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
        S_full = np.concatenate((S_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, vintrinsics), axis=0)
        S_full = np.concatenate((S_full, vintrinsics), axis=0)

    fit, S_full = network_eigenvalue_method(S_full, Minput, Moutput, nregions)
    Soutput = S_full[:nregions,:]

    err = Sin_full[:nregions, :] - fit[:nregions, :]
    cost = np.sum(np.abs(dint)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
    ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

    dint += dval
    Sin_full = np.array(Sinput)
    S_full = np.array(Soutput)
    if fintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
        S_full = np.concatenate((S_full, dint * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sin_full = np.concatenate((Sin_full, vintrinsics), axis=0)
        S_full = np.concatenate((S_full, vintrinsics), axis=0)

    fit, S_full = network_eigenvalue_method(S_full, Minput, Moutput, nregions)

    err = Sin_full[:nregions, :] - fit[:nregions, :]
    # cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(betavals)) + np.sum(np.abs(intrinsic2))
    cost = np.sum(np.abs(dint)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
    ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
    dssq_dbeta1 = (ssqdp - ssqd) / dval

    return dssq_dbeta1, ssqd


def gradients_for_betavals(Sinput, Soutput, fintrinsic1, vintrinsics, beta_int1, Minput, Moutput,
                           betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count):
    nregions, tsize_full = np.shape(Sinput)

    # initialize
    Sinput_full = np.array(Sinput)
    Soutput_full = np.array(Soutput)
    if fintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Soutput_full = np.concatenate((Soutput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
        Soutput_full = np.concatenate((Soutput_full, vintrinsics), axis=0)

    Moutput[ctarget, csource] = betavals

    fit, Soutput_temp = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)
    Soutput = Soutput_temp[:nregions,:]

    err = Sinput_full[:nregions, :] - fit[:nregions, :]
    cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
    ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

    # gradients for betavals
    # cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(betavals)) + np.sum(np.abs(intrinsic2))
    dssq_db = np.zeros(np.shape(betavals))
    for nn in range(nbeta):
        b = copy.deepcopy(betavals)
        b[nn] += dval
        Moutput[ctarget, csource] = b

        fit, Soutput_temp = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)

        err = Sinput_full[:nregions, :] - fit[:nregions, :]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(b))
        ssqdp = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        dssq_db[nn] = (ssqdp - ssqd) / dval

    return dssq_db, ssqd


def network_approach_method(Soutput_full, Minput, Moutput, nregions):
    # Soutput_full = Moutput @ Soutput_full
    # find Soutput_working with given starting values
    method = 1

    nr,nt = np.shape(Soutput_full)
    n_intrinsic = nr-nregions

    # timerecord = [time.time()]
    # estimate fit for test vectors
    initial_value = 0.5
    v_ref = np.zeros((n_intrinsic,nr))
    for nn in range(n_intrinsic):
        v = np.zeros(nr)
        v[nregions+nn] = initial_value
        v_out = network_propagation(v, Moutput, nregions)
        v_ref[nn,:] = v_out

    # approach 1
    if method == 1:
        Soutput_working = copy.deepcopy(Soutput_full)
        for tt in range(nt):
            s = Soutput_full[nregions:,tt]   # intrinsic values input
            v = (s[np.newaxis,:]/initial_value) @ v_ref
            Soutput_working[:nregions,tt] = v[0,:nregions]
        fit = Soutput_working

    # approach 2 (slower)
    if method == 2:
        Soutput_working = copy.deepcopy(Soutput_full)
        for tt in range(nt):
            v = np.zeros(nr)
            v[nregions:] = Soutput_full[nregions:,tt]   # intrinsic values input
            v_out = network_propagation(v, Moutput, nregions)
            Soutput_working[:nregions,tt] = v_out[:nregions]

    fit = Minput @ Soutput_working
    return fit, Soutput_working


def network_propagation(v_in, Moutput, nregions):
    # Soutput_full = Moutput @ Soutput_full
    # find Soutput_working with given starting values
    nr = len(v_in)
    v = copy.deepcopy(v_in)

    dv = 0.05
    alpha = 0.1
    maxiter = 20
    Lweight = 1e-2
    iter = 0
    ssqd = 1e4
    tol = 1e-4

    for aa in range(5):  v = Moutput @ v
    fit = Moutput @ v
    err = v[:nregions] - fit[:nregions]
    cost = np.sum(np.abs(v))
    ssqd = np.sum(err ** 2) + Lweight * cost
    lastgood_v = copy.deepcopy(v)
    while (iter < maxiter) & (alpha > 1e-3):
        iter += 1

        # gradients
        dssq_dv = np.zeros(nregions)
        for rr in range(nregions):
            v2 = copy.deepcopy(v)
            v2[rr] += dv
            fit = Moutput @ v2
            err = v2[:nregions] - fit[:nregions]
            cost = np.sum(np.abs(v2))
            ssqd_dp = np.sum(err ** 2) + Lweight * cost
            dssq_dv[rr] = (ssqd_dp - ssqd) / dv

        v[:nregions] -= alpha * dssq_dv
        fit = Moutput @ v
        err = v[:nregions] - fit[:nregions]
        cost = np.sum(np.abs(v))
        ssqd_new = np.sum(err**2) + Lweight*cost

        if ssqd_new >= ssqd-tol:
            alpha *= 0.5
            # revert back to last good values
            v = copy.deepcopy(lastgood_v)
            dssqd = ssqd - ssqd_new
            # print('network fit: iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
        else:
            # save the good values
            lastgood_v = copy.deepcopy(v)
            dssqd = ssqd - ssqd_new
            ssqd = ssqd_new
            # print('network fit: iter {} alpha {:.3e}  delta ssq {:.2f}  ssqd {:.2f}'.format(iter, alpha, -dssqd, ssqd))

    v_propagated = lastgood_v
    return v_propagated



def network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions):
    #
    # Soutput_full = Moutput @ Soutput_full
    # find Soutput_working with given starting values
    # the form of Moutput is a block matrix with the upper nregions x nregions section
    # giving the beta values
    # the lower nintrinsic x nintrinsic portion is an identity matrix
    # and the upper right nregions x nintrinsic porition is the mixing from the intrinsics
    # to the regions
    # This form ensures that there are are number of eigenvalues = 1, and the number
    # is equal to nintrinsic
    # the corresponding eigenvectors have non-zero values for the intrinsic inputs and for
    # other regions only if there is mixing between them
    nr,nt = np.shape(Soutput_full)
    nintrinsics = nr-nregions

    det = np.linalg.det(Moutput)
    w,v = np.linalg.eig(Moutput)

    # Moutput @ v[:,a] = w[a]*v[:,a]

    # exclude eigenvectors with v[fixed_component] = 0 (or near there)
    # limit = 0.1
    # vcheck = np.abs(v[fixed_component,:])
    # x2 = np.where(vcheck > limit)[0]
    #
    # # find closest eigenvalues to 1
    # diff = np.abs(w[x2] - complex(1,0))
    # s = np.argsort(diff)
    # x = x2[s]

    # check that intrinsics have eigenvalues = 1 (or close to it)
    # assume that the eigenvalues, eigenvectors are always ordered the same as Moutput
    check = np.zeros(nintrinsics)
    tol = 1e-4
    for nn in range(nintrinsics):
        check[nn] = np.abs(w[nn+nregions]-1.0) < tol

    if np.sum(check) < nintrinsics:
        print('--------------------------------------------------------------------------------')
        print('Error:  network_eigenvalue_method:  solution to network fitting cannot be found!')
        print('--------------------------------------------------------------------------------')
    else:
        # M v = a v
        fit1 = np.zeros((nregions,nt))
        for nn in range(nintrinsics):
            # do this for each intrinsic:
            eval = np.real(w[nn+nregions])
            evec = np.real(v[:,nn+nregions])
            for tt in range(nt):
                scale = Soutput_full[nn+nregions,tt]/evec[nn+nregions]
                fit1[:nregions,tt] += evec[:nregions]*scale

        Soutput_working = copy.deepcopy(Soutput_full)
        Soutput_working[:nregions] = fit1
        fit = Minput @ Soutput_working

    return fit, Soutput_working


def network_descent_L1(Soutput_full, Minput, Moutput, nregions):
    # this method is too slow -----------------------------------
    starttime = time.time()
    maxiter = 250
    Soutput_working = copy.deepcopy(Soutput_full)
    nr,nt = np.shape(Soutput_working)
    nvals = nr*nt
    dssq_dv = np.zeros((nr,nt))
    dv = 0.05
    alpha = 0.1
    alpha_limit = 1e-3
    Lweight = 1e-3

    lastgood_Soutput_working = copy.deepcopy(Soutput_working)

    fit = Moutput @ Soutput_working
    err = Soutput_working[:nregions, :] - fit[:nregions, :]
    cost = np.sum(np.abs(Soutput_working))
    ssqd0 = np.sum(err**2) + Lweight*cost

    iter = 0
    ssqd = ssqd0
    while (iter < maxiter) & (alpha > alpha_limit):
        iter += 1

        # gradients
        for rr in range(nregions):
            for tt in range(nt):
                S = copy.deepcopy(Soutput_working)
                S[rr,tt] += dv
                fit = Moutput @ S
                err = S[:nregions, :] - fit[:nregions, :]
                cost = np.sum(np.abs(S))
                ssqd_dp = np.sum(err ** 2) + Lweight * cost
                dssq_dv[rr,tt] = (ssqd_dp - ssqd) / dv

        Soutput_working -= alpha * dssq_dv
        fit = Moutput @ Soutput_working
        err = Soutput_working[:nregions, :] - fit[:nregions, :]
        cost = np.sum(np.abs(Soutput_working))
        ssqd_new = np.sum(err**2) + Lweight*cost

        if ssqd_new >= ssqd:
            alpha *= 0.5
            # revert back to last good values
            Soutput_working = copy.deepcopy(lastgood_Soutput_working)
            dssqd = ssqd - ssqd_new
            # print('network fit: iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
        else:
            # save the good values
            lastgood_Soutput_working = copy.deepcopy(Soutput_working)
            dssqd = ssqd - ssqd_new
            ssqd = ssqd_new
            # print('network fit: iter {} alpha {:.3e}  delta ssq {:.2f}  ssqd {:.2f}'.format(iter, alpha, -dssqd, ssqd))

    endtime = time.time()
    timeelapsed = endtime-starttime
    print('network fit: iter {} ssqd {:.2f}   {:.1f} seconds to complete'.format(iter, ssqd, timeelapsed))
    fit = Minput @ Soutput_working
    return fit, Soutput_working


def network_descent(Soutput_full, Minput, Moutput, nregions):
    # this method is too slow-----------------------------------------------
    starttime = time.time()
    timerecord = [starttime]
    maxiter = 250
    Soutput_working = copy.deepcopy(Soutput_full)
    nr,nt = np.shape(Soutput_working)
    nvals = nr*nt
    dssq_dv = np.zeros((nr,nt))
    dv = 0.05
    alpha = 0.1
    alpha_limit = 1e-3

    lastgood_Soutput_working = copy.deepcopy(Soutput_working)

    fit = Moutput @ Soutput_working
    err = Soutput_working[:nregions, :] - fit[:nregions, :]
    ssqd0 = np.sum(err**2)

    iter = 0
    ssqd = ssqd0
    timerecord += [time.time()]
    while (iter < maxiter) & (alpha > alpha_limit):
        iter += 1

        timerecord += [time.time()]
        # gradients
        for rr in range(nregions):
            for tt in range(nt):
                S = copy.deepcopy(Soutput_working)
                S[rr,tt] += dv
                fit = Moutput @ S
                err = S[:nregions, :] - fit[:nregions, :]
                ssqd_dp = np.sum(err ** 2)
                dssq_dv[rr,tt] = (ssqd_dp - ssqd) / dv

        timerecord += [time.time()]
        Soutput_working -= alpha * dssq_dv
        fit = Moutput @ Soutput_working
        err = Soutput_working[:nregions,:] - fit[:nregions,:]
        ssqd_new = np.sum(err**2)

        if ssqd_new >= ssqd:
            alpha *= 0.5
            # revert back to last good values
            Soutput_working = copy.deepcopy(lastgood_Soutput_working)
            dssqd = ssqd - ssqd_new
            # print('network fit: iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
        else:
            # save the good values
            lastgood_Soutput_working = copy.deepcopy(Soutput_working)
            dssqd = ssqd - ssqd_new
            ssqd = ssqd_new
            # print('network fit: iter {} alpha {:.3e}  delta ssq {:.2f}  ssqd {:.2f}'.format(iter, alpha, -dssqd, ssqd))

    endtime = time.time()
    timerecord += [endtime]
    timeelapsed = endtime-starttime
    print('network fit: iter {} ssqd {:.2f}   {:.1f} seconds to complete'.format(iter, ssqd, timeelapsed))
    fit = Minput @ Soutput_working
    return fit, Soutput_working, timerecord


def network_fit(Soutput_full, Minput, Moutput):
    # this method is fast, but not accurate enough-------------------------
    niter = 10
    Soutput_working = copy.deepcopy(Soutput_full)
    nr,nt = np.shape(Soutput_working)
    for aa in range(niter):
        Soutput_working = Moutput @ Soutput_working
        # temp = np.mean(Soutput_working,axis=1)         # ensure the outputs have zero mean
        # offsetvals = np.tile(temp[:,np.newaxis],(1,nt))
        # Soutput_working -= offsetvals
    fit = Minput @ Soutput_working
    return fit, Soutput_working


def network_sim(Sinput_full, Soutput_full, Minput, Moutput):
    niter = 10
    Soutput_working = copy.deepcopy(Soutput_full)
    for aa in range(niter): Soutput_working = Moutput @ Soutput_working
    fit = Minput @ Soutput_working
    return fit, Soutput_working


def center_tc_by_run(tcdata, nruns):
    # set the mean of each run to zero
    totalt = len(tcdata)
    ts = np.floor(totalt/nruns).astype(int)
    temp = np.reshape(tcdata,(nruns,ts))
    temp = np.mean(temp,axis=1)
    offsetvals = np.tile(temp[:,np.newaxis],(1,ts))
    offsetvals = np.reshape(offsetvals,nruns*ts)
    tcdata_centered = tcdata - offsetvals
    return tcdata_centered


def get_region_num(nclusterlist, number):
    regionend = np.cumsum(nclusterlist)
    cc = np.where(regionend > number)[0]
    regionnum = cc[0]
    return regionnum


def get_region_cluster_pair(nclusterlist, number):
    regionend = np.cumsum(nclusterlist)
    cc = np.where(regionend > number)[0]
    regionnum = cc[0]
    if regionnum == 0:
        clusternum = number
    else:
        clusternum = number - regionend[regionnum-1]
    return regionnum, clusternum


def get_overall_num(nclusterlist, regionnum, clusternum):
    if isinstance(regionnum,list):
        number = [np.sum(nclusterlist[:regionnum[aa]]) + clusternum[aa] for aa in range(len(regionnum))]
    if isinstance(regionnum,np.ndarray):
        number = [np.sum(nclusterlist[:regionnum[aa]]) + clusternum[aa] for aa in range(len(regionnum))]
    if isinstance(regionnum,int):
        number = np.sum(nclusterlist[:regionnum]) + clusternum
    return number

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


def find_best_network(Tcount, network, nclusterlist, starting_target):
    # trace a network based on highest number of signicant betavalue predictions
    nregions = len(nclusterlist)
    nclusterlist = np.array(nclusterlist)
    targetnum_list = [network[aa]['targetnum'] for aa in range(len(network))]

    # every source-target pair in the network:
    source_target_pairs = []
    acount = 0
    for aa in range(len(network)):
        targetnum = network[aa]['targetnum']
        sourcenums = np.array(network[aa]['sourcenums'])
        sourcenums = sourcenums[sourcenums < nregions].astype(int)
        for sn in sourcenums:
            pair = np.array([sn,targetnum])
            if acount == 0:
                source_target_pairs = pair[np.newaxis,:]
            else:
                source_target_pairs = np.append(source_target_pairs,pair[np.newaxis,:],axis = 0)
            acount += 1

    target = starting_target
    tnumber, tcluster = get_region_cluster_pair(nclusterlist, target)

    otherregions = np.setdiff1d(range(nregions),tnumber)
    ncombinations = np.prod(nclusterlist[otherregions])

    cnums = np.zeros(nregions).astype(int)
    full_rnum_base = get_overall_num(nclusterlist, list(range(nregions)), cnums)
    full_cnum_list = np.zeros(nregions).astype(int)
    full_cnum_list[tnumber] = tcluster
    count_record = np.zeros(ncombinations).astype(int)
    for nc in range(ncombinations):
        if np.mod(nc,100000) == 0:  print('done {} of {}  {}'.format(nc,ncombinations,time.ctime()))
        cnums = ind2sub_ndims(nclusterlist[otherregions], nc)
        full_cnum_list[otherregions] = cnums
        rnums = full_cnum_list + full_rnum_base
        source_indices = rnums[source_target_pairs[:,0]]
        target_indices = rnums[source_target_pairs[:,1]]
        count_record[nc] = np.sum(Tcount[source_indices, target_indices])
    print('finished at {}'.format(time.ctime()))

    x = np.argsort(count_record)
    best_index = x[-1]
    cnums = ind2sub_ndims(nclusterlist[otherregions], best_index)
    full_cnum_list[otherregions] = cnums
    rnums = full_cnum_list + full_rnum_base

    return rnums


# temporary-------------------------
# get covariates
# settingsfile = r'C:\Users\Stroman\PycharmProjects\pyspinalfmri3\venv\base_settings_file.npy'
# settings = np.load(settingsfile, allow_pickle=True).flat[0]
# covariates1 = settings['GRPcharacteristicsvalues'][0]  # gender
# covariates2 = settings['GRPcharacteristicsvalues'][1].astype(float)  # painrating

#----------------------------------------------------------------------------------
# main function--------------------------------------------------------------------
starttime = time.ctime()
# main function
outputdir = r'D:/threat_safety_python/SEMresults'
SEMresultsname = os.path.join(outputdir,'SEMresults_newmethod_3.npy')
networkfile = r'D:/threat_safety_python/network_model_with_3intrinsics.xlsx'
network, ncluster_list, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)

# load data--------------------------------------------------------------------
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


# compute grid of dSsource/dStarget--------------------------------------------------------------------
timepoint = 0
dSdSgrid = np.zeros((nclusterstotal, nclusterstotal, NP,2))
for nn in range(NP):
    tp = tplist_full[timepoint][nn]['tp']
    tsize_total = len(tp)
    for ss in range(nclusterstotal):
        dss = dtcdata_centered[ss,tp]
        for tt in range(nclusterstotal):
            dtt = dtcdata_centered[tt,tp]
            dsdt = dss/(dtt+1.0e-20)
            stdval = np.std(dsdt)
            dsdt[np.abs(dsdt) > 3.0*stdval] = 0.0
            dSsdSt = np.mean(dsdt)
            dSsdSt_sem = np.std(dsdt)/np.sqrt(tsize_total)
            dSdSgrid[ss,tt,nn,0] = dSsdSt
            dSdSgrid[ss,tt,nn,1] = dSsdSt_sem
T = dSdSgrid[:, :, :, 0] / (dSdSgrid[:, :, :, 1] + 1.0e-20)

# network mask for T grid etc.--------------------------------------------------------------------
mask = np.zeros((nclusterstotal, nclusterstotal))
for nn in range(len(network)):
    target = network[nn]['targetnum']
    t1 = np.sum(nclusterlist[:target]).astype(int)
    t2 = np.sum(nclusterlist[:(target + 1)])
    sources = network[nn]['sourcenums']
    for mm in range(len(sources)):
        if sources[mm] < nregions:
            s1 = np.sum(nclusterlist[:sources[mm]]).astype(int)
            s2 = np.sum(nclusterlist[:(sources[mm] + 1)])
            mask[s1:s2, t1:t2] = 1

for nn in range(NP):
    dSdSgrid[:, :, nn, 0] *= mask
    dSdSgrid[:, :, nn, 1] *= mask
    T[:, :, nn] *= mask

Tlim = np.abs(T) > 2
Tcount = np.sum(Tlim, axis=2)   # count of how many people have significant estimated beta values for each connection


# setup matrices for modeling network --------------------------------------------------------------------
Nintrinsic = fintrinsic_count + vintrinsic_count
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
exclude = np.where( (ctarget0 >= nregions) | (csource0 >= nregions))[0]   # don't scale intrinsics at the output
keep = np.setdiff1d(list(range(len(ctarget0))),exclude)
ctarget = ctarget0[keep]
csource = csource0[keep]

nbeta = len(ctarget)  # the number of beta values to be estimated

# initialize gradient-descent parameters--------------------------------------------------------------
initial_alpha = 1e-4
initial_alphai = 1e-3
initial_alphab = 1e-4
initial_Lweight = 1.0
initial_dval = 0.05
initial_dvali = 0.05


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
# repeat the process for each participant-----------------------------------------------------------------
betalimit = 3.0
timepoint = 0
SEMresults = []
beta_init_record = []
# person_list = [0,7,10]
# for nperson in person_list:
starttime = time.ctime()
for nperson in range(NP):
    print('starting person {} at {}'.format(nperson,time.ctime()))
    tp = tplist_full[timepoint][nperson]['tp']
    tsize_total = len(tp)
    nruns = nruns_per_person[nperson]

    # get tc data for each region/cluster
    clusterlist = [3,5,13,15,20,25,34,38,43,45]   # picked by "best" function above, with C6RD 3 as starting point
    clusterlist = [4, 9, 14, 15, 20, 28, 32, 35, 41, 47]   # picked by PCA method below
    Sinput = []
    for cval in clusterlist:
        tc1 = tcdata_centered[cval, tp]
        Sinput.append(tc1)
    # Sinput is size:  nregions x tsize_total

    if fintrinsic_count > 0:
        beta_int1 = 0.1    # start the magnitude of intrinsic1 at a small value
        fintrinsic1 = np.array(list(paradigm_centered) * nruns_per_person[nperson])
    else:
        beta_int1 = 0.0

    if vintrinsic_count > 0:
        vintrinsics = np.zeros((vintrinsic_count, tsize_total))    # initialize unknown intrinsic with small random values
        # for v in range(vintrinsic_count):
        #     vtemp = np.mean(Sinput,axis=0)
        #     vtemp = center_tc_by_run(vtemp, nruns)  # keep all tc values at a mean of zero
        #     vintrinsics[v,:] = vtemp

    # initialize beta values based on dSdSgrid and T value of it, for each source-target pair in the network
    # 1) optimize intrinsic1 and intrinsic2 for these beta values
    # 2) optimze betavals for given intrinsic1 and intrinsic2 values
    # 3) repeat

    # initialize beta values
    beta_initial = np.zeros(len(csource))
    for nn in range(len(csource)):
        sourcecluster = clusterlist[csource[nn]]
        targetcluster = clusterlist[ctarget[nn]]
        beta = dSdSgrid[sourcecluster,targetcluster,nperson,0]
        betaT = T[sourcecluster,targetcluster,nperson]
        if np.abs(betaT) > 0.5:
            beta_initial[nn] = beta

    beta_initial[beta_initial >= betalimit] = betalimit
    beta_initial[beta_initial <= -betalimit] = -betalimit
    beta_init_record.append({'beta_initial':beta_initial})

    Soutput = np.zeros((nregions,tsize_total))   # initialize
    betavals = copy.deepcopy(beta_initial) # initialize beta values at zero

    bigitermax = 3
    bigiter = 0
    results_record = []
    lastgood_vintrinsics = copy.deepcopy(vintrinsics)
    lastgood_beta_int1 = copy.deepcopy(beta_int1)
    lastgood_betavals = copy.deepcopy(betavals)
    ssqd_record = []
    while bigiter < bigitermax:
        bigiter += 1
        # on each iteration....
        # first determine values of:  beta_int1, intrinsic2
        # then determine values of: betavals
        alpha = initial_alpha
        alphai = initial_alphai
        alphab = initial_alphab
        Lweight = initial_Lweight
        dval = initial_dval
        dvali = initial_dvali

        # initialize values at the start of each pass------------------
        Sinput_full = np.array(Sinput)
        Soutput_full = np.array(Soutput)
        if fintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full,beta_int1*fintrinsic1[np.newaxis,:]),axis=0)
            Soutput_full = np.concatenate((Soutput_full,beta_int1*fintrinsic1[np.newaxis,:]),axis=0)
        if vintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full,vintrinsics),axis=0)
            Soutput_full = np.concatenate((Soutput_full,vintrinsics),axis=0)
        Moutput[ctarget,csource] = betavals

        # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        fit, Soutput_full = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)
        Soutput = Soutput_full[:nregions,:]
        err = Sinput_full[:nregions,:] - fit[:nregions,:]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
        ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization
        if bigiter == 1:  ssqd_starting = ssqd
        ssqd_record += [ssqd]

        nitermax = 100
        alpha_limit = 1.0e-5

        iter = 0
        vintrinsics_record = []
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        while alphai > alpha_limit  and iter < nitermax  and converging:
            iter += 1

            # gradients for vintrinsics
            # initialize
            Sinput_full = np.array(Sinput)
            Soutput_full = np.array(Soutput)
            if fintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
                Soutput_full = np.concatenate((Soutput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            if vintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
                Soutput_full = np.concatenate((Soutput_full, vintrinsics), axis=0)
            Moutput[ctarget,csource] = betavals

            # gradients in intrinsic2
            # dssq_dI, ssqd = gradients_in_intrinsic2(Sinput, Soutput, intrinsic1, intrinsic2, beta_int1, Minput, Moutput, dvali)
            dssq_dI, ssqd = gradients_in_vintrinsics(Sinput, Soutput, fintrinsic1, vintrinsics, beta_int1,
                                     Minput, Moutput, dvali, fintrinsic_count, vintrinsic_count)
            ssqd_record += [ssqd]
            dssq_dbeta1, ssqd = gradients_in_beta1(Sinput, Soutput, fintrinsic1, vintrinsics, beta_int1, Minput, Moutput, dval, fintrinsic_count, vintrinsic_count)
            ssqd_record += [ssqd]

            # apply the changes
            vintrinsics -= alphai * dssq_dI
            beta_int1 -= alphab * dssq_dbeta1

            vintrinsics_record.append({'i2':vintrinsics})

            Sinput_full = np.array(Sinput)
            Soutput_full = np.array(Soutput)
            if fintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
                Soutput_full = np.concatenate((Soutput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            if vintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
                Soutput_full = np.concatenate((Soutput_full, vintrinsics), axis=0)

            Moutput[ctarget,csource] = betavals
            fit, Soutput_temp = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)

            err = Sinput_full[:nregions,:] - fit[:nregions,:]
            cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
            ssqd_new = np.sum(err ** 2) + Lweight * cost  # L1 regularization

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_temp, Minput, Moutput)
            results_record.append({'Sinput':fit, 'Soutput':Soutput_temp})

            if ssqd_new >= ssqd:
                alphab *= 0.5
                alphai *= 0.5
                # revert back to last good values
                vintrinsics = copy.deepcopy(lastgood_vintrinsics)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                dssqd = ssqd - ssqd_new
                dssq_record = np.ones(3)   # reset the count
                dssq_count = 0
                print('intrinsics:  iter {} alpha {:.3e}  delta ssq {:.4f} - no update'.format(iter,alphai,-dssqd))
            else:
                # save the good values
                lastgood_vintrinsics = copy.deepcopy(vintrinsics)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)
                Soutput = Soutput_temp[:nregions,:]

                dssqd = ssqd - ssqd_new
                ssqd = ssqd_new

                dssq_count += 1
                dssq_count = np.mod(dssq_count, 3)
                dssq_record[dssq_count] = 100.0*dssqd/ssqd_starting
                if np.max(dssq_record) < 0.01:  converging = False

                print('intrinsics:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent'.format(iter,alphai,-dssqd,100.0*ssqd/ssqd_starting))

        # starting point for optimizing betavals ---------------------------------------------------------------
        alpha = initial_alpha
        alphai = initial_alphai
        alphab = initial_alphab
        Lweight = initial_Lweight
        dval = initial_dval
        dvali = initial_dvali

        # initialize values at the start of each pass------------------
        Sinput_full = np.array(Sinput)
        Soutput_full = np.array(Soutput)
        if fintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            Soutput_full = np.concatenate((Soutput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        if vintrinsic_count > 0:
            Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
            Soutput_full = np.concatenate((Soutput_full, vintrinsics), axis=0)

        Moutput[ctarget,csource] = betavals

        # starting point for optimizing betavals with given intrinsics----------------------------------------------------
        # fit, Soutput_full = network_fit(Soutput_full, Minput, Moutput)
        # fit, Soutput_full = network_approach_method(Soutput_full, Minput, Moutput, nregions)
        # fit, Soutput_full = network_descent_L1(Soutput_full, Minput, Moutput, nregions)   # update with more accurate method again
        fit, Soutput_full = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)
        Soutput = Soutput_full[:nregions,:]

        # fit, Soutput_temp = network_fit(Soutput_full, Minput, Moutput)   # use the quick method for estimating ssqd start
        # fit, Soutput_temp = network_approach_method(Soutput_full, Minput, Moutput, nregions)
        fit, Soutput_temp = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)
        err = Sinput_full[:nregions,:] - fit[:nregions,:]
        cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
        ssqd = np.sum(err ** 2) + Lweight * cost  # L1 regularization

        nitermax = 100
        alpha_limit = 1.0e-6

        iter = 0
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        while alpha > alpha_limit  and iter < nitermax and converging:
            iter += 1
            Sinput_full = np.array(Sinput)
            Soutput_full = np.array(Soutput)
            if fintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
                Soutput_full = np.concatenate((Soutput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
            if vintrinsic_count > 0:
                Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
                Soutput_full = np.concatenate((Soutput_full, vintrinsics), axis=0)

            Moutput[ctarget,csource] = betavals

            dssq_db, ssqd = gradients_for_betavals(Sinput, Soutput, fintrinsic1, vintrinsics, beta_int1, Minput, Moutput,
                                             betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count)
            ssqd_record += [ssqd]

            # apply the changes
            betavals -= alpha * dssq_db

            betavals[betavals >= betalimit] = betalimit
            betavals[betavals <= -betalimit] = -betalimit

            Moutput[ctarget,csource] = betavals
            fit, Soutput_temp = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)

            err = Sinput_full[:nregions,:] - fit[:nregions,:]
            cost = np.sum(np.abs(beta_int1)) + np.sum(np.abs(vintrinsics)) + np.sum(np.abs(betavals))
            ssqd_new = np.sum(err ** 2) + Lweight * cost  # L1 regularization

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
            results_record.append({'Sinput':fit, 'Soutput':Soutput_temp})

            if ssqd_new >= ssqd:
                alpha *= 0.5
                # revert back to last good values
                betavals = copy.deepcopy(lastgood_betavals)
                dssqd = ssqd - ssqd_new
                dssq_record = np.ones(3) # reset the count
                dssq_count = 0
                print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
            else:
                # save the good values
                lastgood_betavals = copy.deepcopy(betavals)
                Soutput = Soutput_temp[:nregions,:]

                dssqd = ssqd - ssqd_new
                ssqd = ssqd_new

                dssq_count += 1
                dssq_count = np.mod(dssq_count, 3)
                dssq_record[dssq_count] = 100.0*dssqd/ssqd_starting
                if np.max(dssq_record) < 0.01:  converging = False

                print('beta vals:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent'.format(iter,alpha,-dssqd,100.0*ssqd/ssqd_starting))
    # now repeat it ...

    # show results
    betavals = copy.deepcopy(lastgood_betavals)
    vintrinsics = copy.deepcopy(lastgood_vintrinsics)
    beta_int1 = copy.deepcopy(lastgood_beta_int1)

    Sinput_full = np.array(Sinput)
    Soutput_full = np.array(Soutput)
    if fintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
        Soutput_full = np.concatenate((Soutput_full, beta_int1 * fintrinsic1[np.newaxis, :]), axis=0)
    if vintrinsic_count > 0:
        Sinput_full = np.concatenate((Sinput_full, vintrinsics), axis=0)
        Soutput_full = np.concatenate((Soutput_full, vintrinsics), axis=0)

    Moutput[ctarget, csource] = betavals

    fit, Soutput_full = network_eigenvalue_method(Soutput_full, Minput, Moutput, nregions)
    err = Sinput_full[:nregions, :] - fit[:nregions, :]

    regionnum1 = 0
    regionnum2 = 7
    window1 = 24
    window2 = 25

    nruns = nruns_per_person[nperson]
    tsize = (tsize_total/nruns).astype(int)

    tc = Sinput_full[regionnum1,:]
    tc1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
    tc = fit[regionnum1,:]
    tcf1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)

    plt.close(window1)
    fig = plt.figure(window1, figsize=(12.5, 3.5), dpi=100)
    plt.plot(range(tsize),tc1,'-ob')
    plt.plot(range(tsize),tcf1,'-xr')

    tc = Sinput_full[regionnum2,:]
    tc2 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
    tc = fit[regionnum2,:]
    tcf2 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)

    plt.close(window2)
    fig = plt.figure(window2, figsize=(12.5, 3.5), dpi=100)
    plt.plot(range(tsize),tc2,'-ob')
    plt.plot(range(tsize),tcf2,'-xr')

    # plot intrinsic2
    tc = vintrinsics[0,:]
    tc1 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)
    tc = vintrinsics[1,:]
    tc12 = np.mean(np.reshape(tc, (nruns, tsize)), axis=0)

    plt.close(20)
    fig = plt.figure(20, figsize=(12.5, 3.5), dpi=100)
    plt.subplot(2,1,1)
    plt.plot(range(tsize),tc1,'-og')
    plt.subplot(2,1,2)
    plt.plot(range(tsize),tc2,'-og')

    columns = [name[:3] +' in' for name in rnamelist]
    rows = [name[:3] for name in rnamelist]
    if fintrinsic_count > 0:
        rows += ['fint1']
        columns += ['fint1 in']
    if vintrinsic_count > 0:
        for vv in range(vintrinsic_count):
            rows += ['vint{}'.format(vv+1)]
            columns += ['vint{} in'.format(vv+1)]

    df = pd.DataFrame(Moutput,columns = columns, index = rows)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.2f}'.format
    print(df)

    R1 = np.corrcoef(Sinput_full[regionnum1, :], fit[regionnum1, :])
    Z1 = np.arctanh(R1[0, 1]) * np.sqrt(tsize_total-3)
    results_text1 = 'person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, regionnum1, R1[0, 1], Z1)
    print(results_text1)

    R2 = np.corrcoef(Sinput_full[regionnum2, :], fit[regionnum2, :])
    Z2 = np.arctanh(R2[0, 1]) * np.sqrt(tsize_total-3)
    results_text2 = 'person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, regionnum2, R2[0, 1], Z2)
    print(results_text2)

    entry = {'Sinput':Sinput_full, 'Soutput':Soutput_full, 'beta_int1':beta_int1, 'Moutput':Moutput, 'Minput':Minput,
             'rtext1':results_text1, 'rtext2':results_text2}
    SEMresults.append(copy.deepcopy(entry))

    stoptime = time.ctime()

np.save(SEMresultsname, SEMresults)
print('finished SEM at {}'.format(time.ctime()))
print('     started at {}'.format(starttime))


#------for checking results---------------------
check_results = False
if check_results:
    SEMresults_load = np.load(SEMresultsname, allow_pickle=True)

    resultscheck = np.zeros((NP,4))
    # for nperson in range(NP):
    person_list = [41, 48, 32, 21, 10]

    for nperson in person_list:
        nruns = nruns_per_person[nperson]
        Sinput = SEMresults_load[nperson]['Sinput']
        Soutput = SEMresults_load[nperson]['Soutput']
        Minput = SEMresults_load[nperson]['Minput']
        Moutput = SEMresults_load[nperson]['Moutput']
        # fit, Soutput_temp = network_fit(Soutput, Minput, Moutput)
        # fit, Soutput_temp = network_descent_L1(Soutput, Minput, Moutput, nregions)
        # fit, Soutput_temp = network_approach_method(Soutput, Minput, Moutput, nregions)
        fit, Soutput_temp = network_eigenvalue_method(Soutput, Minput, Moutput, nregions)
        nr, tsize_total = np.shape(Soutput)
        tsize = (tsize_total/nruns).astype(int)

        region1 = 0
        region2 = 5
        region3 = 7
        nametag2 = r'_cord_NRM_PAG'

        rtarget = 5
        rtarget2 = 4
        rsource1 = 0
        rsource2 = 7
        rsource3 = 2
        nametag1 = r'NRMinput'

        window1 = 24
        window2 = 25

        # inputs to NRM as example
        plt.close(window1)
        # fig1 = plt.figure(window1, figsize=(12, 9), dpi=100)
        fig1, ((ax1,ax2), (ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharey=True, figsize=(12, 9), dpi=100, num = window1)
        # ax1 = fig.add_subplot(3,2,4, sharey = True)

        tc = Sinput[rtarget, :]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[rtarget, :]
        tcf = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax4.plot(tc1, '-ob')
        ax4.plot(tcf, '-xr')
        ax4.set_title('target input {}'.format(rnamelist[rtarget]))

        tc = Sinput[rtarget2, :]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[rtarget2, :]
        tc2f = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax2.plot(tc2, '-ob')
        ax2.plot(tc2f, '-xr')
        ax2.set_title('target input {}'.format(rnamelist[rtarget2]))


        tc = Soutput[rsource1, :]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Soutput[rsource2, :]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Soutput[rsource3, :]
        tc3 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax1.plot(tc1, '-xr')
        ax1.set_title('source output {}'.format(rnamelist[rsource1]))
        ax3.plot(tc2, '-xr')
        ax3.set_title('source output {}'.format(rnamelist[rsource2]))
        ax5.plot(tc3, '-xr')
        ax5.set_title('source output {}'.format(rnamelist[rsource3]))

        p,f = os.path.split(SEMresultsname)
        svgname = os.path.join(p,'Person{}_'.format(nperson) + nametag1 + '.svg')
        plt.savefig(svgname)

        # show C6RD, NRM, and PAG as examples (inputs real and fit)
        plt.close(window2)
        # fig2 = plt.figure(window2, figsize=(12, 6), dpi=100)
        fig2, (ax1b,ax2b,ax3b) = plt.subplots(3,sharey=False, figsize=(12, 6), dpi=100, num = window2)
        # ax1 = fig.add_subplot(3,2,4, sharey = True)

        tc = Sinput[region1, :]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[region1, :]
        tcf1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sinput[region2, :]
        tc2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[region2, :]
        tcf2 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sinput[region3, :]
        tc3 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = fit[region3, :]
        tcf3 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        ax1b.plot(tc1, '-ob')
        ax1b.plot(tcf1, '-xr')
        ax1b.set_title('target {}'.format(rnamelist[region1]))
        ax2b.plot(tc2, '-ob')
        ax2b.plot(tcf2, '-xr')
        ax2b.set_title('target {}'.format(rnamelist[region2]))
        ax3b.plot(tc3, '-ob')
        ax3b.plot(tcf3, '-xr')
        ax3b.set_title('target {}'.format(rnamelist[region3]))

        p,f = os.path.split(SEMresultsname)
        svgname = os.path.join(p,'Person{}_'.format(nperson) + nametag2 + '.svg')
        plt.savefig(svgname)


        R1 = np.corrcoef(Sinput[region1, :], fit[region1, :])
        Z1 = np.arctanh(R1[0,1]) * np.sqrt(tsize_total-3)
        print('person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, region1,R1[0,1],Z1))
        resultscheck[nperson,0] = R1[0,1]
        resultscheck[nperson,1] = Z1

        R2 = np.corrcoef(Sinput[region2, :], fit[region2, :])
        Z2 = np.arctanh(R2[0,1]) * np.sqrt(tsize_total-3)
        print('person {} region {}   R = {:.3f}  Z = {:.2f}'.format(nperson, region2,R2[0,1],Z2))
        resultscheck[nperson,2] = R2[0,1]
        resultscheck[nperson,3] = Z2

        # intrinsic2
        tc = Soutput[nregions+1,:]
        tc1 = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)
        tc = Sinput[nregions+1,:]
        tc1b = np.mean(np.reshape(tc, (nruns, tsize)),axis=0)

        plt.close(4)
        fig = plt.figure(4, figsize=(12.5, 3.5), dpi=100)
        plt.plot(tc1, '-og')
        plt.plot(tc1b, '-r')

        #
        columns = [name[:3] + ' in' for name in rnamelist]
        rows = [name[:3] for name in rnamelist]
        if fintrinsic_count > 0:
            rows += ['fint1']
            columns += ['fint1 in']
        if vintrinsic_count > 0:
            for vv in range(vintrinsic_count):
                rows += ['vint{}'.format(vv + 1)]
                columns += ['vint{} in'.format(vv + 1)]



        df = pd.DataFrame(Moutput, columns=columns, index=rows)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        pd.options.display.float_format = '{:.2f}'.format
        print(df)

        p,f = os.path.split(SEMresultsname)
        xlname = os.path.join(p,'Person{}_Moutput_v2.xlsx'.format(nperson))
        df.to_excel(xlname)


    Mrecord = np.zeros((12,12,NP))
    for nperson in range(NP):
        Sinput = SEMresults_load[nperson]['Sinput']
        Soutput = SEMresults_load[nperson]['Soutput']
        Minput = SEMresults_load[nperson]['Minput']
        Moutput = SEMresults_load[nperson]['Moutput']
        Mrecord[:,:,nperson] = Moutput

    Rrecord = np.zeros((10,10))
    R2record = np.zeros((10,10))
    for aa in range(10):
        for bb in range(10):
            m = Mrecord[aa,bb,:]
            if np.var(m) > 0:
                R = np.corrcoef(covariates2, m)
                Rrecord[aa,bb] = R[0,1]
                b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], covariates2[np.newaxis, :])
                R2record[aa,bb] = R2

    x = np.argmax(np.abs(R2record))
    aa,bb = np.unravel_index(x, np.shape(R2record))
    # aa,bb = (6,7)
    m = Mrecord[aa,bb,:]
    plt.close(35)
    fig = plt.figure(35), plt.plot(covariates2, m, 'ob')
    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis, :], covariates2[np.newaxis, :])
    plt.plot(covariates2, fit[0, :], '-b')



# # test matrix concepts
# n1 = 4
# n2 = 2
# M = np.zeros((n1+n2,n1+n2))
# M[:n1,:n1] = np.random.randn(n1,n1)
# M[n1:,n1:] = np.eye(n2,n2)
# # mixing part is crtical:
# M[2,n1+1] = 1
#
# # upper part
# w1,v1 = np.linalg.eig(M[:n1,:n1])
#
# # lower part
# w2,v2 = np.linalg.eig(M[n1:,n1:])
#
# # total
# w3,v3 = np.linalg.eig(M)


check_connections = False
if check_connections:
    nregions = len(nclusterlist)
    ncombo_set = np.floor(nregions/2).astype(int)
    nleaveout = nregions-ncombo_set
    nclusterlist = np.array(nclusterlist)

    list1 = list(range(ncombo_set))
    ncombinations = np.prod(nclusterlist)
    ncombinations1 = np.prod(nclusterlist[list1])

    # list3 = np.sort(np.random.choice(nregions,ncombo_set, replace=False))
    # list4 = np.sort(np.random.choice(nregions,ncombo_set, replace=False))

    EVR1 = np.zeros((NP,ncombinations1,3))
    EVR2 = np.zeros((NP,ncombinations,3))
    nkeep = 100
    xlist = np.zeros((NP,nkeep))  # keep a record of best 1st round picks for each person

    for nperson in range(NP):
        starttime = time.ctime()
        print('starting person {} at {}'.format(nperson, time.ctime()))
        tp = tplist_full[timepoint][nperson]['tp']
        tcdata_centered_person = tcdata_centered[:, tp]
        tsize_total = len(tp)
        nruns = nruns_per_person[nperson]

        # set 1
        cnums = np.zeros(ncombo_set).astype(int)
        full_rnum_base = get_overall_num(nclusterlist, list1, cnums)
        full_rnum_base = np.array(full_rnum_base).astype(int)

        print('     part 1 at {}'.format(time.ctime()))
        for nc in range(ncombinations1):
            cnums = ind2sub_ndims(nclusterlist[list1], nc)
            clusterlist = np.array(cnums) + full_rnum_base
            Sinput = tcdata_centered_person[clusterlist,:]

            pca = PCA(n_components=3)
            pca.fit(Sinput)
            EVR1[nperson,nc,:] = pca.explained_variance_ratio_

        # save a record of the best finds so far
        evr_values = EVR1[nperson,:,0]
        x = np.argsort(-evr_values)
        xlist[nperson,:] = x[:nkeep]

    print('collect the best starts for each person  {}'.format(time.ctime()))
    x2 = xlist[:,0]

    fixedindices = list1
    full_rnum_base = get_overall_num(nclusterlist, list(range(nregions)), np.zeros(nregions))
    full_rnum_base = np.array(full_rnum_base).astype(int)
    for nperson in range(NP):
        print('starting person {} at {}'.format(nperson, time.ctime()))
        # search through the top starting combinations
        tp = tplist_full[timepoint][nperson]['tp']
        tcdata_centered_person = tcdata_centered[:, tp]
        for ss, x in enumerate(x2):
            cnums = ind2sub_ndims(nclusterlist[list1], x)
            fixedvals = cnums
            w = all_flat_indices_nfixed(nclusterlist, fixedindices, fixedvals)
            for nc in w:
                cnums = ind2sub_ndims(nclusterlist, nc)
                clusterlist = np.array(cnums) + full_rnum_base
                Sinput = tcdata_centered_person[clusterlist,:]

                pca = PCA(n_components=3)
                pca.fit(Sinput)
                EVR2[nperson,nc,:] = pca.explained_variance_ratio_

    # look for the best combination based on whole set
    p,f = os.path.split(SEMresultsname)
    EVRname = os.path.join(p,'explained_variance_PCA_2.npy')
    np.save(EVRname, EVR2)

    count = np.count_nonzero(EVR2[:,:,0],axis=0)
    totalval = np.sum(EVR2[:,:,0],axis=0)
    x = np.where(count < 4)[0]
    totalval[x] = 0    # exclude values with too few samples
    nonzeroavg = totalval/(count + 1e-6)

    x = np.argmax(nonzeroavg)     # find where the average of the samles is the greatest value
    cnums = ind2sub_ndims(nclusterlist, x)

    # check cnums result------------------------
    full_rnum_base = get_overall_num(nclusterlist, list(range(nregions)), np.zeros(nregions))
    full_rnum_base = np.array(full_rnum_base).astype(int)
    clusterlist = np.array(cnums) + full_rnum_base

    EVRcheck = np.zeros((NP,3))
    for nperson in range(NP):
        tp = tplist_full[timepoint][nperson]['tp']
        tcdata_centered_person = tcdata_centered[:, tp]
        Sinput = tcdata_centered_person[clusterlist, :]
        pca = PCA(n_components=3)
        pca.fit(Sinput)
        EVRcheck[nperson, :] = pca.explained_variance_ratio_


    EVRname = os.path.join(p,'explained_variance_check.npy')
    np.save(EVRname, EVRcheck)


check_Moutput = False
if check_Moutput:
    # check it
    SEMresults_load = np.load(SEMresultsname, allow_pickle=True)
    NP = len(SEMresults_load)
    Moutput = SEMresults_load[0]['Moutput']
    nr1, nr2 = np.shape(Moutput)
    Mrecord = np.zeros((nr1,nr2,NP))
    for nperson in range(NP):
        Sinput = SEMresults_load[nperson]['Sinput']
        Soutput = SEMresults_load[nperson]['Soutput']
        Minput = SEMresults_load[nperson]['Minput']
        Moutput = SEMresults_load[nperson]['Moutput']
        Mrecord[:,:,nperson] = Moutput

    Mpos = np.zeros(np.shape(Mrecord))
    Mpos[Mrecord > 0] = 1
    Mneg = np.zeros(np.shape(Mrecord))
    Mneg[Mrecord < 0] = 1
    Mposneg = np.sum(Mpos, axis = 2) - np.sum(Mneg, axis = 2)

    columns = [name[:3] + ' in' for name in rnamelist]
    columns += ['int1 in', 'int2 in']
    rows = [name[:3] for name in rnamelist]
    rows += ['int1', 'int2']

    df = pd.DataFrame(Mposneg, columns=columns, index=rows)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.0f}'.format
    print(df)

    p, f = os.path.split(SEMresultsname)
    xlname = os.path.join(p, 'Moutput_pos_neg_counts.xlsx')
    df.to_excel(xlname)
