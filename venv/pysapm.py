# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')

# calculate eigenvectors and intrinsic inputs
# then calculate matrix of beta values from those
# Sinput = Minput @ Sconn
#  Sconn = Mconn @ Sconn    - eigenvalue problem
# for every time point, each connection value is determined by a scaled value of the
# sum of the eigenvectors
# so Sconn =  Meigv @ Mintrinsics
#
# new concept for V5
#   - calculate the intrinsic inputs based on the choice of beta values to test
#   Mconn is the mixing matrix of betavalues for each connection in the network
#   Sconn = Mconn @ Sconn
#   Meigv is the matrix of eigenvectors of Mconn, corrsponding to each intrinsic input
#       - these vectors are scaled so the value corresponding to the intrinsic input is equal to 1
#   Mintrinsics is the matrix of intrinsic values
#   Meigv is size [ncon x Nintrinsic]
#   Mintrinsic is size [Nintrinsic x tsize]
#   Sconn = Meigv @ Mintrinsics
#   Sinput = Minput @ Sconn
#   therefore  Mintrinsics = inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sinput

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
# import parallel_functions as pf
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib
import load_templates
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import HuberRegressor


plt.rcParams.update({'font.size': 10})


from scipy.signal import butter, lfilter
from scipy.signal import freqs

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
    for nn in range(nregions):
        sem_region_list.append(dnclusters.loc[nn,'name'])
        cname = dnclusters.loc[nn,'name']
        if 'vintrinsic' in cname:  vintrinsic_count += 1
        if 'fintrinsic' in cname:  fintrinsic_count += 1
        entry = {'name':dnclusters.loc[nn,'name'],'nclusters':dnclusters.loc[nn,'nclusters']}
        nclusterlist.append(entry)

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

    return network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count

def sapm_error_function(Sinput,fit,Lweight,betavals,beta_int1, Mintrinsic):
    # critical point:
    # the cost or error function used for gradient descent must include scaling to offset
    # differences in variance between different regions
    # Without this scaling regions with lower variance are given greater weighting in the balance
    # to determine optimal fit parameters, at the expense of fitting to higher variance regions.
    # Scaling the cost or error funciton by the variance in each region avoid this problem.
    # IF different cost or error functions are used, they must take the region variance into consideration.
    #

    R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
    R2avg = np.mean(R2list)
    R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

    error = np.sum( np.sum((Sinput - fit)**2, axis=1) / np.var(Sinput, axis=1))

    all_bvals = np.append(betavals,beta_int1)
    cost = np.mean(np.abs(all_bvals)) # L1 regularization
    cost2 = np.abs(R2avg-R2total)   # ideally, want R2avg and R2total to be similar
    # cost = np.mean(all_bvals**2) # L2 regularization

    ssqd = error + Lweight * (cost+cost2)
    return ssqd


def gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight):
    # calculate change in error term with small changes in betavalues
    # include beta_int1
    nbetavals = len(betavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
    ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

    # gradients for betavals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        b = copy.deepcopy(betavals)
        b[nn] += dval
        Mconn[ctarget, csource] = copy.deepcopy(b)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp = sapm_error_function(Sinput, fit, Lweight, b, beta_int1, Mintrinsic)
        dssq_db[nn] = (ssqdp - ssqd) / dval

    # gradients for beta_int1
    b = copy.deepcopy(beta_int1)
    b += dval
    Mconn[ctarget, csource] = betavals
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1)
    ssqdp = sapm_error_function(Sinput, fit, Lweight, betavals, b, Mintrinsic)
    dssq_dbeta1 = (ssqdp - ssqd) / dval

    return dssq_db, ssqd, dssq_dbeta1


def update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight, alphalist, alphabint, latent_flag = []):
    # calculate change in error term with small changes in betavalues
    # include beta_int1
    if len(latent_flag) < len(betavals): latent_flag = np.zeros(len(betavals))
    nbetavals = len(betavals)
    updatebflag = np.zeros(nbetavals)
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
    ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

    # print('update_betavals_sequentially: first ssqd: {:.3f}'.format(ssqd))

    # gradients for betavals
    dssq_db = np.zeros(nbetavals)
    for nn in range(nbetavals):
        # if latent_flag[nn] == 0:
        b = copy.deepcopy(betavals)
        b[nn] += dval
        Mconn[ctarget, csource] = copy.deepcopy(b)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        ssqdp = sapm_error_function(Sinput, fit, Lweight, b, beta_int1, Mintrinsic)
        dssq_db[nn] = (ssqdp - ssqd) / dval

        b[nn] = betavals[nn] - alphalist[nn]*dssq_db[nn]
        Mconn[ctarget, csource] = copy.deepcopy(b)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        ssqd_new = sapm_error_function(Sinput, fit, Lweight, b, beta_int1, Mintrinsic)

        # print('    betaval {}  ssqd_new: {:.3f}'.format(nn,ssqd_new))
        if ssqd_new < ssqd:
            betavals = copy.deepcopy(b)
            ssqd = copy.deepcopy(ssqd_new)
            updatebflag[nn] = 1
        else:
            updatebflag[nn] = 0
            alphalist[nn] *= 0.5
    # print('    after update ssqd: {:.3f}'.format(ssqd))

    # gradients for beta_int1
    b = copy.deepcopy(beta_int1)
    b += dval
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, b, fintrinsic1)
    ssqdp = sapm_error_function(Sinput, fit, Lweight, betavals, b, Mintrinsic)
    dssq_dbeta1 = (ssqdp - ssqd) / dval

    b = beta_int1 - alphabint * dssq_dbeta1
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             b, fintrinsic1)
    ssqd_new = sapm_error_function(Sinput, fit, Lweight, betavals, b, Mintrinsic)

    if ssqd_new < ssqd:
        beta_int1 = copy.deepcopy(b)
        ssqd = copy.deepcopy(ssqd_new)
        updatebintflag = 1
    else:
        updatebintflag = 0
        alphabint *= 0.5

    # print('     after bint update ssqd: {:.3f}'.format(ssqd))

    # final result
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)

    ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

    # print('     final ssqd: {:.3f}'.format(ssqd))
    return betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint


#
# def network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon):
#     #
#     # Soutput_full = Moutput @ Soutput_full
#     # find Soutput_working with given starting values
#     # the form of Moutput is a block matrix with the upper nregions x nregions section
#     # giving the beta values
#     # the lower nintrinsic x nintrinsic portion is an identity matrix
#     # and the upper right nregions x nintrinsic porition is the mixing from the intrinsics
#     # to the regions
#     # This form ensures that there are are number of eigenvalues = 1, and the number
#     # is equal to nintrinsic
#     # the corresponding eigenvectors have non-zero values for the intrinsic inputs and for
#     # other regions only if there is mixing between them
#     nr,nt = np.shape(Sconn_full)
#     nintrinsics = nr-ncon
#
#     det = np.linalg.det(Mconn)
#     w,v = np.linalg.eig(Mconn)
#
#     # Moutput @ v[:,a] = w[a]*v[:,a]
#
#     # check that intrinsics have eigenvalues = 1 (or close to it)
#     # assume that the eigenvalues, eigenvectors are always ordered the same as Moutput
#     check = np.zeros(nintrinsics)
#     tol = 1e-4
#     for nn in range(nintrinsics):
#         check[nn] = np.abs(w[nn+ncon]-1.0) < tol
#
#     if np.sum(check) < nintrinsics:
#         print('--------------------------------------------------------------------------------')
#         print('Error:  network_eigenvalue_method:  solution to network fitting cannot be found!')
#         print('--------------------------------------------------------------------------------')
#     else:
#         # M v = a v
#         fit1 = np.zeros((ncon,nt))
#         for nn in range(nintrinsics):
#             # do this for each intrinsic:
#             eval = np.real(w[nn+ncon])
#             evec = np.real(v[:,nn+ncon])
#             for tt in range(nt):
#                 scale = Sconn_full[nn+ncon,tt]/evec[nn+ncon]
#                 fit1[:ncon,tt] += evec[:ncon]*scale
#
#         Sconn_working = copy.deepcopy(Sconn_full)
#         Sconn_working[:ncon] = fit1
#         fit = Minput @ Sconn_working
#
#     return fit, Sconn_working


def network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1):
    # # Sinput[:nregions,:] = Minput @ Meigv @ Mintrinsic   #  [nr x tsize] = [nr x ncon] @ [ncon x Nint] @ [Nint x tsize]
    # # Mintrinsic = np.linalg.inv(Meigv.T @ Minput.T @ Minput @ Meigv) @ Meigv.T @ Minput.T @ Sin
    # fit based on eigenvectors alone, with intrinsic values calculated
    nregions,tsize_total = np.shape(Sinput)
    Nintrinsic = fintrinsic_count + vintrinsic_count
    e,v = np.linalg.eig(Mconn)    # Mconn is nbeta x nbeta   where nbeta = ncon + Nintrinsic
    Meigv = np.real(v[:,-Nintrinsic:])
    # scale to make the term corresponding to each intrinsic = 1
    for aa in range(Nintrinsic):
        Meigv[:,aa] = Meigv[:,aa]/Meigv[(-Nintrinsic+aa),aa]
    M1 = Minput @ Meigv

    # Sinput = M1 @ Mintrinsic
    # Mintrinsic = inv(M1.T @ M1) @ M1.T @ Sinput
    if fintrinsic_count > 0:
        # this part needs to be fixed
        Mintrinsic = np.zeros((Nintrinsic, tsize_total))
        Mint_variable = np.linalg.inv(M1[:,1:].T @ M1[:,1:]) @ M1[:,1:].T @ Sinput
        Mint_fixed = beta_int1*fintrinsic1
        Mintrinsic[0,:] = Mint_fixed
        Mintrinsic[1:,:] = Mint_variable
    else:
        Mintrinsic = np.linalg.inv(M1.T @ M1) @ M1.T @ Sinput

    fit = Minput @ Meigv @ Mintrinsic
    err = np.sum((Sinput - fit)**2)

    return fit, Mintrinsic, Meigv, err


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


def load_filtered_cluster_properties(clusterdataname, networkfile):
    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = cluster_data['cluster_properties']
    clusterregionlist = [cluster_properties[x]['rname'] for x in range(len(cluster_properties))]

    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)
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
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)

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
            et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)
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

    # latent_flag = np.zeros(len(ctarget))
    # found_latent_list = []
    # for nn in range(len(ctarget)):
    #     if csource[nn] >= ncon:
    #         if not csource[nn] in found_latent_list:
    #             latent_flag[nn] = 1
    #             found_latent_list += [csource[nn]]

    latent_flag = np.zeros(len(ctarget))
    # set all the latent flag values to zero, so that they are updated during the gradient descent method
    # found_latent_list = []
    # for nn in range(len(ctarget)):
    #     if csource[nn] >= ncon  and ctarget[nn] < ncon:
    #         found_latent_list += [csource[nn]]
    #         occurence = np.count_nonzero(found_latent_list == csource[nn])
    #         latent_flag[nn] = csource[nn]-ncon+1

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
                  'reciprocal_flag':reciprocal_flag}   #, 'tcdata_std':tcdata_std
    if normalizevar:
        SAPMparams['tcdata_std'] = tcdata_std
    print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
    np.save(SAPMparametersname, SAPMparams)


# prep data for single output model
#---------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def prep_data_sem_physio_model_SO(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint = 'all',
                                  epoch = 'all', fullgroup = False, normalizevar = False, filter_tcdata = False):
# model each region as having a single output that is common to all regions it projects to
    outputdir, f = os.path.split(SAPMparametersname)
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)

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
            et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
            et2 = (timepoint + np.floor(epoch / 2)).astype(int)
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

            if filter_tcdata:
                # low-pass filter the tcdata
                fs = 1.0  # use relative frequencies, to allow for different sampling rates
                cutoff = 0.2
                for ii in range(nregions):
                    tcdata_centered[ii, tp] = butter_lowpass_filter(tcdata_centered[ii, tp], cutoff, fs, order=5)

        tplist1.append({'tp': tpoints})

        if normalizevar:
            # normalize the data to have the same variance, for each person
            tcdata_std[:,nn] = np.std(tcdata_centered[:,tpoints],axis=1)
            scale_factor = np.repeat(tcdata_std[:,nn][:,np.newaxis],len(tpoints),axis=1)
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
    # Mconn = np.zeros((nbeta, nbeta))
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
                # conn2 = conn1
                # Mconn[conn1, conn2] = 1  # set the intrinsic beta values
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
            betaname = '{}_{}'.format(source, target)
            x = betanamelist.index(betaname)
            # Minput[target, x] = 1
            Minput[target, source] = 1

    # save parameters for looking at results later
    SAPMparams = {'betanamelist': betanamelist, 'beta_list': beta_list, 'nruns_per_person': nruns_per_person,
                 'nclusterstotal': nclusterstotal, 'rnamelist': rnamelist, 'nregions': nregions,
                 'cluster_properties': cluster_properties, 'cluster_data': cluster_data,
                 'network': network, 'fintrinsic_count': fintrinsic_count, 'vintrinsic_count': vintrinsic_count,
                 'fintrinsic_region':fintrinsic_region, 'sem_region_list': sem_region_list,
                 'nclusterlist': nclusterlist, 'tsize': tsize, 'tplist_full': tplist_full,
                 'tcdata_centered': tcdata_centered, 'ctarget':ctarget ,'csource':csource,
                 'Mconn':Mconn, 'Minput':Minput, 'timepoint':timepoint, 'epoch':epoch, 'latent_flag':latent_flag,
                  'reciprocal_flag':reciprocal_flag}   #, 'tcdata_std':tcdata_std
    if normalizevar:
        SAPMparams['tcdata_std'] = tcdata_std
    print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
    np.save(SAPMparametersname, SAPMparams)


#---------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def prep_null_data_sem_physio_model(nsamples, networkfile, regiondataname, clusterdataname, SAPMparametersname,
                                    timepoint = 'all', epoch = 'all', fullgroup = False, addglobalbias = False):
    outputdir, f = os.path.split(SAPMparametersname)
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)

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
    # NP = len(nruns_per_person)  # number of people in the data set

    # for null data sets, replace NP with nsamples--------------------------------
    NP = nsamples
    nruns = nruns_per_person[0]
    nruns_per_person = (nruns*np.ones(nsamples)).astype(int)
    nruns_total = np.sum(nruns_per_person)
    tcdata = np.random.randn(nclusterstotal, (tsize*nruns_total).astype(int))   # make a new tcdata set out of random values

    if addglobalbias:
        globalbias = np.random.randn(1, (tsize*nruns_total).astype(int))
        globalbias = np.repeat(globalbias,nclusterstotal,axis=0)
        tcdata += globalbias
    #-----------------------------------------------------------------------------

    # original method:
    # tcdata = []
    # for i in range(nregions):
    #     tc = region_properties[i]['tc']
    #     if i == 0:
    #         tcdata = tc
    #     else:
    #         tcdata = np.append(tcdata, tc, axis=0)

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

    # latent_flag = np.zeros(len(ctarget))
    # found_latent_list = []
    # for nn in range(len(ctarget)):
    #     if csource[nn] >= ncon:
    #         if not csource[nn] in found_latent_list:
    #             latent_flag[nn] = 1
    #             found_latent_list += [csource[nn]]

    latent_flag = np.zeros(len(ctarget))
    found_latent_list = []
    for nn in range(len(ctarget)):
        if csource[nn] >= ncon  and ctarget[nn] < ncon:
            found_latent_list += [csource[nn]]
            occurence = np.count_nonzero(found_latent_list == csource[nn])
            latent_flag[nn] = csource[nn]-ncon+1

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
                  'reciprocal_flag':reciprocal_flag}
    print('saving SAPM parameters to file: {}'.format(SAPMparametersname))
    np.save(SAPMparametersname, SAPMparams)


#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [], betascale = 0.01, verbose = True):
    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    initial_alpha = 0.1
    initial_Lweight = 1e-12
    initial_dval = 0.01
    nitermax = 200
    alpha_limit = 1.0e-4
    repeat_limit = 2
    repeat_count = 0

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
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    Mconn = SAPMparams['Mconn']
    Minput = SAPMparams['Minput']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    latent_flag = SAPMparams['latent_flag']
    reciprocal_flag = SAPMparams['reciprocal_flag']

    ntime, NP = np.shape(tplist_full)
    Nintrinsics = vintrinsic_count + fintrinsic_count
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
        # Sinput_scalefactor = np.zeros(len(clusterlist))
        for nc,cval in enumerate(clusterlist):
            tc1 = tcdata_centered[cval, tp]
            # Sinput_scalefactor[nc] = np.var(tc1)
            # tc1 /= np.var(tc1)
            Sinput.append(tc1)
        Sinput = np.array(Sinput)
        # Sinput is size:  nregions x tsize_total

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

            ftemp = fintrinsic_base[et1:et2]
            fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
            if np.var(ftemp) > 1.0e-3:
                Sint = Sinput[fintrinsic_region,:]
                Sint = Sint - np.mean(Sint)
                # need to add constant to fit values
                G = np.concatenate((fintrinsic1[np.newaxis, :],np.ones((1,tsize_total))),axis=0)
                b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
                beta_int1 = b[0]
            else:
                beta_int1 = 0.0
        else:
            beta_int1 = 0.0
            fintrinsic1 = []

        lastgood_beta_int1 = copy.deepcopy(beta_int1)

        # initialize beta values-----------------------------------
        if isinstance(betascale,str):
            if betascale == 'shotgun':
                nbeta = len(csource)
                beta_initial = betaval_init_shotgun(initial_Lweight, csource, ctarget, Sinput, Minput, Mconn, fintrinsic_count,
                                     vintrinsic_count, beta_int1, fintrinsic1, nreps=10000)
            else:
                # read saved beta_initial values
                b = np.load(betascale,allow_pickle=True).flat[0]
                beta_initial = b['beta_initial']
        else:
            beta_initial = betascale*np.random.randn(len(csource))

        beta_init_record.append({'beta_initial':beta_initial})

        # initalize Sconn
        betavals = copy.deepcopy(beta_initial) # initialize beta values at zero
        lastgood_betavals = copy.deepcopy(betavals)

        results_record = []
        ssqd_record = []

        alpha = copy.deepcopy(initial_alpha)
        Lweight = copy.deepcopy(initial_Lweight)
        dval = copy.deepcopy(initial_dval)

        Mconn[ctarget,csource] = copy.deepcopy(betavals)
        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        # fit, Sconn_full = network_eigenvalue_method(Sconn_full, Minput, Mconn, ncon)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

        ssqd_starting = copy.deepcopy(ssqd)
        ssqd_record += [ssqd]
        # ssqd_starting = 1e20   # start big

        iter = 0
        # vintrinsics_record = []
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        sequence_count = 0
        repeat_count = 0
        while alpha > alpha_limit and repeat_count < 1 and iter < nitermax and converging:
            iter += 1
            # gradients in betavals and beta_int1
            Mconn[ctarget, csource] = copy.deepcopy(betavals)
            fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                     vintrinsic_count, beta_int1, fintrinsic1)

            dssq_db, ssqd, dssq_dbeta1 = gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                                                fintrinsic_count, vintrinsic_count, beta_int1,
                                                                fintrinsic1, Lweight)
            ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
            ssqd_record += [ssqd]

            # fix some beta values at zero, if specified
            if len(fixed_beta_vals) > 0:
                dssq_db[fixed_beta_vals] = 0

            # apply the changes
            # limit the betaval changes
            dsmax = 0.1 / alpha
            dssq_db[dssq_db < -dsmax] = -dsmax
            dssq_db[dssq_db > dsmax] = dsmax

            betavals -= alpha * dssq_db
            beta_int1 -= alpha * dssq_dbeta1
            # beta_int1 = np.abs(beta_int1)   # limit beta_int1

            Mconn[ctarget, csource] = copy.deepcopy(betavals)
            fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                     vintrinsic_count, beta_int1, fintrinsic1)
            ssqd_new = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)
            # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

            R2list = [1-np.sum((Sinput[x,:]-fit[x,:])**2)/np.sum(Sinput[x,:]**2) for x in range(nregions)]
            R2total = np.mean(R2list)

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
            results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

            tol = 1e-6
            if ssqd_new > ssqd-tol:
                alpha *= 0.5
                # revert back to last good values
                betavals = copy.deepcopy(lastgood_betavals)
                beta_int1 = copy.deepcopy(lastgood_beta_int1)
                dssqd = ssqd - ssqd_new
                dssq_record = np.ones(3)  # reset the count
                dssq_count = 0
                sequence_count = 0

                # if alpha < alpha_limit and repeat_count < repeat_limit:
                #     repeat_count += 1
                #     alpha = initial_alpha

                print('{} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f}  delta ssq > 0  - no update'.format(nperson, iter, alpha,ssqd_new))
            else:
                # save the good values
                lastgood_betavals = copy.deepcopy(betavals)
                lastgood_beta_int1 = copy.deepcopy(beta_int1)

                dssqd = ssqd - ssqd_new
                ssqd = copy.deepcopy(ssqd_new)

                sequence_count += 1
                if sequence_count > 5:
                    alpha *= 1.3
                    sequence_count = 0

                dssq_count += 1
                dssq_count = np.mod(dssq_count, 3)
                # dssq_record[dssq_count] = 100.0 * dssqd / ssqd_starting
                dssq_record[dssq_count] = copy.deepcopy(dssqd)
                if np.max(dssq_record) < tol:  converging = False

            print('{} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f}  delta ssq {:.4f}  relative: {:.1f} percent  '
                  'R2 {:.3f}'.format(nperson, iter, alpha, ssqd, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
            # now repeat it ...

        # fit the results now to determine output signaling from each region
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'R2total':R2total, 'Mintrinsic':Mintrinsic, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
                 'Meigv':Meigv, 'betavals':betavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
                 'fintrinsic_base':fintrinsic_base}

        # person_results.append(entry)
        SAPMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()

    # # sort the results to be consistent across data sets
    # if NP > 1:
    #     SAPMresultsr = sort_SAPM_results2(SAPMresults, vintrinsic_count, fintrinsic_count, latent_flag)
    # else:
    #     SAPMresultsr = copy.deepcopy(SAPMresults)
    # np.save(SAPMresultsname, SAPMresultsr)

    np.save(SAPMresultsname, SAPMresults)
    print('finished SAPM at {}'.format(time.ctime()))
    print('     started at {}'.format(starttime))
    print('     results written to {}'.format(SAPMresultsname))
    return SAPMresultsname


#----------------------------------------------------------------------------------
# primary function--------------------------------------------------------------------
def sem_physio_model2(clusterlist, fintrinsic_base, SAPMresultsname, SAPMparametersname, fixed_beta_vals = [],
                      betascale = 0.01, nitermax = 100, verbose = True, initial_nitermax_stage1 = 20,
                      initial_nsteps_stage1 = 20):
    starttime = time.ctime()

    # initialize gradient-descent parameters--------------------------------------------------------------
    initial_alpha = 1e-2
    initial_Lweight = 1.0
    initial_dval = 0.01
    # nitermax = 300
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
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    Mconn = SAPMparams['Mconn']
    Minput = SAPMparams['Minput']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    latent_flag = SAPMparams['latent_flag']
    reciprocal_flag = SAPMparams['reciprocal_flag']

    ntime, NP = np.shape(tplist_full)
    Nintrinsics = vintrinsic_count + fintrinsic_count
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
        # Sinput_scalefactor = np.zeros(len(clusterlist))
        for nc,cval in enumerate(clusterlist):
            tc1 = tcdata_centered[cval, tp]
            # Sinput_scalefactor[nc] = np.var(tc1)
            # tc1 /= np.var(tc1)
            Sinput.append(tc1)
        Sinput = np.array(Sinput)

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

            ftemp = fintrinsic_base[et1:et2]
            fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
            if np.var(ftemp) > 1.0e-3:
                Sint = Sinput[fintrinsic_region,:]
                Sint = Sint - np.mean(Sint)
                # need to add constant to fit values
                G = np.concatenate((fintrinsic1[np.newaxis, :],np.ones((1,tsize_total))),axis=0)
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
                beta_initial = betaval_init_shotgun(initial_Lweight, csource, ctarget, Sinput, Minput, Mconn, fintrinsic_count,
                                     vintrinsic_count, beta_int1, fintrinsic1, nreps=10000)
                beta_initial = beta_initial[np.newaxis,:]
                nitermax_stage1 = 0
            else:
                # read saved beta_initial values
                b = np.load(betascale,allow_pickle=True).flat[0]
                beta_initial = b['beta_initial']
                beta_initial = beta_initial[np.newaxis,:]
                nitermax_stage1 = 0
            nsteps_stage1 = 1
            beta_initial[0,latent_flag > 0] = 1.0
        else:
            nsteps_stage1 = copy.deepcopy(initial_nsteps_stage1)
            beta_initial = betascale*np.random.randn(nsteps_stage1,nbeta)
            beta_initial[:,latent_flag > 0] = 1.0
            nitermax_stage1 = copy.deepcopy(initial_nitermax_stage1)

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

            alphalist = initial_alpha*np.ones(nbeta)
            alphabint = copy.deepcopy(initial_alpha)
            alphamax = copy.deepcopy(initial_alpha)
            Lweight = copy.deepcopy(initial_Lweight)
            dval = copy.deepcopy(initial_dval)

            # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
            Mconn[ctarget,csource] = copy.deepcopy(betavals)
            fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
            ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

            ssqd_starting = copy.deepcopy(ssqd)
            ssqd_old = copy.deepcopy(ssqd)
            ssqd_record += [ssqd]

            iter = 0
            converging = True
            dssq_record = np.ones(3)
            dssq_count = 0
            sequence_count = 0

            while alphamax > alpha_limit and iter < nitermax_stage1 and converging:
                iter += 1

                betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint = \
                    update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                                        fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
                                                        alphalist, alphabint, latent_flag)

                ssqd_record_stage1 += [ssqd]

                err_total = Sinput - fit
                Smean = np.mean(Sinput)
                errmean = np.mean(err_total)
                # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

                # R2list = [1-np.sum((Sinput[x,:]-fit[x,:])**2)/np.sum(Sinput[x,:]**2) for x in range(nregions)]
                R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
                R2avg = np.mean(R2list)
                R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

                # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
                results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
                atemp = np.append(alphalist,alphabint)
                alphamax = np.max(atemp)
                alphalist[alphalist < alpha_limit] = alpha_limit
                if alphabint < alpha_limit:  alphabint = copy.deepcopy(alpha_limit)

                ssqchange = ssqd - ssqd_old
                if np.abs(ssqchange) < 1e-5: converging = False

                if verbose:
                    print('SAPM  {} stage1 pass {} iter {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(nperson,
                                    ns, iter, np.mean(alphalist), ssqd, ssqchange, 100.*ssqd/ssqd_starting, R2avg, R2total))
                ssqd_old = copy.deepcopy(ssqd)
                # now repeat it ...
            stage1_ssqd[ns] = ssqd
            stage1_results.append({'betavals':betavals})
        # get the best betavals from stage1 so far ...
        x = np.argmin(stage1_ssqd)
        betavals = stage1_results[x]['betavals']

        # stage 2
        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        if verbose: print('starting stage 2 ....')
        lastgood_betavals = copy.deepcopy(betavals)
        alphalist = initial_alpha * np.ones(nbeta)
        alphabint = copy.deepcopy(initial_alpha)
        alphamax = copy.deepcopy(initial_alpha)
        Lweight = copy.deepcopy(initial_Lweight)
        dval = copy.deepcopy(initial_dval)

        # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

        ssqd_starting = copy.deepcopy(ssqd)
        ssqd_old = copy.deepcopy(ssqd)
        ssqd_record += [ssqd]

        iter = 0
        converging = True
        dssq_record = np.ones(3)
        dssq_count = 0
        sequence_count = 0

        while alphamax > alpha_limit and iter < nitermax and converging:
            iter += 1

            betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint = \
                update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                             fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
                                             alphalist, alphabint, latent_flag)

            # print('iter {}  ssqd = {:.3f}'.format(iter,ssqd))

            ssqd_record += [ssqd]

            err_total = Sinput - fit
            Smean = np.mean(Sinput)
            errmean = np.mean(err_total)
            # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)

            # R2list = [1-np.sum((Sinput[x,:]-fit[x,:])**2)/np.sum(Sinput[x,:]**2) for x in range(nregions)]
            R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
            R2avg = np.mean(R2list)
            R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)

            # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
            results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
            atemp = np.append(alphalist, alphabint)
            alphamax = np.max(atemp)
            alphalist[alphalist < alpha_limit] = alpha_limit
            if alphabint < alpha_limit:  alphabint = copy.deepcopy(alpha_limit)

            ssqchange = ssqd - ssqd_old
            if np.abs(ssqchange) < 1e-5: converging = False

            if verbose:
                print('SAPM  {} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  percent {:.1f}  R2 avg {:.3f}  R2 total {:.3f}'.format(
                        nperson,
                        iter, np.mean(alphalist), ssqd, ssqchange, 100. * ssqd / ssqd_starting, R2avg, R2total))
            ssqd_old = copy.deepcopy(ssqd)
            # now repeat it ...


        # fit the results now to determine output signaling from each region
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1)
        Sconn = Meigv @ Mintrinsic    # signalling over each connection

        entry = {'Sinput':Sinput, 'Sconn':Sconn, 'beta_int1':beta_int1, 'Mconn':Mconn, 'Minput':Minput,
                 'R2total':R2total, 'R2avg':R2avg, 'Mintrinsic':Mintrinsic, 'fintrinsic_count':fintrinsic_count, 'vintrinsic_count':vintrinsic_count,
                 'Meigv':Meigv, 'betavals':betavals, 'fintrinsic1':fintrinsic1, 'clusterlist':clusterlist,
                 'fintrinsic_base':fintrinsic_base}

        # person_results.append(entry)
        SAPMresults.append(copy.deepcopy(entry))

        stoptime = time.ctime()

    np.save(SAPMresultsname, SAPMresults)
    if verbose:
        print('finished SAPM at {}'.format(time.ctime()))
        print('     started at {}'.format(starttime))
        print('     results written to {}'.format(SAPMresultsname))
    return SAPMresultsname


#-------------------------------initialize betavals--------------------------------
def betaval_init_shotgun(Lweight, csource, ctarget, Sinput, Minput, Mconn, fintrinsic_count,
                                vintrinsic_count, beta_int1, fintrinsic1, nreps = 100000):
    search_record = []
    betascale = 0.2
    for rr in range(nreps):
        # initialize beta values at random values-----------------------------------
        betavals = betascale * np.random.randn(len(csource))  # initialize beta values at zero
        Mconn[ctarget, csource] = betavals
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                        vintrinsic_count, beta_int1, fintrinsic1)
        ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

        R2 = 1 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)
        search_record.append({'betavals': betavals, 'ssqd': ssqd, 'R2': R2})

    ssqd_list = np.array([search_record[x]['ssqd'] for x in range(nreps)])
    R2_list = np.array([search_record[x]['R2'] for x in range(nreps)])
    b_list = np.array([search_record[x]['betavals'] for x in range(nreps)])

    x = ssqd_list.argmin()
    best_betavals = b_list[x, :]
    print('best betavals gives R2 = {:.2f}'.format(R2_list[x]))

    return best_betavals

#
# #
# # gradient descent method per person
# Does this need to be in a separate module?
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# gradient descent method per person
def gradient_descent_per_person_original(data):
    nperson = data['nperson']
    clusterlist = data['clusterlist']
    tsize = data['tsize']
    tplist_full = data['tplist_full']
    nruns_per_person = data['nruns_per_person']
    nclusterlist = data['nclusterlist']
    Minput = data['Minput']
    fintrinsic_count = data['fintrinsic_count']
    fintrinsic_region = data['fintrinsic_region']
    vintrinsic_count = data['vintrinsic_count']
    epoch = data['epoch']
    timepoint = data['timepoint']
    tcdata_centered = data['tcdata_centered']
    ctarget = data['ctarget']
    csource = data['csource']
    latent_flag = data['latent_flag']
    Mconn = data['Mconn']
    ntime = data['ntime']
    NP = data['NP']
    epochnum = data['epochnum']
    fintrinsic_base = data['fintrinsic_base']
    initial_alpha = data['initial_alpha']
    initial_Lweight = data['initial_Lweight']
    initial_dval = data['initial_dval']
    alpha_limit = data['alpha_limit']
    nitermax = data['nitermax']
    fixed_beta_vals = data['fixed_beta_vals']
    verbose = data['verbose']
    beta_initial = data['beta_initial']

    # if verbose: print('starting person {} at {}'.format(nperson, time.ctime()))
    tp = tplist_full[epochnum][nperson]['tp']
    tsize_total = len(tp)
    nruns = nruns_per_person[nperson]

    # load the data for this person
    Sinput = []
    for nc, cval in enumerate(clusterlist):
        tc1 = tcdata_centered[cval, tp]
        Sinput.append(tc1)
    Sinput = np.array(Sinput)

    # Sinput is size:  nregions x tsize_total

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

        ftemp = fintrinsic_base[et1:et2]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
        if np.var(ftemp) > 1.0e-3:
            Sint = Sinput[fintrinsic_region, :]
            Sint = Sint - np.mean(Sint)
            # need to add constant to fit values
            G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
            b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
            beta_int1 = b[0]
        else:
            beta_int1 = 0.0
    else:
        beta_int1 = 0.0
        fintrinsic1 = []

    lastgood_beta_int1 = copy.deepcopy(beta_int1)

    # initalize Sconn
    betavals = copy.deepcopy(beta_initial)  # initialize beta values at zero
    lastgood_betavals = copy.deepcopy(betavals)

    results_record = []
    ssqd_record = []

    alpha = copy.deepcopy(initial_alpha)
    Lweight = copy.deepcopy(initial_Lweight)
    dval = copy.deepcopy(initial_dval)

    Mconn[ctarget, csource] = copy.deepcopy(betavals)

    # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
    ssqd_starting = copy.deepcopy(ssqd)
    ssqd_record += [ssqd]

    iter = 0
    converging = True
    dssq_record = np.ones(3)
    dssq_count = 0
    sequence_count = 0
    while alpha > alpha_limit and iter < nitermax and converging:
        iter += 1
        # gradients in betavals and beta_int1
        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        dssq_db, ssqd, dssq_dbeta1 = gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
                                                            fintrinsic_count, vintrinsic_count, beta_int1,
                                                            fintrinsic1, Lweight)
        ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
        ssqd_record += [ssqd]

        # fix some beta values at zero, if specified
        if len(fixed_beta_vals) > 0:
            dssq_db[fixed_beta_vals] = 0

        # apply the changes
        # limit the betaval changes
        dsmax = 0.1 / alpha
        dssq_db[dssq_db < -dsmax] = -dsmax
        dssq_db[dssq_db > dsmax] = dsmax

        betavals -= alpha * dssq_db
        beta_int1 -= alpha * dssq_dbeta1
        # beta_int1 = np.abs(beta_int1) # limit beta_int1

        Mconn[ctarget, csource] = copy.deepcopy(betavals)
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        ssqd_new = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)

        err_total = Sinput - fit
        Smean = np.mean(Sinput)
        errmean = np.mean(err_total)
        R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)
        if R2total < 0: R2total = 0.0

        # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
        results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})

        if ssqd_new >= ssqd:
            alpha *= 0.5
            # revert back to last good values
            betavals = copy.deepcopy(lastgood_betavals)
            beta_int1 = copy.deepcopy(lastgood_beta_int1)
            dssqd = ssqd - ssqd_new
            dssq_record = np.ones(3)  # reset the count
            dssq_count = 0
            sequence_count = 0
            if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
        else:
            # save the good values
            lastgood_betavals = copy.deepcopy(betavals)
            lastgood_beta_int1 = copy.deepcopy(beta_int1)

            dssqd = ssqd - ssqd_new
            ssqd = copy.deepcopy(ssqd_new)

            dssq_count += 1
            dssq_count = np.mod(dssq_count, 3)
            # dssq_record[dssq_count] = 100.0 * dssqd / ssqd_starting
            dssq_record[dssq_count] = copy.deepcopy(dssqd)
            if np.max(dssq_record) < 1e-6:  converging = False

        if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent  '
                          'R2 {:.3f}'.format(iter, alpha, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
        # now repeat it ...

    # fit the results now to determine output signaling from each region
    Mconn[ctarget, csource] = copy.deepcopy(betavals)
    fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
                                                             beta_int1, fintrinsic1)
    Sconn = Meigv @ Mintrinsic  # signalling over each connection

    entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
             'R2total': R2total, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'betavals': betavals,
             'fintrinsic1': fintrinsic1, 'fintrinsic_base': fintrinsic_base}

    return entry

#
# # gradient descent method per person
# def gradient_descent_per_person(data):
#     # print('running gradient_descent_per_person (in pysapm.py)')
#     nperson = data['nperson']
#     tsize = data['tsize']
#     tplist_full = data['tplist_full']
#     nruns_per_person = data['nruns_per_person']
#     nclusterlist = data['nclusterlist']
#     Minput = data['Minput']
#     fintrinsic_count = data['fintrinsic_count']
#     fintrinsic_region = data['fintrinsic_region']
#     vintrinsic_count = data['vintrinsic_count']
#     epoch = data['epoch']
#     timepoint = data['timepoint']
#     tcdata_centered = data['tcdata_centered']
#     ctarget = data['ctarget']
#     csource = data['csource']
#     latent_flag = data['latent_flag']
#     Mconn = data['Mconn']
#     ntime = data['ntime']
#     NP = data['NP']
#     component_data = data['component_data']
#     average_data = data['average_data']
#     epochnum = data['epochnum']
#     fintrinsic_base = data['fintrinsic_base']
#     PCloadings = data['PCloadings']
#     initial_alpha = data['initial_alpha']
#     initial_Lweight = data['initial_Lweight']
#     initial_dval = data['initial_dval']
#     alpha_limit = data['alpha_limit']
#     nitermax = data['nitermax']
#     fixed_beta_vals = data['fixed_beta_vals']
#     verbose = data['verbose']
#     beta_initial = data['beta_initial']
#
#     # if verbose: print('starting person {} at {}'.format(nperson, time.ctime()))
#     tp = tplist_full[epochnum][nperson]['tp']
#     tsize_total = len(tp)
#     nruns = nruns_per_person[nperson]
#
#     # PCparams = {'components': component_data, 'loadings': original_loadings}
#     Sinput = []
#     for rval in range(len(nclusterlist)):
#         r1 = np.sum(nclusterlist[:rval]).astype(int)
#         r2 = np.sum(nclusterlist[:(rval + 1)]).astype(int)
#         L = PCloadings[r1:r2]
#         L = np.repeat(L[:, np.newaxis], tsize_total, axis=1)
#         C = component_data[r1:r2, tp]
#         tc1 = np.sum(L * C, axis=0) + average_data[r1, tp]
#         # tc1 /= np.var(tc1)
#         Sinput.append(tc1)
#
#     Sinput = np.array(Sinput)
#     # Sinput is size:  nregions x tsize_total
#
#     # setup fixed intrinsic based on the model paradigm
#     # need to account for timepoint and epoch....
#     if fintrinsic_count > 0:
#         if epoch >= tsize:
#             et1 = 0
#             et2 = tsize
#         else:
#             if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
#                 et1 = (timepoint - np.floor(epoch / 2)).astype(int)
#                 et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#             else:
#                 et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
#                 et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#         if et1 < 0: et1 = 0
#         if et2 > tsize: et2 = tsize
#         epoch = et2 - et1
#
#         ftemp = fintrinsic_base[et1:et2]
#         fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
#         if np.var(ftemp) > 1.0e-3:
#             Sint = Sinput[fintrinsic_region, :]
#             Sint = Sint - np.mean(Sint)
#             # need to add constant to fit values
#             G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
#             b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
#             beta_int1 = b[0]
#         else:
#             beta_int1 = 0.0
#     else:
#         beta_int1 = 0.0
#         fintrinsic1 = []
#
#     lastgood_beta_int1 = copy.deepcopy(beta_int1)
#
#     # initialize beta values-----------------------------------
#     # beta_initial = np.zeros(len(csource))
#     # betascale = 0.0
#     # beta_initial = betascale * np.random.randn(len(csource))
#
#     # initalize Sconn
#     betavals = copy.deepcopy(beta_initial)  # initialize beta values at zero
#     lastgood_betavals = copy.deepcopy(betavals)
#
#     results_record = []
#     ssqd_record = []
#
#     alpha = copy.deepcopy(initial_alpha)
#     Lweight = copy.deepcopy(initial_Lweight)
#     dval = copy.deepcopy(initial_dval)
#
#     Mconn[ctarget, csource] = copy.deepcopy(betavals)
#
#     # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
#     fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
#                                                              beta_int1, fintrinsic1)
#     ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
#     ssqd_starting = copy.deepcopy(ssqd)
#     ssqd_record += [ssqd]
#
#     iter = 0
#     converging = True
#     dssq_record = np.ones(3)
#     dssq_count = 0
#     sequence_count = 0
#     while alpha > alpha_limit and iter < nitermax and converging:
#         iter += 1
#         # gradients in betavals and beta_int1
#         Mconn[ctarget, csource] = copy.deepcopy(betavals)
#         fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                  vintrinsic_count, beta_int1, fintrinsic1)
#         dssq_db, ssqd, dssq_dbeta1 = gradients_for_betavals(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
#                                                             fintrinsic_count, vintrinsic_count, beta_int1,
#                                                             fintrinsic1, Lweight)
#         ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
#         ssqd_record += [ssqd]
#
#         # fix some beta values at zero, if specified
#         if len(fixed_beta_vals) > 0:
#             dssq_db[fixed_beta_vals] = 0
#
#         # apply the changes
#         # limit the betaval changes
#         dsmax = 0.1 / alpha
#         dssq_db[dssq_db < -dsmax] = -dsmax
#         dssq_db[dssq_db > dsmax] = dsmax
#
#         betavals -= alpha * dssq_db
#         beta_int1 -= alpha * dssq_dbeta1
#         # beta_int1 = np.abs(beta_int1) # limit beta_int1
#
#         Mconn[ctarget, csource] = copy.deepcopy(betavals)
#         fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
#                                                                  vintrinsic_count, beta_int1, fintrinsic1)
#         ssqd_new = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
#
#         err_total = Sinput - fit
#         Smean = np.mean(Sinput)
#         errmean = np.mean(err_total)
#         R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)
#         if R2total < 0: R2total = 0.0
#
#         # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
#         results_record.append({'Sinput': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
#
#         if ssqd_new >= ssqd:
#             alpha *= 0.5
#             # revert back to last good values
#             betavals = copy.deepcopy(lastgood_betavals)
#             beta_int1 = copy.deepcopy(lastgood_beta_int1)
#             dssqd = ssqd - ssqd_new
#             dssq_record = np.ones(3)  # reset the count
#             dssq_count = 0
#             sequence_count = 0
#             if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq > 0  - no update'.format(iter, alpha))
#         else:
#             # save the good values
#             lastgood_betavals = copy.deepcopy(betavals)
#             lastgood_beta_int1 = copy.deepcopy(beta_int1)
#
#             dssqd = ssqd - ssqd_new
#             ssqd = copy.deepcopy(ssqd_new)
#
#             # sequence_count += 1
#             # if sequence_count > 5:
#             #     alpha *= 1.5
#             #     sequence_count = 0
#
#             dssq_count += 1
#             dssq_count = np.mod(dssq_count, 3)
#             # dssq_record[dssq_count] = 100.0 * dssqd / ssqd_starting
#             dssq_record[dssq_count] = copy.deepcopy(dssqd)
#             if np.max(dssq_record) < 1e-6:  converging = False
#
#         if verbose: print('beta vals:  iter {} alpha {:.3e}  delta ssq {:.4f}  relative: {:.1f} percent  '
#                           'R2 {:.3f}'.format(iter, alpha, -dssqd, 100.0 * ssqd / ssqd_starting, R2total))
#         # now repeat it ...
#
#     # fit the results now to determine output signaling from each region
#     Mconn[ctarget, csource] = copy.deepcopy(betavals)
#     fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
#                                                              beta_int1, fintrinsic1)
#     Sconn = Meigv @ Mintrinsic  # signalling over each connection
#
#     entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
#              'R2total': R2total, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'betavals': betavals,
#              'fintrinsic1': fintrinsic1, 'PCloadings': PCloadings, 'fintrinsic_base': fintrinsic_base}
#
#     return entry
#
# #------------------------------------------------------------------------
# # gradient descent method per person
# def gradient_descent_clustermix_per_person2(data):
#     # print('running gradient_descent_per_person (in pysapm.py)')
#     nperson = data['nperson']
#     tsize = data['tsize']
#     tplist_full = data['tplist_full']
#     nruns_per_person = data['nruns_per_person']
#     nclusterlist = data['nclusterlist']
#     Minput = data['Minput']
#     fintrinsic_count = data['fintrinsic_count']
#     fintrinsic_region = data['fintrinsic_region']
#     vintrinsic_count = data['vintrinsic_count']
#     epoch = data['epoch']
#     timepoint = data['timepoint']
#     tcdata_centered = data['tcdata_centered']
#     ctarget = data['ctarget']
#     csource = data['csource']
#     latent_flag = data['latent_flag']
#     Mconn = data['Mconn']
#     ntime = data['ntime']
#     NP = data['NP']
#     # average_data = data['average_data']
#     epochnum = data['epochnum']
#     fintrinsic_base = data['fintrinsic_base']
#     initial_alpha = data['initial_alpha']
#     initial_Lweight = data['initial_Lweight']
#     initial_dval = data['initial_dval']
#     alpha_limit = data['alpha_limit']
#     nitermax = data['nitermax']
#     fixed_beta_vals = data['fixed_beta_vals']
#     verbose = data['verbose']
#     beta_initial = data['beta_initial']
#
#     # PCloadings = data['PCloadings']
#     # component_data = data['component_data']
#     clusterweights = data['clusterweights']
#     tcdata = data['tcdata']
#
#     # if verbose: print('starting person {} at {}'.format(nperson, time.ctime()))
#     tp = tplist_full[epochnum][nperson]['tp']
#     tsize_total = len(tp)
#     nruns = nruns_per_person[nperson]
#
#     # PCparams = {'components': component_data, 'loadings': original_loadings}
#     # data are weighted sums of the original cluster data
#     Sinput = []
#     for rval in range(len(nclusterlist)):
#         r1 = np.sum(nclusterlist[:rval]).astype(int)
#         r2 = np.sum(nclusterlist[:(rval + 1)]).astype(int)
#         L = clusterweights[r1:r2]
#         L = np.repeat(L[:, np.newaxis], tsize_total, axis=1)
#         C = tcdata[r1:r2, tp]
#         tc1 = np.sum(L * C, axis=0)
#         # tc1 /= np.var(tc1)
#         Sinput.append(tc1)
#
#     Sinput = np.array(Sinput)
#     # Sinput is size:  nregions x tsize_total
#
#     # setup fixed intrinsic based on the model paradigm
#     # need to account for timepoint and epoch....
#     if fintrinsic_count > 0:
#         if epoch >= tsize:
#             et1 = 0
#             et2 = tsize
#         else:
#             if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
#                 et1 = (timepoint - np.floor(epoch / 2)).astype(int)
#                 et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#             else:
#                 et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
#                 et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#         if et1 < 0: et1 = 0
#         if et2 > tsize: et2 = tsize
#         epoch = et2 - et1
#
#         ftemp = fintrinsic_base[et1:et2]
#         fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
#         if np.var(ftemp) > 1.0e-3:
#             Sint = Sinput[fintrinsic_region, :]
#             Sint = Sint - np.mean(Sint)
#             # need to add constant to fit values
#             G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
#             b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
#             beta_int1 = b[0]
#         else:
#             beta_int1 = 0.0
#     else:
#         beta_int1 = 0.0
#         fintrinsic1 = []
#
#     lastgood_beta_int1 = copy.deepcopy(beta_int1)
#
#     # initialize beta values-----------------------------------
#     # beta_initial = np.zeros(len(csource))
#     # betascale = 0.0
#     # beta_initial = betascale * np.random.randn(len(csource))
#
#     # initalize Sconn
#     betavals = copy.deepcopy(beta_initial)  # initialize beta values at zero
#     lastgood_betavals = copy.deepcopy(betavals)
#
#     results_record = []
#     ssqd_record = []
#
#     alpha = copy.deepcopy(initial_alpha)
#     alphalist = initial_alpha*np.ones(len(betavals))
#     alphabint = copy.deepcopy(initial_alpha)
#     alphamax = copy.deepcopy(initial_alpha)
#     Lweight = copy.deepcopy(initial_Lweight)
#     dval = copy.deepcopy(initial_dval)
#
#     Mconn[ctarget, csource] = copy.deepcopy(betavals)
#
#     # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
#     fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
#                                                              beta_int1, fintrinsic1)
#     ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
#     ssqd_starting = copy.deepcopy(ssqd)
#     ssqd_old = copy.deepcopy(ssqd)
#     ssqd_record += [ssqd]
#
#     iter = 0
#     converging = True
#     dssq_record = np.ones(3)
#     dssq_count = 0
#     sequence_count = 0
#
#     while alphamax > alpha_limit and iter < nitermax and converging:
#         iter += 1
#
#         betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint = \
#             update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
#                                                 fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
#                                                 alphalist, alphabint)
#
#         ssqd_record += [ssqd]
#
#         err_total = Sinput - fit
#         Smean = np.mean(Sinput)
#         errmean = np.mean(err_total)
#         # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)
#
#         # R2list = [1-np.sum((Sinput[x,:]-fit[x,:])**2)/np.sum(Sinput[x,:]**2) for x in range(nregions)]
#         R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
#         # R2total = np.mean(R2list)
#         R2avg = np.mean(R2list)
#         R2total = 1.0 - np.sum((Sinput - fit) ** 2) / np.sum(Sinput ** 2)
#
#         # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
#         results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
#         atemp = np.append(alphalist,alphabint)
#         alphamax = np.max(atemp)
#         alphalist[alphalist < alpha_limit] = alpha_limit
#         if alphabint < alpha_limit:  alphabint = copy.deepcopy(alpha_limit)
#
#         ssqchange = ssqd - ssqd_old
#         if ssqchange < 1e-4: converging = False
#
#         # print('{} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  R2 {:.3f}'.format(nperson,
#         #                 iter, np.mean(alphalist), ssqd, ssqchange, R2total))
#         ssqd_old = copy.deepcopy(ssqd)
#         # now repeat it ...
#
#     # fit the results now to determine output signaling from each region
#     Mconn[ctarget, csource] = copy.deepcopy(betavals)
#     fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
#                                                              beta_int1, fintrinsic1)
#     Sconn = Meigv @ Mintrinsic  # signalling over each connection
#
#     entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
#              'R2total': R2total, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'betavals': betavals,
#              'fintrinsic1': fintrinsic1, 'clusterweights': clusterweights, 'fintrinsic_base': fintrinsic_base}
#
#     return entry

#
# #------------------------------------------------------------------------
# # gradient descent method per person
# def gradient_descent_per_person2(data):
#     # print('running gradient_descent_per_person (in pysapm.py)')
#     nperson = data['nperson']
#     tsize = data['tsize']
#     tplist_full = data['tplist_full']
#     nruns_per_person = data['nruns_per_person']
#     nclusterlist = data['nclusterlist']
#     Minput = data['Minput']
#     fintrinsic_count = data['fintrinsic_count']
#     fintrinsic_region = data['fintrinsic_region']
#     vintrinsic_count = data['vintrinsic_count']
#     epoch = data['epoch']
#     timepoint = data['timepoint']
#     tcdata_centered = data['tcdata_centered']
#     ctarget = data['ctarget']
#     csource = data['csource']
#     latent_flag = data['latent_flag']
#     Mconn = data['Mconn']
#     ntime = data['ntime']
#     NP = data['NP']
#     component_data = data['component_data']
#     average_data = data['average_data']
#     epochnum = data['epochnum']
#     fintrinsic_base = data['fintrinsic_base']
#     PCloadings = data['PCloadings']
#     initial_alpha = data['initial_alpha']
#     initial_Lweight = data['initial_Lweight']
#     initial_dval = data['initial_dval']
#     alpha_limit = data['alpha_limit']
#     nitermax = data['nitermax']
#     fixed_beta_vals = data['fixed_beta_vals']
#     verbose = data['verbose']
#     beta_initial = data['beta_initial']
#
#     # if verbose: print('starting person {} at {}'.format(nperson, time.ctime()))
#     tp = tplist_full[epochnum][nperson]['tp']
#     tsize_total = len(tp)
#     nruns = nruns_per_person[nperson]
#
#     # PCparams = {'components': component_data, 'loadings': original_loadings}
#     Sinput = []
#     for rval in range(len(nclusterlist)):
#         r1 = np.sum(nclusterlist[:rval]).astype(int)
#         r2 = np.sum(nclusterlist[:(rval + 1)]).astype(int)
#         L = PCloadings[r1:r2]
#         L = np.repeat(L[:, np.newaxis], tsize_total, axis=1)
#         C = component_data[r1:r2, tp]
#         tc1 = np.sum(L * C, axis=0) + average_data[r1, tp]
#         # tc1 /= np.var(tc1)
#         Sinput.append(tc1)
#
#     Sinput = np.array(Sinput)
#     # Sinput is size:  nregions x tsize_total
#
#     # setup fixed intrinsic based on the model paradigm
#     # need to account for timepoint and epoch....
#     if fintrinsic_count > 0:
#         if epoch >= tsize:
#             et1 = 0
#             et2 = tsize
#         else:
#             if np.floor(epoch / 2).astype(int) == np.ceil(epoch / 2).astype(int):  # even numbered epoch
#                 et1 = (timepoint - np.floor(epoch / 2)).astype(int)
#                 et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#             else:
#                 et1 = (timepoint - np.floor(epoch / 2)).astype(int) - 1
#                 et2 = (timepoint + np.floor(epoch / 2)).astype(int)
#         if et1 < 0: et1 = 0
#         if et2 > tsize: et2 = tsize
#         epoch = et2 - et1
#
#         ftemp = fintrinsic_base[et1:et2]
#         fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])
#         if np.var(ftemp) > 1.0e-3:
#             Sint = Sinput[fintrinsic_region, :]
#             Sint = Sint - np.mean(Sint)
#             # need to add constant to fit values
#             G = np.concatenate((fintrinsic1[np.newaxis, :], np.ones((1, tsize_total))), axis=0)
#             b, fit, R2, total_var, res_var = pysem.general_glm(Sint, G)
#             beta_int1 = b[0]
#         else:
#             beta_int1 = 0.0
#     else:
#         beta_int1 = 0.0
#         fintrinsic1 = []
#
#     lastgood_beta_int1 = copy.deepcopy(beta_int1)
#
#     # initialize beta values-----------------------------------
#     # beta_initial = np.zeros(len(csource))
#     # betascale = 0.0
#     # beta_initial = betascale * np.random.randn(len(csource))
#
#     # initalize Sconn
#     betavals = copy.deepcopy(beta_initial)  # initialize beta values at zero
#     lastgood_betavals = copy.deepcopy(betavals)
#
#     results_record = []
#     ssqd_record = []
#
#     alpha = copy.deepcopy(initial_alpha)
#     alphalist = initial_alpha*np.ones(len(betavals))
#     alphabint = copy.deepcopy(initial_alpha)
#     alphamax = copy.deepcopy(initial_alpha)
#     Lweight = copy.deepcopy(initial_Lweight)
#     dval = copy.deepcopy(initial_dval)
#
#     Mconn[ctarget, csource] = copy.deepcopy(betavals)
#
#     # # starting point for optimizing intrinsics with given betavals----------------------------------------------------
#     fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
#                                                              beta_int1, fintrinsic1)
#     ssqd = sapm_error_function(Sinput, fit, Lweight, betavals, beta_int1, Mintrinsic)
#     ssqd_starting = copy.deepcopy(ssqd)
#     ssqd_old = copy.deepcopy(ssqd)
#     ssqd_record += [ssqd]
#
#     iter = 0
#     converging = True
#     dssq_record = np.ones(3)
#     dssq_count = 0
#     sequence_count = 0
#
#     while alphamax > alpha_limit and iter < nitermax and converging:
#         iter += 1
#
#         betavals, beta_int1, fit, updatebflag, updatebintflag, dssq_db, dssq_dbeta1, ssqd, alphalist, alphabint = \
#             update_betavals_sequentially(Sinput, Minput, Mconn, betavals, ctarget, csource, dval,
#                                                 fintrinsic_count, vintrinsic_count, beta_int1, fintrinsic1, Lweight,
#                                                 alphalist, alphabint)
#
#         ssqd_record += [ssqd]
#
#         err_total = Sinput - fit
#         Smean = np.mean(Sinput)
#         errmean = np.mean(err_total)
#         # R2total = 1 - np.sum((err_total - errmean) ** 2) / np.sum((Sinput - Smean) ** 2)
#
#         # R2list = [1-np.sum((Sinput[x,:]-fit[x,:])**2)/np.sum(Sinput[x,:]**2) for x in range(nregions)]
#         R2list = 1.0 - np.sum((Sinput - fit) ** 2, axis=1) / np.sum(Sinput ** 2, axis=1)
#         R2total = np.mean(R2list)
#
#         # Sinput_sim, Soutput_sim = network_sim(Sinput_full, Soutput_full, Minput, Moutput)
#         results_record.append({'Sinput': Sinput, 'fit': fit, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv})
#         atemp = np.append(alphalist,alphabint)
#         alphamax = np.max(atemp)
#         alphalist[alphalist < alpha_limit] = alpha_limit
#         if alphabint < alpha_limit:  alphabint = copy.deepcopy(alpha_limit)
#
#         ssqchange = ssqd - ssqd_old
#         if ssqchange < 1e-4: converging = False
#
#         # print('{} beta vals:  iter {} alpha {:.3e}  ssqd {:.2f} change {:.3f}  R2 {:.3f}'.format(nperson,
#         #                 iter, np.mean(alphalist), ssqd, ssqchange, R2total))
#         ssqd_old = copy.deepcopy(ssqd)
#         # now repeat it ...
#
#     # fit the results now to determine output signaling from each region
#     Mconn[ctarget, csource] = copy.deepcopy(betavals)
#     fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count, vintrinsic_count,
#                                                              beta_int1, fintrinsic1)
#     Sconn = Meigv @ Mintrinsic  # signalling over each connection
#
#     entry = {'Sinput': Sinput, 'Sconn': Sconn, 'beta_int1': beta_int1, 'Mconn': Mconn, 'Minput': Minput,
#              'R2total': R2total, 'Mintrinsic': Mintrinsic, 'Meigv': Meigv, 'betavals': betavals,
#              'fintrinsic1': fintrinsic1, 'PCloadings': PCloadings, 'fintrinsic_base': fintrinsic_base}
#
#     return entry

#
# #----------------------------------------------------------------------------------
# # --------------------------------------------------------------------
# def sem_physio_model_weightedclusters(tcdata, clusterweights, fintrinsic_base, SAPMresultsname,
#                                  SAPMparametersname, nitermax = 250, alpha_limit = 1e-5,
#                                  subsample = [1,0], fixed_beta_vals = [], betascale = 0.01, verbose = False,
#                                  nprocessors = 8):
#     starttime = time.ctime()
#
#     # instead of working with specific clusters, this version uses a mix of clusters
#     # as a continuum, in order to find the optimal clusters
#     # principal components information about clusters are contained in:
#     # PCparams = {'components': component_data, 'loadings': original_loadings}
#     # how the components are mixed for each region are contained in PCloadings
#
#     # initialize gradient-descent parameters--------------------------------------------------------------
#     initial_alpha = 1e-2
#     initial_Lweight = 1e-12
#     initial_dval = 0.01
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
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     fintrinsic_region = SAPMparams['fintrinsic_region']
#     Mconn = SAPMparams['Mconn']
#     Minput = SAPMparams['Minput']
#     timepoint = SAPMparams['timepoint']
#     epoch = SAPMparams['epoch']
#     latent_flag = SAPMparams['latent_flag']
#
#     if isinstance(betascale, str):
#         # read saved beta_initial values
#         b = np.load(betascale, allow_pickle=True).flat[0]
#         beta_initial = b['beta_initial']
#     else:
#         beta_initial = betascale * np.random.randn(len(csource))
#
#     tplist_full = SAPMparams['tplist_full']
#     ntime, NP = np.shape(tplist_full)
#     #---------------------------------------------------------------------------------------------------------
#     #---------------------------------------------------------------------------------------------------------
#     # repeat the process for each participant-----------------------------------------------------------------
#     betalimit = 3.0
#     epochnum = 0
#     SAPMresults = []
#     beta_init_record = []
#
#     # data for gradient_descent_per_person
#     ntime, NP = np.shape(SAPMparams['tplist_full'])
#     fixed_beta_vals = []
#     verbose = False
#
#     data = {'nperson':0,
#             'tsize':SAPMparams['tsize'],
#             'tplist_full':SAPMparams['tplist_full'],
#             'nruns_per_person':SAPMparams['nruns_per_person'],
#             'nclusterlist':SAPMparams['nclusterlist'],
#             'Minput':SAPMparams['Minput'],
#             'fintrinsic_count':SAPMparams['fintrinsic_count'],
#             'fintrinsic_region':SAPMparams['fintrinsic_region'],
#             'vintrinsic_count':SAPMparams['vintrinsic_count'],
#             'epoch':SAPMparams['epoch'],
#             'timepoint':SAPMparams['timepoint'],
#             'tcdata_centered':SAPMparams['tcdata_centered'],
#             'ctarget':SAPMparams['ctarget'],
#             'csource':SAPMparams['csource'],
#             'latent_flag':SAPMparams['latent_flag'],
#             'Mconn':SAPMparams['Mconn'],
#             'ntime':ntime,
#             'NP':NP,
#             'epochnum' :epochnum,
#             'fintrinsic_base' :fintrinsic_base,
#             'initial_alpha' :initial_alpha,
#             'initial_Lweight' :initial_Lweight,
#             'initial_dval' :initial_dval,
#             'alpha_limit' :alpha_limit,
#             'nitermax' :nitermax,
#             'fixed_beta_vals' :fixed_beta_vals,
#             'verbose' :verbose,
#             'beta_initial':beta_initial,
#             'tcdata':tcdata,
#             'clusterweights':clusterweights}
#
#     # 'PCloadings': PCloadings,
#     # 'component_data': PCparams['components'],
#
#     # setup iterable input parameters
#     input_data = []
#     for nperson in range(subsample[1], NP, subsample[0]):
#         oneval = copy.deepcopy(data)
#         oneval['nperson'] = nperson
#         input_data.append(oneval)
#     p,f = os.path.split(SAPMparametersname)
#     search_data_name = os.path.join(p,'cluster_search_data.npy')
#
#     startpool = time.time()
#     if nprocessors <= 1:
#         # SAPMresults = [gradient_descent_per_person2(input_data[n]) for n in range(len(input_data))]
#         SAPMresults = [gradient_descent_clustermix_per_person2(input_data[n]) for n in range(len(input_data))]
#
#     else:
#         pool = mp.Pool(nprocessors)
#         # print('runnning gradient_descent_per_person ... (with {} processors)'.format(nprocessors))
#         # SAPMresults = pool.map(gradient_descent_per_person2, input_data)
#         SAPMresults = pool.map(gradient_descent_clustermix_per_person2, input_data)
#         pool.close()
#     donepool = time.time()
#     # print('time to run gradient-descent with {} processors:  {:.1f} sec'.format(nprocessors, donepool-startpool))
#
#     stoptime = time.ctime()
#
#     if verbose:
#         print('finished SAPM at {}'.format(time.ctime()))
#         print('     started at {}'.format(starttime))
#
#     return SAPMresults, search_data_name

#
# # ----------------------------------------------------------------------------------
# # --------------------------------------------------------------------
# def sem_physio_model_PCAclusters(PCparams, PCloadings, fintrinsic_base, SAPMresultsname,
#                                  SAPMparametersname, nitermax=250, alpha_limit=1e-5,
#                                  subsample=[1, 0], fixed_beta_vals=[], betascale=0.01, verbose=False,
#                                  nprocessors=8):
#     starttime = time.ctime()
#
#     # instead of working with specific clusters, this version uses a mix of clusters
#     # as a continuum, in order to find the optimal clusters
#     # principal components information about clusters are contained in:
#     # PCparams = {'components': component_data, 'loadings': original_loadings}
#     # how the components are mixed for each region are contained in PCloadings
#
#     # initialize gradient-descent parameters--------------------------------------------------------------
#     initial_alpha = 1e-2
#     initial_Lweight = 1e-12
#     initial_dval = 0.01
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
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     fintrinsic_region = SAPMparams['fintrinsic_region']
#     Mconn = SAPMparams['Mconn']
#     Minput = SAPMparams['Minput']
#     timepoint = SAPMparams['timepoint']
#     epoch = SAPMparams['epoch']
#     latent_flag = SAPMparams['latent_flag']
#
#     if isinstance(betascale, str):
#         # read saved beta_initial values
#         b = np.load(betascale, allow_pickle=True).flat[0]
#         beta_initial = b['beta_initial']
#     else:
#         beta_initial = betascale * np.random.randn(len(csource))
#
#     tplist_full = SAPMparams['tplist_full']
#     ntime, NP = np.shape(tplist_full)
#     # ---------------------------------------------------------------------------------------------------------
#     # ---------------------------------------------------------------------------------------------------------
#     # repeat the process for each participant-----------------------------------------------------------------
#     betalimit = 3.0
#     epochnum = 0
#     SAPMresults = []
#     beta_init_record = []
#
#     # data for gradient_descent_per_person
#     ntime, NP = np.shape(SAPMparams['tplist_full'])
#     fixed_beta_vals = []
#     verbose = False
#
#     data = {'nperson': 0,
#             'tsize': SAPMparams['tsize'],
#             'tplist_full': SAPMparams['tplist_full'],
#             'nruns_per_person': SAPMparams['nruns_per_person'],
#             'nclusterlist': SAPMparams['nclusterlist'],
#             'Minput': SAPMparams['Minput'],
#             'fintrinsic_count': SAPMparams['fintrinsic_count'],
#             'fintrinsic_region': SAPMparams['fintrinsic_region'],
#             'vintrinsic_count': SAPMparams['vintrinsic_count'],
#             'epoch': SAPMparams['epoch'],
#             'timepoint': SAPMparams['timepoint'],
#             'tcdata_centered': SAPMparams['tcdata_centered'],
#             'ctarget': SAPMparams['ctarget'],
#             'csource': SAPMparams['csource'],
#             'latent_flag': SAPMparams['latent_flag'],
#             'Mconn': SAPMparams['Mconn'],
#             'ntime': ntime,
#             'NP': NP,
#             'average_data': PCparams['average'],
#             'epochnum': epochnum,
#             'fintrinsic_base': fintrinsic_base,
#             'initial_alpha': initial_alpha,
#             'initial_Lweight': initial_Lweight,
#             'initial_dval': initial_dval,
#             'alpha_limit': alpha_limit,
#             'nitermax': nitermax,
#             'fixed_beta_vals': fixed_beta_vals,
#             'verbose': verbose,
#             'beta_initial': beta_initial,
#             'PCloadings': PCloadings,
#             'component_data': PCparams['components']}
#
#     # setup iterable input parameters
#     input_data = []
#     for nperson in range(subsample[1], NP, subsample[0]):
#         oneval = copy.deepcopy(data)
#         oneval['nperson'] = nperson
#         input_data.append(oneval)
#     p, f = os.path.split(SAPMparametersname)
#     search_data_name = os.path.join(p, 'cluster_search_data.npy')
#
#     startpool = time.time()
#     if nprocessors <= 1:
#         SAPMresults = [gradient_descent_per_person2(input_data[n]) for n in range(len(input_data))]
#     else:
#         pool = mp.Pool(nprocessors)
#         # print('runnning gradient_descent_per_person ... (with {} processors)'.format(nprocessors))
#         SAPMresults = pool.map(gradient_descent_per_person2, input_data)
#         pool.close()
#     donepool = time.time()
#     # print('time to run gradient-descent with {} processors:  {:.1f} sec'.format(nprocessors, donepool-startpool))
#
#     stoptime = time.ctime()
#
#     if verbose:
#         print('finished SAPM at {}'.format(time.ctime()))
#         print('     started at {}'.format(starttime))
#
#     return SAPMresults, search_data_name


# ----------------------------------------------------------------------------------
# --------------------------------------------------------------------
def sem_physio_model2_fast(tcdata, clusterlist, fintrinsic_base, SAPMresultsname,
                                 SAPMparametersname, nitermax = 250, alpha_limit = 1e-5,
                                 subsample = [1,0], fixed_beta_vals = [], betascale = 0.01, verbose = False,
                                 nprocessors = 1):
    starttime = time.ctime()

    # instead of working with specific clusters, this version uses a mix of clusters
    # as a continuum, in order to find the optimal clusters
    # principal components information about clusters are contained in:
    # PCparams = {'components': component_data, 'loadings': original_loadings}
    # how the components are mixed for each region are contained in PCloadings

    # initialize gradient-descent parameters--------------------------------------------------------------
    initial_alpha = 1e-2
    initial_Lweight = 0.0
    initial_dval = 0.01

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
    ctarget = SAPMparams['ctarget']
    csource = SAPMparams['csource']
    fintrinsic_region = SAPMparams['fintrinsic_region']
    Mconn = SAPMparams['Mconn']
    Minput = SAPMparams['Minput']
    timepoint = SAPMparams['timepoint']
    epoch = SAPMparams['epoch']
    latent_flag = SAPMparams['latent_flag']

    if isinstance(betascale, str):
        # read saved beta_initial values
        b = np.load(betascale, allow_pickle=True).flat[0]
        beta_initial = b['beta_initial']
    else:
        beta_initial = betascale * np.random.randn(len(csource))

    tplist_full = SAPMparams['tplist_full']
    ntime, NP = np.shape(tplist_full)
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # repeat the process for each participant-----------------------------------------------------------------
    betalimit = 3.0
    epochnum = 0
    SAPMresults = []
    beta_init_record = []

    # data for gradient_descent_per_person
    ntime, NP = np.shape(SAPMparams['tplist_full'])
    fixed_beta_vals = []
    verbose = False

    # specifiy the clusters to use, by using the "cluster weights" as in other methods
    # clusterweights = np.zeros(nclusterstotal)
    # for region in range(nregions):
    #     r1 = sum(nclusterlist[:region])
    #     r2 = sum(nclusterlist[:(region + 1)])
    #     temp_weights = np.zeros(nclusterlist[region])
    #     temp_weights[clusters[region]] = 1.0
    #     clusterweights[r1:r2] = temp_weights

    data = {'nperson':0,
            'tsize':SAPMparams['tsize'],
            'tplist_full':SAPMparams['tplist_full'],
            'nruns_per_person':SAPMparams['nruns_per_person'],
            'nclusterlist':SAPMparams['nclusterlist'],
            'Minput':SAPMparams['Minput'],
            'fintrinsic_count':SAPMparams['fintrinsic_count'],
            'fintrinsic_region':SAPMparams['fintrinsic_region'],
            'vintrinsic_count':SAPMparams['vintrinsic_count'],
            'epoch':SAPMparams['epoch'],
            'timepoint':SAPMparams['timepoint'],
            'tcdata_centered':SAPMparams['tcdata_centered'],
            'ctarget':SAPMparams['ctarget'],
            'csource':SAPMparams['csource'],
            'latent_flag':SAPMparams['latent_flag'],
            'Mconn':SAPMparams['Mconn'],
            'ntime':ntime,
            'NP':NP,
            'epochnum' :epochnum,
            'fintrinsic_base' :fintrinsic_base,
            'initial_alpha' :initial_alpha,
            'initial_Lweight' :initial_Lweight,
            'initial_dval' :initial_dval,
            'alpha_limit' :alpha_limit,
            'nitermax' :nitermax,
            'fixed_beta_vals' :fixed_beta_vals,
            'verbose' :verbose,
            'beta_initial':beta_initial,
            'clusterlist':clusterlist}

    # 'PCloadings': PCloadings,
    # 'component_data': PCparams['components'],

    # setup iterable input parameters
    input_data = []
    for nperson in range(subsample[1], NP, subsample[0]):
        oneval = copy.deepcopy(data)
        oneval['nperson'] = nperson
        input_data.append(oneval)
    p,f = os.path.split(SAPMparametersname)
    # search_data_name = os.path.join(p,'cluster_search_data.npy')

    startpool = time.time()
    if nprocessors <= 1:
        SAPMresults = [gradient_descent_per_person_original(input_data[n]) for n in range(len(input_data))]
        # SAPMresults = [gradient_descent_clustermix_per_person2(input_data[n]) for n in range(len(input_data))]

    else:
        pool = mp.Pool(nprocessors)
        # print('runnning gradient_descent_per_person ... (with {} processors)'.format(nprocessors))
        # SAPMresults = pool.map(gradient_descent_clustermix_per_person2, input_data)
        SAPMresults = pool.map(gradient_descent_per_person_original, input_data)
        pool.close()
    donepool = time.time()
    # print('time to run gradient-descent with {} processors:  {:.1f} sec'.format(nprocessors, donepool-startpool))

    stoptime = time.ctime()

    if verbose:
        print('finished SAPM at {}'.format(time.ctime()))
        print('     started at {}'.format(starttime))

    return SAPMresults


def mod_tplist_for_bootstrap(tplist_full, epoch, modtype, percent_replace = 0, tsize =40):
    # modtype can be 'random', 'allodds', 'allevens', 'firsthalf', 'lasthalf'
    tplist = copy.deepcopy(tplist_full[epoch])
    NP = len(tplist)
    stilllooking = True
    if modtype == 'allodds':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            for tt in range(0,nt,2): tpb[tt] = tpb[tt+1]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'allevens':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            for tt in range(1,nt,2): tpb[tt] = tpb[tt-1]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'firsthalf':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            tt = np.floor(nt/2).astype(int)
            tpb[-tt:] = tpb[:tt]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'lasthalf':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            tpb = copy.deepcopy(tp)
            tt = np.floor(nt/2).astype(int)
            tpb[:tt] = tpb[-tt:]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'oddruns':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            nruns = np.floor(nt/tsize).astype(int)
            tpb = copy.deepcopy(tp)
            replaceruns = list(range(0,nruns,2))
            for rr in replaceruns:
                if nruns > (rr+1):
                    tr1 = rr*tsize
                    tr2 = (rr+1)*tsize
                    tt1 = (rr+1)*tsize
                    tt2 = (rr+2)*tsize
                    tpb[tr1:tr2] = tpb[tt1:tt2]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'evenruns':
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            nruns = np.floor(nt/tsize).astype(int)
            tpb = copy.deepcopy(tp)
            replaceruns = list(range(1,nruns,2))
            for rr in replaceruns:
                tr1 = rr*tsize
                tr2 = (rr+1)*tsize
                tt1 = (rr-1)*tsize
                tt2 = rr*tsize
                tpb[tr1:tr2] = tpb[tt1:tt2]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    if modtype == 'random' or stilllooking:
        for nn in range(NP):
            tp = copy.deepcopy(tplist[nn]['tp'])
            nt = len(tp)
            if percent_replace <= 0:
                nreplace = 1
            else:
                nreplace = np.floor(percent_replace*nt/100.0).astype(int)
            tpb = copy.deepcopy(tp)
            ntlist = list(range(nt))
            treplace = random.sample(ntlist,nreplace)
            ntlist2 = [x for x in ntlist if x not in treplace]
            twith = random.sample(ntlist2,nreplace)
            for tt in range(nreplace): tpb[treplace[tt]] = tpb[twith[tt]]
            tplist[nn]['tp'] = tpb
        stilllooking = False

    tplist_full2 = copy.deepcopy(tplist_full)
    tplist_full2[epoch] = tplist
    return tplist_full2

#
# def loadings_gradients(beta, betascale, PCparams,PCloadings,paradigm_centered,SAPMresultsname,SAPMparametersname,subsample, nprocessors, Lweight = 1.0e-8):
#     SAPMresults, search_data_name = sem_physio_model_PCAclusters(PCparams, PCloadings, paradigm_centered, SAPMresultsname,
#                                               SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
#                                               subsample = subsample, fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#     nclusters_total = len(PCloadings)
#
#     # cost function
#     R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#     basecost = np.sum(1 - R2list) + Lweight * np.sum(np.abs(PCloadings))
#
#     # the weightings should indicate how much each cluster should contribute to a weighted average
#     # use testload as a weighted average of the original cluster PCloadings
#
#     # gradients in PCloadings
#     load_gradients = np.zeros(nclusters_total)
#     gradcalcstart = time.time()
#     for aa in range(nclusters_total):
#         testload = copy.deepcopy(PCloadings)
#         testload[aa] += beta
#         SAPMresults, search_data_name = sem_physio_model_PCAclusters(PCparams, testload, paradigm_centered, SAPMresultsname,
#                                                   SAPMparametersname, nitermax = 100, alpha_limit = 1e-5, subsample = subsample,
#                                                   fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#
#         # cost function
#         R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#         testcost = np.sum(1 - R2list) + Lweight * np.sum(np.abs(testload))
#
#         load_gradients[aa] = (testcost - basecost) / beta
#     gradcalcend = time.time()
#     print('calculating load gradients took {:.1f} seconds'.format(gradcalcend - gradcalcstart))
#
#     return load_gradients, basecost
#
#
#
# def weight_gradients(beta, betascale, tcdata,clusterweights,paradigm_centered,SAPMresultsname,SAPMparametersname,subsample, nprocessors, Lweight = 1.0e-8):
#     SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, clusterweights, paradigm_centered, SAPMresultsname,
#                                               SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
#                                               subsample = subsample, fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#     nclusters_total = len(clusterweights)
#
#     # cost function
#     R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#     basecost = np.sum(1 - R2list)
#
#     # the weightings should indicate how much each cluster should contribute to a weighted average
#     # use testload as a weighted average of the original cluster PCloadings
#
#     # gradients in PCloadings
#     wgradients = np.zeros(nclusters_total)
#     gradcalcstart = time.time()
#     for aa in range(nclusters_total):
#         testweight = copy.deepcopy(clusterweights)
#         testweight[aa] += beta
#         testweight /= np.sum(testweight)
#         SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, testweight, paradigm_centered, SAPMresultsname,
#                                                   SAPMparametersname, nitermax = 100, alpha_limit = 1e-5, subsample = subsample,
#                                                   fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#
#         # cost function
#         R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#         testcost = np.sum(1 - R2list)
#
#         wgradients[aa] = (testcost - basecost) / beta
#     gradcalcend = time.time()
#     print('calculating weight gradients took {:.1f} seconds'.format(gradcalcend - gradcalcstart))
#
#     return wgradients, basecost
#
#
#
# def single_weight_gradient(weightindex, beta, betascale, tcdata,clusterweights, nclusterlist,paradigm_centered,SAPMresultsname,SAPMparametersname,subsample, nprocessors, Lweight = 1.0e-8):
#     SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, clusterweights, paradigm_centered, SAPMresultsname,
#                                               SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
#                                               subsample = subsample, fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#     nclusters_total = len(clusterweights)
#
#     # cost function
#     R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#     basecost = np.sum(1 - R2list)
#
#     # the weightings should indicate how much each cluster should contribute to a weighted average
#     # use testload as a weighted average of the original cluster PCloadings
#
#     # gradients in PCloadings
#     testweight = copy.deepcopy(clusterweights)
#     testweight[weightindex] += beta
#     nregions = len(nclusterlist)
#     for aa in range(nregions):
#         r1 = sum(nclusterlist[:aa])
#         r2 = sum(nclusterlist[:(aa + 1)])
#         weightset = copy.deepcopy(testweight[r1:r2])
#         testweight[r1:r2] = weightset/np.sum(weightset)
#     SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, testweight, paradigm_centered, SAPMresultsname,
#                                               SAPMparametersname, nitermax = 100, alpha_limit = 1e-5, subsample = subsample,
#                                               fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#
#     # cost function
#     R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#     testcost = np.sum(1 - R2list)
#
#     wgradient = (testcost - basecost) / beta
#     return wgradient, basecost


# gradient descent method to find best clusters------------------------------------
# def SAPM_cluster_search2b(outputdir, SAPMresultsname, SAPMparametersname, networkfile, DBname, regiondataname, clusterdataname, nprocessors,
#                         samplesplit, samplestart=0, initial_clusters = [], timepoint = 'all', epoch = 'all', betascale = 0.0):
#
#     if not os.path.exists(outputdir): os.mkdir(outputdir)
#
#     # load paradigm data--------------------------------------------------------------------
#     xls = pd.ExcelFile(DBname, engine='openpyxl')
#     df1 = pd.read_excel(xls, 'paradigm1_BOLD')
#     del df1['Unnamed: 0']  # get rid of the unwanted header column
#     fields = list(df1.keys())
#     paradigm = df1['paradigms_BOLD']
#     timevals = df1['time']
#     paradigm_centered = paradigm - np.mean(paradigm)
#     dparadigm = np.zeros(len(paradigm))
#     dparadigm[1:] = np.diff(paradigm_centered)
#
#     # get cluster info and setup for saving information later
#     cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
#     # cluster_properties = cluster_data['cluster_properties']
#     cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)
#     nregions = len(cluster_properties)
#     nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
#     rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
#     namelist_addon = ['R '+n for n in rnamelist]
#     namelist = rnamelist + namelist_addon
#
#     # ---------------------
#     # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
#     prep_data_sem_physio_model_SO(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
#                                   fullgroup=False, normalizevar=False, filter_tcdata = False)
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
#
#     # for one set of PCloadings
#     Lweight = 1.0e-20
#     beta = 0.01
#     alpha = 1.0
#     initial_alpha = copy.deepcopy(alpha)
#     alphalimit = 1e-4
#     maxiter = 100
#     subsample = [samplesplit,samplestart]  # [2,0] use every 2nd data set, starting with samplestart
#
#     nclusters_total = np.sum(nclusterlist)
#     clusterweights = np.random.randn(nclusters_total)  # initially set the clusterweights to random values
#     clusterweights = np.ones(nclusters_total)  # initially set the clusterweights to equal values
#     for aa in range(nregions):
#         r1 = sum(nclusterlist[:aa])
#         r2 = sum(nclusterlist[:(aa + 1)])
#         weightset = clusterweights[r1:r2]
#         clusterweights[r1:r2] = weightset/np.sum(weightset)
#
#     if len(initial_clusters) == nregions:
#         clusterweights = np.zeros(nclusters_total)
#         for aa in range(nregions):
#             cluster = initial_clusters[aa]
#             r1 = sum(nclusterlist[:aa])
#             r2 = sum(nclusterlist[:(aa + 1)])
#             weightset = clusterweights[r1:r2]
#             weightset[cluster] = 1.0
#             clusterweights[r1:r2] = weightset
#
#     lastgood_clusterweights = copy.deepcopy(clusterweights)
#
#     # gradient descent to find best cluster combination
#     iter = 0
#     costrecord = []
#     print('starting gradient descent search of weighted clusters at {}'.format(time.ctime()))
#     recalculate_wgradients = True
#     runcount = 0
#     runrecord = []
#     while (alpha > alphalimit) and (iter < maxiter):
#         # subsample[1] = iter % 2   # vary which data sets are used out of the subsample
#         iter += 1
#         # gradients in PCloadings
#         # update one weight at a time
#
#         if recalculate_wgradients:
#             clusterweights_start = copy.deepcopy(clusterweights)
#
#             SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, clusterweights, paradigm_centered,
#                                 SAPMresultsname, SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
#                                 subsample = subsample, fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#             R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#             basecost = np.sum(1-R2list)
#
#             wgradients, basecost2 = weight_gradients(beta, betascale, tcdata, clusterweights, paradigm_centered, SAPMresultsname,
#                              SAPMparametersname, subsample, nprocessors, Lweight)
#             # print('basecost {:.4f}  basecost2 {:.4f}'.format(basecost,basecost2))
#             if iter == 1: lastcost = copy.deepcopy(basecost)
#         else:
#             print('not calculating load gradients')
#
#         clusterweights -= alpha*wgradients
#         for aa in range(nregions):
#             r1 = sum(nclusterlist[:aa])
#             r2 = sum(nclusterlist[:(aa + 1)])
#             weightset = copy.deepcopy(clusterweights[r1:r2])
#             clusterweights[r1:r2] = weightset / np.sum(weightset)
#
#         SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, clusterweights, paradigm_centered,
#                             SAPMresultsname, SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
#                             subsample = subsample, fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#         R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#         newcost = np.sum(1-R2list)
#
#         costrecord += [basecost]
#         entry = {'clusterweights_start':clusterweights_start, 'clusterweights':clusterweights, 'newcost':newcost, 'lastcost':lastcost, 'basecost':basecost, 'basecost2':basecost2}
#         runrecord.append(entry)
#
#         if newcost < basecost:
#             lastgood_clusterweights = copy.deepcopy(clusterweights)
#             recalculate_wgradients = True
#             runcount += 1
#             if runcount > 2:
#                 alpha= np.min([10.0*initial_alpha, 1.3*alpha])
#             print('iter {}  new cost = {:.5e}  base cost = {:.5e}  base cost2 = {:.5e}  delta cost = {:.5e}  alpha = {:.4e}   {}'.format(iter,newcost, lastcost,basecost2, newcost-lastcost,alpha,time.ctime()))
#             lastcost = copy.deepcopy(newcost)
#         else:
#             clusterweights = copy.deepcopy(lastgood_clusterweights)
#             alpha *= 0.5
#             recalculate_wgradients = False
#             runcount = 0
#             print('iter {} - no improvement   new cost = {:.5e}  base cost = {:.5e}  base cost2 = {:.5e}  alpha = {:.2e}   {}'.format(iter,newcost, lastcost,basecost2,alpha,time.ctime()))
#
#         # save results on each iteration in case the user wants to abort the run...
#         results = {'costrecord':costrecord, 'clusterweights':clusterweights}
#         outputname = os.path.join(outputdir, 'GDresults_weight.npy')
#         np.save(outputname, results)
#
#         # peek at results
#         best_clusters = np.zeros(nregions)
#         for region in range(nregions):
#             r1 = sum(nclusterlist[:region])
#             r2 = sum(nclusterlist[:(region + 1)])
#             p = clusterweights[r1:r2]
#
#             # look for best match
#             x = np.argmax(p)
#             best_clusters[region] = x
#             best_clusters = best_clusters.astype(int)
#         print('\nbest cluster set so far is : {}'.format(best_clusters))
#
#
#     # look at final results in more detail---------------------------
#     finaloutputstring = ''
#     best_clusters = np.zeros(nregions)
#     for region in range(nregions):
#         r1 = sum(nclusterlist[:region])
#         r2 = sum(nclusterlist[:(region + 1)])
#         print('\nclusterweights region {}'.format(region))
#         p = clusterweights[r1:r2]
#         outputstring = ''
#         for cc in range(nclusterlist[region]):
#             outputstring += '{:.3f} '.format(p[cc])
#         print(outputstring)
#         x = np.argmax(p)
#         best_clusters[region] = x
#         best_clusters = best_clusters.astype(int)
#     print('\nbest cluster set is : {}'.format(best_clusters))
#     print('\n')
#
#     return best_clusters
#
#
# # gradient descent method to find best clusters------------------------------------
# def SAPM_cluster_search2(outputdir, SAPMresultsname, SAPMparametersname, networkfile, DBname, regiondataname, clusterdataname, nprocessors,
#                         samplesplit, samplestart=0, initial_clusters = [], timepoint = 'all', epoch = 'all', betascale = 0.0):
#
#     if not os.path.exists(outputdir): os.mkdir(outputdir)
#
#     # load paradigm data--------------------------------------------------------------------
#     xls = pd.ExcelFile(DBname, engine='openpyxl')
#     df1 = pd.read_excel(xls, 'paradigm1_BOLD')
#     del df1['Unnamed: 0']  # get rid of the unwanted header column
#     fields = list(df1.keys())
#     paradigm = df1['paradigms_BOLD']
#     timevals = df1['time']
#     paradigm_centered = paradigm - np.mean(paradigm)
#     dparadigm = np.zeros(len(paradigm))
#     dparadigm[1:] = np.diff(paradigm_centered)
#
#     # get cluster info and setup for saving information later
#     cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
#     # cluster_properties = cluster_data['cluster_properties']
#     cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)
#     nregions = len(cluster_properties)
#     nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
#     rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
#     namelist_addon = ['R '+n for n in rnamelist]
#     namelist = rnamelist + namelist_addon
#
#     # ---------------------
#     # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
#     prep_data_sem_physio_model_SO(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
#                                   fullgroup=False, normalizevar=False)
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
#
#     # for one set of PCloadings
#     Lweight = 1.0e-20
#     beta = 0.001
#     alpha = 1e0
#     initial_alpha = copy.deepcopy(alpha)
#     alphalimit = 1e-4
#     maxiter = 100
#     subsample = [samplesplit,samplestart]  # [2,0] use every 2nd data set, starting with samplestart
#
#     nclusters_total = np.sum(nclusterlist)
#     clusterweights = np.random.randn(nclusters_total)  # initially set the clusterweights to random values
#     clusterweights = np.ones(nclusters_total)  # initially set the clusterweights to equal values
#     for aa in range(nregions):
#         r1 = sum(nclusterlist[:aa])
#         r2 = sum(nclusterlist[:(aa + 1)])
#         weightset = clusterweights[r1:r2]
#         clusterweights[r1:r2] = weightset/np.sum(weightset)
#
#     if len(initial_clusters) == nregions:
#         clusterweights = np.zeros(nclusters_total)
#         for aa in range(nregions):
#             cluster = initial_clusters[aa]
#             r1 = sum(nclusterlist[:aa])
#             r2 = sum(nclusterlist[:(aa + 1)])
#             weightset = clusterweights[r1:r2]
#             weightset[cluster] = 1.0
#             clusterweights[r1:r2] = weightset
#
#     lastgood_clusterweights = copy.deepcopy(clusterweights)
#
#     # gradient descent to find best cluster combination
#     iter = 0
#     costrecord = []
#     print('starting gradient descent search of weighted clusters at {}'.format(time.ctime()))
#     # recalculate_wgradients = True
#     runcount = np.zeros(nclusters_total)
#     alpha_per_cluster = alpha*np.ones(nclusters_total)
#     alphamax = copy.deepcopy(alpha)
#     runrecord = []
#     while (alphamax > alphalimit) and (iter < maxiter):
#         # subsample[1] = iter % 2   # vary which data sets are used out of the subsample
#         iter += 1
#         # gradients in PCloadings
#         # update one weight at a time
#
#         for weightindex in range(nclusters_total):
#             # if recalculate_wgradients:
#             clusterweights_start = copy.deepcopy(clusterweights)
#
#             SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, clusterweights, paradigm_centered,
#                                 SAPMresultsname, SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
#                                 subsample = subsample, fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#             R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#             basecost = np.sum(1-R2list)
#
#             wgradient, basecost2 = single_weight_gradient(weightindex, beta, betascale, tcdata, clusterweights, nclusterlist, paradigm_centered, SAPMresultsname, SAPMparametersname, subsample, nprocessors, Lweight)
#
#             clusterweights[weightindex] -= alpha_per_cluster[weightindex]*wgradient
#             clusterweights[clusterweights < 0] = 0.0
#             for aa in range(nregions):
#                 r1 = sum(nclusterlist[:aa])
#                 r2 = sum(nclusterlist[:(aa + 1)])
#                 weightset = copy.deepcopy(clusterweights[r1:r2])
#                 clusterweights[r1:r2] = weightset / np.sum(weightset)
#             if iter == 1 and weightindex == 0: lastcost = copy.deepcopy(basecost)
#             # else:
#             #     print('not calculating load gradients')
#
#             SAPMresults, search_data_name = sem_physio_model_weightedclusters(tcdata, clusterweights, paradigm_centered,
#                                 SAPMresultsname, SAPMparametersname, nitermax = 100, alpha_limit = 1e-5,
#                                 subsample = subsample, fixed_beta_vals = [], betascale = betascale, verbose = False, nprocessors = nprocessors)
#             R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#             newcost = np.sum(1-R2list)
#
#             costrecord += [basecost]
#             entry = {'clusterweights_start':clusterweights_start, 'clusterweights':clusterweights, 'newcost':newcost, 'lastcost':lastcost, 'basecost':basecost, 'basecost2':basecost2}
#             runrecord.append(entry)
#
#             if newcost < basecost:
#                 lastgood_clusterweights = copy.deepcopy(clusterweights)
#                 # recalculate_wgradients = True
#                 runcount[weightindex] += 1
#                 if runcount[weightindex] > 2:
#                     alpha_per_cluster[weightindex] = np.min([initial_alpha, 1.3*alpha_per_cluster[weightindex]])
#                 print('iter {} index {}  new cost = {:.3e}  base cost = {:.3e}  delta cost = {:.3e}  alpha = {:.2e}   {}'.format(iter, weightindex,newcost, lastcost, newcost-lastcost,alpha_per_cluster[weightindex],time.ctime()))
#                 lastcost = copy.deepcopy(newcost)
#             else:
#                 clusterweights = copy.deepcopy(lastgood_clusterweights)
#                 alpha_per_cluster[weightindex] *= 0.5
#                 # recalculate_wgradients = False
#                 runcount[weightindex] = 0
#                 print('iter {} index {} - no improvement   new cost = {:.3e}  base cost = {:.3e}  alpha = {:.2e}   {}'.format(iter,weightindex,newcost, lastcost,alpha_per_cluster[weightindex],time.ctime()))
#             alphamax = np.max(alpha_per_cluster)
#
#         # save results on each iteration in case the user wants to abort the run...
#         results = {'costrecord':costrecord, 'clusterweights':clusterweights}
#         outputname = os.path.join(outputdir, 'GDresults_weight.npy')
#         np.save(outputname, results)
#
#         # peek at results
#         best_clusters = np.zeros(nregions)
#         for region in range(nregions):
#             r1 = sum(nclusterlist[:region])
#             r2 = sum(nclusterlist[:(region + 1)])
#             p = clusterweights[r1:r2]
#
#             # look for best match
#             x = np.argmax(p)
#             best_clusters[region] = x
#             best_clusters = best_clusters.astype(int)
#         print('\nbest cluster set so far is : {}'.format(best_clusters))
#
#
#     # look at final results in more detail---------------------------
#     finaloutputstring = ''
#     best_clusters = np.zeros(nregions)
#     for region in range(nregions):
#         r1 = sum(nclusterlist[:region])
#         r2 = sum(nclusterlist[:(region + 1)])
#         print('\nclusterweights region {}'.format(region))
#         p = clusterweights[r1:r2]
#         outputstring = ''
#         for cc in range(nclusterlist[region]):
#             outputstring += '{:.3f} '.format(p[cc])
#         print(outputstring)
#         x = np.argmax(p)
#         best_clusters[region] = x
#         best_clusters = best_clusters.astype(int)
#     print('\nbest cluster set is : {}'.format(best_clusters))
#     print('\n')
#
#     return best_clusters
#
# # gradient descent method to find best clusters------------------------------------
# def SAPM_cluster_search(outputdir, SAPMresultsname, SAPMparametersname, networkfile, DBname, regiondataname,
#                         clusterdataname, nprocessors,
#                         samplesplit, samplestart=0, initial_clusters=[], timepoint='all', epoch='all', betascale=0.0):
#     if not os.path.exists(outputdir): os.mkdir(outputdir)
#
#     # load paradigm data--------------------------------------------------------------------
#     xls = pd.ExcelFile(DBname, engine='openpyxl')
#     df1 = pd.read_excel(xls, 'paradigm1_BOLD')
#     del df1['Unnamed: 0']  # get rid of the unwanted header column
#     fields = list(df1.keys())
#     paradigm = df1['paradigms_BOLD']
#     timevals = df1['time']
#     paradigm_centered = paradigm - np.mean(paradigm)
#     dparadigm = np.zeros(len(paradigm))
#     dparadigm[1:] = np.diff(paradigm_centered)
#
#     # get cluster info and setup for saving information later
#     cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
#     # cluster_properties = cluster_data['cluster_properties']
#     cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)
#     nregions = len(cluster_properties)
#     nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
#     rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
#     namelist_addon = ['R ' + n for n in rnamelist]
#     namelist = rnamelist + namelist_addon
#
#     # ---------------------
#     # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
#     prep_data_sem_physio_model_SO(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
#                                   fullgroup=False, normalizevar=False, filter_tcdata = True)
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
#     # need to get principal components for each region to model the clusters as a continuum
#
#     nclusters_total, tsize_total = np.shape(tcdata)
#     component_data = np.zeros(np.shape(tcdata))
#     average_data = np.zeros(np.shape(tcdata))
#     ncmax = np.max(nclusterlist)
#     original_loadings = np.zeros((nregions, ncmax, ncmax))
#     weights = np.zeros((nregions, ncmax))
#     for regionnum in range(nregions):
#         r1 = sum(nclusterlist[:regionnum])
#         r2 = sum(nclusterlist[:(regionnum + 1)])
#
#         nstates = nclusterlist[regionnum]  # the number to look at
#         pca = PCA(n_components=nstates)
#         tcdata_region = tcdata[r1:r2, :]
#         pca.fit(tcdata_region)
#         S_pca_ = pca.fit(tcdata_region).transform(tcdata_region)
#
#         components = pca.components_
#         evr = pca.explained_variance_ratio_
#         # use components in SAPM in place of original region data
#
#         # get loadings
#         mu = np.mean(tcdata_region, axis=0)
#         mu = np.repeat(mu[np.newaxis, :], nstates, axis=0)
#
#         loadings = pca.transform(tcdata_region)
#         fit_check = (loadings @ components) + mu
#
#         component_data[r1:r2, :] = components
#         average_data[r1:r2, :] = mu
#         original_loadings[regionnum, :nstates, :nstates] = loadings
#         weights[regionnum, :nstates] = evr
#
#     # scale component_data to make original_loadings near maximum of 1
#     PCscalefactor = original_loadings.max()
#     original_loadings /= PCscalefactor
#     component_data *= PCscalefactor
#     PCparams = {'components': component_data, 'average': average_data, 'loadings': original_loadings,
#                 'weights': weights}
#
#     # for one set of PCloadings
#     Lweight = 1.0e-20
#     beta = 0.1
#     alpha = 1.0
#     initial_alpha = copy.deepcopy(alpha)
#     alphalimit = 1e-4
#     maxiter = 50
#     subsample = [samplesplit, samplestart]  # [2,0] use every 2nd data set, starting with samplestart
#
#     PCloadings = 1e-4 * np.random.randn(nclusters_total)
#     for aa in range(nregions):
#         L = original_loadings[aa, :, :]
#         r1 = sum(nclusterlist[:aa])
#         r2 = sum(nclusterlist[:(aa + 1)])
#         PCloadings[r1:r2] = np.mean(L,
#                                     axis=0)  # initially set the PCloadings to the average for all clusters in the region
#
#     if len(initial_clusters) == nregions:
#         for aa in range(nregions):
#             L = original_loadings[aa, :, :]
#             cluster = initial_clusters[aa]
#             r1 = sum(nclusterlist[:aa])
#             r2 = sum(nclusterlist[:(aa + 1)])
#             PCloadings[r1:r2] = L[cluster, :]
#
#     lastgood_PCloadings = copy.deepcopy(PCloadings)
#
#     # gradient descent to find best cluster combination
#     iter = 0
#     costrecord = []
#     print('starting gradient descent search of clusters at {}'.format(time.ctime()))
#     recalculate_load_gradients = True
#     runcount = 0
#     while (alpha > alphalimit) and (iter < maxiter):
#         # subsample[1] = iter % 2   # vary which data sets are used out of the subsample
#         iter += 1
#         # gradients in PCloadings
#         if recalculate_load_gradients:
#             SAPMresults, search_data_name = sem_physio_model_PCAclusters(PCparams, PCloadings, paradigm_centered,
#                                 SAPMresultsname, SAPMparametersname,nitermax=150, alpha_limit=1e-5,
#                                  subsample=subsample, fixed_beta_vals=[], betascale=betascale, verbose=False,
#                                  nprocessors=nprocessors)
#
#             R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#             basecost = np.sum(1 - R2list) + Lweight * np.sum(np.abs(PCloadings))
#
#             load_gradients, basecost2 = loadings_gradients(beta, betascale, PCparams, PCloadings, paradigm_centered,
#                                                           SAPMresultsname, SAPMparametersname, subsample, nprocessors,
#                                                           Lweight)
#             if iter == 1: lastcost = copy.deepcopy(basecost)
#         else:
#             print('not calculating load gradients')
#         PCloadings -= alpha * load_gradients
#
#         SAPMresults, search_data_name = sem_physio_model_PCAclusters(PCparams, PCloadings, paradigm_centered,
#                     SAPMresultsname, SAPMparametersname, nitermax=150, alpha_limit=1e-5, subsample=subsample, fixed_beta_vals=[],
#                     betascale=betascale, verbose=False, nprocessors=nprocessors)
#
#         print('size of SAPMresults is {}'.format(np.shape(SAPMresults)))
#         R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
#         newcost = np.sum(1 - R2list) + Lweight * np.sum(np.abs(PCloadings))
#         R2cost_portion = np.sum(1 - R2list)
#         L1cost_portion = Lweight * np.sum(np.abs(PCloadings))
#
#         costrecord += [basecost]
#
#         if newcost < lastcost:
#             lastgood_PCloadings = copy.deepcopy(PCloadings)
#             recalculate_load_gradients = True
#             runcount += 1
#             if runcount > 2:
#                 alpha = np.min([5.0*initial_alpha, 1.3*alpha])
#                 runcount = 0
#             print('iter {}  new cost = {:.3e}  base cost = {:.3e}  delta cost = {:.3e}  alpha = {:.2e}   {}'.format(iter,
#                         newcost, lastcost,newcost - lastcost,alpha, time.ctime()))
#             lastcost = copy.deepcopy(newcost)
#         else:
#             PCloadings = copy.deepcopy(lastgood_PCloadings)
#             alpha *= 0.5
#             recalculate_load_gradients = False
#             runcount = 0
#             print(
#                 'iter {} - no improvement   new cost = {:.3e}  base cost = {:.3e}  R2 portion = {:.2e}   L1 portion = {:.2e}  alpha = {:.2e}   {}'.format(
#                     iter, newcost, lastcost, R2cost_portion, L1cost_portion, alpha, time.ctime()))
#
#         # save results on each iteration in case the user wants to abort the run...
#         results = {'costrecord': costrecord, 'PCloadings': PCloadings, 'original_loadings': original_loadings,
#                    'PCscalefactor': PCscalefactor}
#         outputname = os.path.join(outputdir, 'GDresults2.npy')
#         np.save(outputname, results)
#
#         # peek at results
#         best_clusters = np.zeros(nregions)
#         for region in range(nregions):
#             L = original_loadings[region, :, :]
#             r1 = sum(nclusterlist[:region])
#             r2 = sum(nclusterlist[:(region + 1)])
#             p = PCloadings[r1:r2]
#
#             # look for best match
#             nclusters = nclusterlist[region]
#             d = np.zeros(nclusters)
#             w = weights[region, :]
#             for cc in range(nclusters):
#                 d[cc] = np.sqrt(np.sum(w * (L[cc, :] - p) ** 2))
#             x = np.argmin(d)
#             best_clusters[region] = x
#             best_clusters = best_clusters.astype(int)
#         print('\nbest cluster set so far is : {}'.format(best_clusters))
#
#     # look at final results in more detail---------------------------
#     finaloutputstring = ''
#     best_clusters = np.zeros(nregions)
#     for region in range(nregions):
#         print('\noriginal loadings region {}'.format(region))
#         L = original_loadings[region, :, :]
#         nclusters = nclusterlist[region]
#         outputstring = ''
#         for cc in range(nclusters):
#             outputstring += 'cluster{}:  '.format(cc)
#             for dd in range(nclusters):
#                 outputstring += '{:.3f} '.format(L[cc, dd])
#             outputstring += '\n'
#         print(outputstring)
#
#         r1 = sum(nclusterlist[:region])
#         r2 = sum(nclusterlist[:(region + 1)])
#         print('\nPCloadings region {}'.format(region))
#         p = PCloadings[r1:r2]
#         outputstring = ''
#         for cc in range(nclusters):
#             outputstring += '{:.3f} '.format(p[cc])
#         print(outputstring)
#
#         # look for best match
#         nclusters = nclusterlist[region]
#         d = np.zeros(nclusters)
#         w = weights[region, :]
#         for cc in range(nclusters):
#             d[cc] = np.sqrt(np.sum(w * (L[cc, :] - p) ** 2))
#
#         # convert distances to confidence level that each cluster is the best choice
#         proximity_score = 1.0 / (d ** 2 + 1.0e-3)
#         proximity_percent = 100.0 * proximity_score / np.sum(proximity_score)
#
#         print('\ndistance between PCloadings and original {}'.format(region))
#         outputstring = ''
#         finaloutputstring += '\nRegion {} cluster percents:  '.format(region)
#         for cc in range(nclusters):
#             outputstring += 'cluster{}  {:.3f}   estimated {:.1f} percent best choice \n'.format(cc, d[cc],
#                                                                                                  proximity_percent[cc])
#             finaloutputstring += '{:.1f} '.format(proximity_percent[cc])
#         print(outputstring)
#
#         x = np.argmin(d)
#         best_clusters[region] = x
#         best_clusters = best_clusters.astype(int)
#     print('\nbest cluster set is : {}'.format(best_clusters))
#     print('\n')
#     print(finaloutputstring)
#
#     return best_clusters


# gradient descent method to find best clusters------------------------------------
def SAPM_cluster_stepsearch(outputdir, SAPMresultsname, SAPMparametersname, networkfile, DBname, regiondataname,
                        clusterdataname, samplesplit, samplestart=0, initial_clusters=[], timepoint='all', epoch='all', betascale=0.1):
    overall_start_time_text = time.ctime()
    overall_start_time = time.time()

    if not os.path.exists(outputdir): os.mkdir(outputdir)

    # load paradigm data--------------------------------------------------------------------
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    df1 = pd.read_excel(xls, 'paradigm1_BOLD')
    del df1['Unnamed: 0']  # get rid of the unwanted header column
    fields = list(df1.keys())
    paradigm = df1['paradigms_BOLD']
    timevals = df1['time']
    paradigm_centered = paradigm - np.mean(paradigm)
    dparadigm = np.zeros(len(paradigm))
    dparadigm[1:] = np.diff(paradigm_centered)

    # get cluster info and setup for saving information later
    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    # cluster_properties = cluster_data['cluster_properties']
    cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)
    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]
    namelist_addon = ['R ' + n for n in rnamelist]
    namelist = rnamelist + namelist_addon

    # ---------------------
    # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
    prep_data_sem_physio_model_SO(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch,
                                  fullgroup=False, normalizevar=True, filter_tcdata = False)
    print('-----------------------------------------------------------------------------------')
    print('NOTE: data are being adjusted for normalized variance for the cluster search method')
    print('-----------------------------------------------------------------------------------')

    SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
    tcdata = SAPMparams['tcdata_centered']  # data for all regions/clusters concatenated along time dimension for all runs
    # need to get principal components for each region to model the clusters as a continuum

    nclusters_total, tsize_total = np.shape(tcdata)

    # for one set of PCloadings
    # Lweight = 1.0e-20
    # beta = 0.1
    # alpha = 1.0
    # initial_alpha = copy.deepcopy(alpha)
    # alphalimit = 1e-4
    maxiter = 50
    subsample = [samplesplit,samplestart]  # [2,0] use every 2nd data set, starting with samplestart

    full_rnum_base = np.array([np.sum(nclusterlist[:x]) for x in range(len(nclusterlist))]).astype(int)
    initial_clusters = np.array(initial_clusters)
    if (initial_clusters < 0).any():
        fixed_clusters = np.where(initial_clusters > 0)[0]
    else:
        fixed_clusters = []
    cluster_numbers = copy.deepcopy(initial_clusters)
    if (len(initial_clusters) != nregions) or (len(fixed_clusters) > 0):
        # pick random starting clusters
        cluster_numbers = np.zeros(nregions)
        for nn in range(nregions):
            cnum = np.random.choice(range(nclusterlist[nn]))
            cluster_numbers[nn] = cnum
        cluster_numbers = np.array(cluster_numbers).astype(int)
        if len(fixed_clusters) > 0:
            cluster_numbers[fixed_clusters] = initial_clusters[fixed_clusters]

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
        nitermax = 30
        initial_nitermax_stage1 = 10
        initial_nsteps_stage1 = 10

    output = sem_physio_model2(cluster_numbers+full_rnum_base, paradigm_centered, SAPMresultsname, SAPMparametersname,
                               fixed_beta_vals=[], betascale=betascale, nitermax = nitermax, verbose=False,
                               initial_nitermax_stage1=initial_nitermax_stage1, initial_nsteps_stage1=initial_nsteps_stage1)

    SAPMresults = np.load(output,allow_pickle=True)

    # SAPMresults = sem_physio_model2_fast(tcdata, cluster_numbers+full_rnum_base, paradigm_centered, SAPMresultsname,
    #                        SAPMparametersname, nitermax=150, alpha_limit=1e-5,
    #                        subsample=subsample, fixed_beta_vals=[], betascale=0.0, verbose=False,
    #                        nprocessors=1)

    R2list = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
    basecost = np.sum(1 - R2list)
    lastcost = copy.deepcopy(basecost)
    results_record = []
    while converging and (iter < maxiter):
        iter += 1
        nbetterclusters = 0
        random_region_order = list(range(nregions))
        np.random.shuffle(random_region_order)
        for nnn in random_region_order:
            cost_values = np.zeros(nclusterlist[nnn])
            print('testing region {}'.format(nnn))
            if nnn in fixed_clusters:
                print('cluster for region {} is fixed at {}'.format(nnn,cluster_numbers[nnn]))
            else:
                for ccc in range(nclusterlist[nnn]):
                    test_clusters = copy.deepcopy(cluster_numbers)
                    if test_clusters[nnn] == ccc:   # no change in cluster number from last run
                        cost_values[ccc] = lastcost
                        print('  using cluster {}  average R2 is {:.3f} - current cluster'.format(ccc,cost_values[ccc]))
                    else:
                        test_clusters[nnn] = ccc
                        output = sem_physio_model2(test_clusters+full_rnum_base, paradigm_centered, SAPMresultsname, SAPMparametersname,
                                                        fixed_beta_vals=[], betascale=betascale, nitermax=nitermax, verbose=False,
                                                        initial_nitermax_stage1=initial_nitermax_stage1, initial_nsteps_stage1=initial_nsteps_stage1)
                        SAPMresults = np.load(output, allow_pickle=True)

                        # SAPMresults = sem_physio_model2_fast(tcdata, test_clusters+full_rnum_base, paradigm_centered, SAPMresultsname,
                        #                                      SAPMparametersname, nitermax=150, alpha_limit=1e-5,
                        #                                      subsample=subsample, fixed_beta_vals=[], betascale=0.0, verbose=False,
                        #                                      nprocessors=1)

                        # print('size of SAPMresults is {}'.format(np.shape(SAPMresults)))
                        R2list = np.array([SAPMresults[x]['R2avg'] for x in range(len(SAPMresults))])
                        R2list2 = np.array([SAPMresults[x]['R2total'] for x in range(len(SAPMresults))])
                        cost_values[ccc] = np.sum(1 - R2list)
                        entry = {'R2list':R2list, 'R2list2':R2list2, 'region':nnn, 'cluster':ccc}
                        results_record.append(entry)
                        print('  using cluster {}  average R2 is {:.3f}'.format(ccc,cost_values[ccc]))

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

    outputname = os.path.join(outputdir, 'step_descent_record.npy')
    np.save(outputname, results_record)
    print('results record written to {}'.format(outputname))

    overall_end_time_text = time.ctime()
    overall_end_time = time.time()
    dtime = overall_end_time-overall_start_time
    dtimem = np.floor(dtime/60).astype(int)
    dtimes = np.round(dtime % 60).astype(int)
    print('Cluster search started at {}\n             and ended at {}\n     {} minutes {} sec total'.format(overall_start_time_text, overall_end_time_text, dtimem,dtimes))
    return cluster_numbers


# main program
def SAPMrun(cnums, regiondataname, clusterdataname, SAPMresultsname, SAPMparametersname, networkfile, DBname, timepoint,
            epoch, betascale = 0.01, reload_existing = False, multiple_output = False):
    # load paradigm data--------------------------------------------------------------------
    xls = pd.ExcelFile(DBname, engine='openpyxl')
    df1 = pd.read_excel(xls, 'paradigm1_BOLD')
    del df1['Unnamed: 0']  # get rid of the unwanted header column
    fields = list(df1.keys())
    paradigm = df1['paradigms_BOLD']
    timevals = df1['time']
    paradigm_centered = paradigm - np.mean(paradigm)
    dparadigm = np.zeros(len(paradigm))
    dparadigm[1:] = np.diff(paradigm_centered)

    # load some data, setup some parameters...
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(networkfile)
    ncluster_list = np.array([nclusterlist[x]['nclusters'] for x in range(len(nclusterlist))])
    cluster_name = [nclusterlist[x]['name'] for x in range(len(nclusterlist))]
    not_latent = [x for x in range(len(cluster_name)) if 'intrinsic' not in cluster_name[x]]
    ncluster_list = ncluster_list[not_latent]
    full_rnum_base = [np.sum(ncluster_list[:x]) for x in range(len(ncluster_list))]
    namelist = [cluster_name[x] for x in not_latent]
    namelist += ['Rtotal']
    namelist += ['R ' + cluster_name[x] for x in not_latent]

    # full_rnum_base =  np.array([0,5,10,15,20,25,30,35,40,45])
    #
    # namelist = ['C6RD',  'DRt', 'Hypothalamus','LC', 'NGC', 'NRM', 'NTS', 'PAG', 'PBN', 'Thalamus',
    #         'Rtotal', 'R C6RD',  'R DRt', 'R Hyp','R LC', 'R NGC', 'R NRM', 'R NTS', 'R PAG',
    #         'R PBN', 'R Thal']

    # starting values
    cnums_original = copy.deepcopy(cnums)
    excelsheetname = 'clusters'

    # run the analysis with SAPM
    clusterlist = np.array(cnums) + full_rnum_base
    # prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
    if multiple_output:
        prep_data_sem_physio_model(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
    else:
        prep_data_sem_physio_model_SO(networkfile, regiondataname, clusterdataname, SAPMparametersname, timepoint, epoch)
    output = sem_physio_model2(clusterlist, paradigm_centered, SAPMresultsname, SAPMparametersname,
                               fixed_beta_vals = [], betascale = betascale)

    SAPMresults = np.load(output, allow_pickle=True)
    NP = len(SAPMresults)
    R2list =np.zeros(len(SAPMresults))
    R2list2 =np.zeros(len(SAPMresults))
    for nperson in range(NP):
        R2list[nperson] = SAPMresults[nperson]['R2avg']
        R2list2[nperson] = SAPMresults[nperson]['R2total']
        # R2list[nperson] = SAPMresults[nperson][0]['R2total']
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
    sources = np.where(m == 1)[0]
    rsources = [beta_list[ss]['pair'][0] for ss in sources]
    nsources = len(sources)
    nregions = len(rnamelist)
    checkdims = np.shape(Sinput_avg)
    if np.ndim(Sinput_avg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    # ncon,tsize = np.shape(Sconn_avg)
    # if ncon < len(beta_list):
    #     single_output = True
    # else:
    #     single_output = False

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
    sources = np.where(m == 1)[0]
    rsources = [beta_list[ss]['pair'][0] for ss in sources]
    nsources = len(sources)
    nregions = len(rnamelist)
    checkdims = np.shape(Sinput_reg)
    if np.ndim(Sinput_reg) > 2:  nv = checkdims[2]
    tsize = checkdims[1]

    # get beta values from Mconn
    # print('size of Sconn_reg is {}'.format(np.shape(Sconn_reg)))
    # ncon, tsize, nregparams = np.shape(Sconn_reg)
    # if ncon < len(beta_list):
    #     single_output = True
    # else:
    #     single_output = False

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

    # m = Mconn_avg[:,sources[0]]
    # targets2ndlevel_list = np.where(m != 0.)[0]
    # textlist = []
    # for ss in sources:
    #     text = betanamelist[ss] + ': '
    #     beta = Mconn_avg[targets2ndlevel_list,ss]
    #     for ss2 in range(len(beta)):
    #         valtext = '{:.2f} '.format(beta[ss2])
    #         text1 = '{}{}'.format(valtext,betanamelist[targets2ndlevel_list[ss2]])
    #         text += text1 + ', '
    #     textlist += [text[:-1]]

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
        # axs[ss,0].plot(x, tc1+tc1p, '-r')
        # axs[ss,0].plot(x, tc1-tc1p, '--r')
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
    # p, f = os.path.split(SAPMresultsname)

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
        tcf1 = fit_avg[regionlist[nn], :]
        t = np.array(range(len(tc1)))

        if len(Sinput_sem) > 0:
            tc1_sem = Sinput_sem[regionlist[nn], :]
            axs[nn].errorbar(t, tc1, tc1_sem, marker = 'o', markerfacecolor = 'b', markeredgecolor = 'b', linestyle = '-', color = 'b', linewidth=1, markersize=4)
        else:
            axs[nn].plot(t, tc1, '-ob', linewidth=1, markersize=4)

        if len(fit_sem) > 0:
            tcf1_sem = fit_sem[regionlist[nn], :]
            axs[nn].errorbar(t, tcf1, tcf1_sem, marker = 'o', markerfacecolor = 'r', markeredgecolor = 'r', linestyle = '-', color = 'r', linewidth=1, markersize=4)
        else:
            axs[nn].plot(t, tcf1, '-xr', linewidth=1, markersize=4)

        axs[nn].set_title('target {}'.format(rnamelist[regionlist[nn]]))
        if setylim:
            axs[nn].set_ylim((ymin,ymax))

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

    # p, f = os.path.split(SAPMresultsname)
    if display_in_GUI:
        svgname = 'output figure written to GUI ... not saved'
        print(svgname)
        TargetCanvas.draw()
    else:
        svgname = os.path.join(outputdir, 'Avg_' + nametag + '.svg')
        plt.savefig(svgname)

    return svgname, Rtext_record, Rval_record


def write_Mconn_values2(Mconn, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format = 'f', pthresh = 0.05,
                        sigflag = [], multiple_output = False):
    # get beta values from Mconn
    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mconn)

    if np.size(sigflag) == 0:
        sigflag = np.zeros(np.shape(Mconn))

    Tvals = Mconn / (Mconn_sem + 1.0e-20)
    Tthresh = stats.t.ppf(1 - pthresh, NP - 1)
    if np.isnan(Tthresh):  Tthresh = 0.0


    # if nr1 < len(betanamelist):
    #     single_output = True
    # else:
    #     single_output = False

    if multiple_output:
        labeltext_record = []
        valuetext_record = []
        Ttext_record = []
        T_record = []
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
                    else:
                        valuetext = '{:.3e} {} {:.3e} '.format(Mconn[n1, n2], chr(177), Mconn_sem[n1, n2])
                        Ttext = 'T = {:.2e} '.format(Tvals[n1,n2])

                    labeltext_record += [labeltext]
                    valuetext_record += [valuetext]
                    Ttext_record += [Ttext]
                    T_record += [T]
                    if showval:
                        print(labeltext)
                        print(valuetext)
                        print(Ttext)
    else:
        labeltext_record = []
        valuetext_record = []
        Ttext_record = []
        T_record = []
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
                else:
                    valuetext = '{:.3e} {} {:.3e} '.format(Mconn[tpair[1],tpair[0]], chr(177), Mconn_sem[tpair[1],tpair[0]])
                    Ttext = 'T = {:.2e} '.format(Tvals[tpair[1],tpair[0]])

                labeltext_record += [labeltext]
                valuetext_record += [valuetext]
                Ttext_record += [Ttext]
                T_record += [T]
                if showval:
                    print(labeltext)
                    print(valuetext)
                    print(Ttext)
    return labeltext_record, valuetext_record, Ttext_record, T_record, Tthresh



def write_Mreg_values(Mint, Mslope, R2, betanamelist, rnamelist, beta_list, format = 'f', R2thresh = 0.1,
                      sigflag = [], multiple_output = False):

    nregions = len(rnamelist)
    nr1, nr2 = np.shape(Mslope)

    # if nr1 < len(betanamelist):
    #     single_output = True
    # else:
    #     single_output = False

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

#
# def plot_correlated_results(SAPMresultsname, SAPMparametersname, connection_name, covariates, figurenumber = 1):
#     outputdir = r'D:\threat_safety_python\individual_differences\fixed_C6RD0'
#     # SAPMresultsname = os.path.join(outputdir, 'SEMphysio_model.npy')
#     SAPMresults_load = np.load(SAPMresultsname, allow_pickle=True)
#
#     # SAPMparametersname = os.path.join(outputdir, 'SEMparameters_model5.npy')
#     SAPMparams = np.load(SAPMparametersname, allow_pickle=True).flat[0]
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     rnamelist = SAPMparams['rnamelist']
#     betanamelist = SAPMparams['betanamelist']
#     beta_list = SAPMparams['beta_list']
#     Mconn = SAPMparams['Mconn']
#
#     # for nperson in range(NP)
#     NP = len(SAPMresults_load)
#     nconn, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
#     nbeta = np.shape(SAPMresults_load[0]['betavals'])[0]
#     beta_record = np.zeros((NP,nbeta))
#     for nn in range(NP):
#         beta_record[nn,:] = SAPMresults_load[nn]['betavals']
#
#     labeltext_record, sources_per_target, intrinsic_flag = betavalue_labels(csource, ctarget, rnamelist, betanamelist, beta_list, Mconn)
#
#     x = labeltext_record.index(connection_name)
#     beta = beta_record[:,x]
#
#     # prep regression lines
#     b, fit, R2 = pydisplay.simple_GLMfit(covariates, beta)
#
#     plt.close(figurenumber)
#     fig = plt.figure(figurenumber)
#     plt.plot(covariates, beta, color=(0, 0, 0), linestyle='None', marker='o', markerfacecolor=(0, 0, 0),
#                     markersize=4)
#     plt.plot(covariates, fit, color=(0, 0, 0), linestyle='solid', marker='None')
#     textlabel = '{}'.format(connection_name)
#     plt.title(textlabel)


def display_matrix(M,columntitles,rowtitles,outputformat = 'float', excelname = ''):

    # columns = [name[:3] +' in' for name in betanamelist]
    # rows = [name[:3] for name in betanamelist]

    df = pd.DataFrame(M,columns = columntitles, index = rowtitles)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    if outputformat == 'float':
        pd.options.display.float_format = '{:.2f}'.format
    else:
        pd.options.display.float_format = '{:.0f}'.format
    print(df)

    if len(excelname) > 0:
        df.to_excel(excelname)


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
    intrinsic_flag = np.zeros(nbeta)
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
            intrinsic_flag[nn] = 1
        else:
            ts = rnamelist[tpair[0]]
            if len(ts) > 4:  ts = ts[:4]
        tt = rnamelist[tpair[1]]
        if len(tt) > 4:  tt = tt[:4]

        sname = betanamelist[n2]
        spair = beta_list[n2]['pair']
        if spair[0] >= nregions:
            ss = 'int{}'.format(spair[0] - nregions)
            intrinsic_flag[nn] = 1
        else:
            ss = rnamelist[spair[0]]
            if len(ss) > 4:  ss = ss[:4]
        st = rnamelist[spair[1]]
        if len(st) > 4:  st = st[:4]
        labeltext = '{}-{}-{}'.format(ss, st, tt)

        labeltext_record += [labeltext]

    return labeltext_record, sources_per_target, intrinsic_flag



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
    # p, f = os.path.split(SAPMresultsname)
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

    # Sinput_feature_reg_sorted = Sinput_feature_reg[si,:]
    # textoutputs = {'regions': np.array(cname_list)[si], 'int': np.array(Sinput_feature_reg[si,0]), 'slope': np.array(Sinput_feature_reg[si,1]),
    #                'R2': np.array(Sinput_feature_reg[si,2]), 'Z': np.array(Sinput_feature_reg[si,3]), 'p': np.array(Sinput_feature_reg[si,4]),
    #                'Z thresh': np.array(Zthresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
    # p, f = os.path.split(SAPMresultsname)
    df2 = pd.DataFrame(textoutputs)
    xlname = os.path.join(outputdir, descriptor + '.xlsx')
    with pd.ExcelWriter(xlname) as writer:
        df1.to_excel(writer, sheet_name='Sconn')
        df2.to_excel(writer, sheet_name = 'Sinput')

    outputname = xlname



def display_SAPM_results(window, outputnametag, covariates, outputtype, outputdir, SAPMparametersname, SAPMresultsname,
                         variation_number, group, target = '', pthresh = 0.05, setylimits = [], TargetCanvas = [],
                         display_in_GUI = False, multiple_output = False):
    # options of results to display:
    # 1) average input time-courses compared with model input
    # 2) modelled input signaling with corresponding source time-courses (outputs from source regions)
    # 3) t-test comparisons between groups, or w.r.t. zero (outputs to excel files)
    # 4) regression with continuous covariate
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']

    # load SAPM parameters
    VN = variation_number
    # print('displaying variation number {}'.format(VN))
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
    # NP,nvariations = np.shape(SAPMresults_load)
    NP = len(SAPMresults_load)

    # resultscheck = np.zeros((NP, 4))
    # nbeta, tsize_full = np.shape(SAPMresults_load[0][0]['Sconn'])
    nbeta, tsize_full = np.shape(SAPMresults_load[0]['Sconn'])
    ncon = nbeta - Nintrinsic
    # paradigm_centered = SAPMresults_load[0][0]['fintrinsic_base']  # model paradigm used for fixed pattern latent inputs
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
    ftemp = paradigm_centered[et1:et2]

    Mrecord = np.zeros((nbeta, nbeta, NP))
    R2totalrecord = np.zeros(NP)
    for nperson in range(NP):
        # Sinput = SAPMresults_load[nperson][VN]['Sinput']
        # Sconn = SAPMresults_load[nperson][VN]['Sconn']
        # Minput = SAPMresults_load[nperson][VN]['Minput']
        # Mconn = SAPMresults_load[nperson][VN]['Mconn']
        # beta_int1 = SAPMresults_load[nperson][VN]['beta_int1']
        # R2total = SAPMresults_load[nperson][VN]['R2total']
        # Meigv = SAPMresults_load[nperson][VN]['Meigv']
        # betavals = SAPMresults_load[nperson][VN]['betavals']

        Sinput = SAPMresults_load[nperson]['Sinput']
        Sconn = SAPMresults_load[nperson]['Sconn']
        Minput = SAPMresults_load[nperson]['Minput']
        Mconn = SAPMresults_load[nperson]['Mconn']
        beta_int1 = SAPMresults_load[nperson]['beta_int1']
        R2total = SAPMresults_load[nperson]['R2total']
        Meigv = SAPMresults_load[nperson]['Meigv']
        betavals = SAPMresults_load[nperson]['betavals']

        nruns = nruns_per_person[nperson]
        fintrinsic1 = np.array(list(ftemp) * nruns_per_person[nperson])

        # ---------------------------------------------------
        fit, Mintrinsic, Meigv, err = network_eigenvector_method(Sinput, Minput, Mconn, fintrinsic_count,
                                                                 vintrinsic_count, beta_int1, fintrinsic1)
        nr, tsize_total = np.shape(Sinput)
        tsize = (tsize_total / nruns).astype(int)
        nbeta,tsize2 = np.shape(Sconn)

        if nperson == 0:
            Sinput_total = np.zeros((nr,tsize, NP))
            Sconn_total = np.zeros((nbeta,tsize, NP))
            fit_total = np.zeros((nr,tsize, NP))

        tc = Sinput
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        Sinput_total[:,:,nperson] = tc1

        tc = Sconn
        tc1 = np.mean(np.reshape(tc, (nbeta, nruns, tsize)), axis=1)
        Sconn_total[:,:,nperson] = tc1

        tc = fit
        tc1 = np.mean(np.reshape(tc, (nr, nruns, tsize)), axis=1)
        fit_total[:,:,nperson] = tc1

        Mrecord[:, :, nperson] = Mconn
        R2totalrecord[nperson] = R2total

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
    p = covariates[np.newaxis, g1]
    if len(np.unique(p)) > len(g1)/3:  # assume the values are continuous
        continuouscov = True
        p -= np.mean(p)
        G = np.concatenate((np.ones((1, len(g1))),p), axis=0) # put the intercept term first
    else:
        continuouscov = False

    #-------------------------------------------------------------------------------------
    # significance of average Mconn values -----------------------------------------------
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']
    Mconn_avg = np.mean(Mrecord[:, :, g1], axis=2)
    Mconn_sem = np.std(Mrecord[:, :, g1], axis=2) / np.sqrt(len(g1))
    if outputtype == 'B_Significance':
        descriptor = outputnametag + '_Bsig'
        # pthresh = 0.05
        # Tthresh = stats.t.ppf(1 - pthresh, NP - 1)
        print('\n\nAverage B values')
        labeltext, valuetext, Ttext, T, Tthresh = write_Mconn_values2(Mconn_avg, Mconn_sem, NP, betanamelist, rnamelist, beta_list, format='f', pthresh=pthresh, multiple_output=multiple_output)

        pthresh_list = ['{:.3e}'.format(pthresh)]*len(Ttext)
        Tthresh_list = ['{:.3f}'.format(Tthresh)]*len(Ttext)

        Rtextlist = [' ']*10
        Rvallist = [0]*10

        # sort output by magnitude of T
        si = np.argsort(np.abs(T))[::-1]
        T = np.array(T)[si]
        si2 = np.where(T < 1e3)  # dont write out values where B is always = 1
        textoutputs = {'regions': np.array(labeltext)[si[si2]], 'B': np.array(valuetext)[si[si2]], 'T': np.array(Ttext)[si[si2]],
                       'T thresh': np.array(Tthresh_list)[si[si2]], 'p thresh': np.array(pthresh_list)[si[si2]]}
        # p, f = os.path.split(SAPMresultsname)
        df = pd.DataFrame(textoutputs)
        xlname = os.path.join(outputdir, descriptor + '.xlsx')
        df.to_excel(xlname)
        outputname = xlname
        return outputname

    #-------------------------------------------------------------------------------
    #-------------B-value regression with continuous covariate------------------------------
    # regression of Mrecord with continuous covariate
    # glm_fit
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']
    if outputtype == 'B_Regression':
        print('generating results for B_Regression...')
        descriptor = outputnametag + '_Breg'
        Mregression = np.zeros((nbeta,nbeta,3))
        for aa in range(nbeta):
            for bb in range(nbeta):
                m = Mrecord[aa,bb,g1]
                if np.var(m) > 0:
                    b, fit, R2, total_var, res_var = pysem.general_glm(m[np.newaxis,:], G)
                    Mregression[aa,bb,:] = [b[0,0],b[0,1],R2]

        Zthresh = stats.norm.ppf(1 - pthresh)
        # Z = arctanh(R)*sqrt(NP-1)
        Rthresh = np.tanh(Zthresh/np.sqrt(NP-1))
        R2thresh = Rthresh**2

        print('\n\nMconn regression with continuous covariate values')
        labeltext, inttext, slopetext, R2text, R2, R2thresh = write_Mreg_values(Mregression[:,:,0], Mregression[:,:,1], Mregression[:,:,2], betanamelist, rnamelist, beta_list, format='f', R2thresh=R2thresh, multiple_output = multiple_output)

        if len(labeltext) > 0:
            pthresh_list = ['{:.3e}'.format(pthresh)] * len(inttext)
            R2thresh_list = ['{:.3f}'.format(R2thresh)] * len(inttext)

            # sort output by magnitude of R2
            si = np.argsort(np.abs(R2))[::-1]
            R2 = np.array(R2)[si]
            textoutputs = {'regions': np.array(labeltext)[si], 'int': np.array(inttext)[si],
                           'slope': np.array(slopetext)[si],'R2': np.array(R2text)[si],
                           'R2 thresh': np.array(R2thresh_list)[si], 'p thresh': np.array(pthresh_list)[si]}
            # p, f = os.path.split(SAPMresultsname)
            df = pd.DataFrame(textoutputs)
            xlname = os.path.join(outputdir, descriptor + '.xlsx')
            df.to_excel(xlname)
        else:
            xlname = 'NA'
            print('Regression of B values with covariate ... no significant values found at p < {}'.format(pthresh))
        outputname = xlname

        print('finished generating results for B_Regression...')

        # testing other regression options
        target = []
        descriptor = outputnametag + '_BOLDstdev_vs_Covariate'
        regress_signal_features_with_cov(target, covariates[g1], Minput, Sinput_total[:, :, g1], fit_total[:, :, g1],
                                         Sconn_total[:, :, g1], beta_list, rnamelist, pthresh, outputdir, descriptor)
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

        regionnum = [rnamelist.index(target) ]   # input a region
        nametag = rnamelist[regionnum[0]] + '_' + outputnametag   # create name for saving figure

        if len(setylimits) > 0:
            ylim = setylimits[0]
            yrangethis = [-ylim,ylim]
        else:
            yrangethis = []
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
    # inputs to C6RD
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
                                beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis, TargetCanvas, multiple_output)

        outputname = plot_region_inputs_average(window, target, nametag1, Minput, Sinput_avg, Sinput_sem, fit_avg, fit_sem, Sconn_avg,
                                   Sconn_sem, beta_list, rnamelist, betanamelist, Mconn_avg, outputdir, yrangethis, TargetCanvas, multiple_output)

        print('finished generating outputs for Plot_SourceModel...')
        return outputname

    # Draw SAPM Diagram -----------------------------
    # outputoptions = ['B_Significance', 'B_Regression', 'Plot_BOLDModel', 'Plot_SourceModel', 'DrawSAPMdiagram']
    # if outputtype == 'DrawSAPMdiagram':
    #     print('getting ready to draw the SAPM network diagram ...')
    #     outputname = 'NA'
    #
    #   return outputname

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

    # v12 = np.array(pos2)-np.array(pos1)
    # d12 = np.sqrt(1/((v12[0]/ovd[0])**2 + (v12[1]/ovd[1])**2))
    # pe1b = pos1 + d12*v12

    # point on ellipse 1 on line from region 1 to region 0
    # v21 = np.array(pos1)-np.array(pos2)
    # d21 = np.sqrt(1/((v21[0]/ovd[0])**2 + (v21[1]/ovd[1])**2))
    # pe2 = pos2 + d21*v21

    # smooth arc line in region 1, betwen arrows for pos0-->pos1 and pos1-->pos2
    # line starts along vector v01 at point pe1a
    # line ends along vector v12 at point pe1b

    # angle of line along vector v01, wrt x axis
    angleA = (180/np.pi)*np.arctan2(v01[1],v01[0])
    angleA = np.round(angleA).astype(int)

    # angle of line along vector v12, wrt x axis
    # angleB = (180/np.pi)*np.arctan2(v12[1],v12[0])
    # angleB = np.round(angleB).astype(int)
    # anglediff = np.abs(angleB-angleA)

    # pe1ab_connectionstyle = "angle3,angleA={},angleB={}".format(angleA,angleB)

    # special case
    # specialcase = False
    # if np.abs(anglediff-180.0) < 1.0:
    #     specialcase = True
    #     pe1ab_connectionstyle = "arc3,rad=0"
    #     pe1ab_connectionstyle = "bar,fraction=0"
    #
    # if np.abs(anglediff) < 1.0:
    #     specialcase = False
    #     pe1ab_connectionstyle = "arc3,rad=0"

    # shift lines slightly to allow for reciprocal connections
    # offset = 0.007
    # dpos1 = np.array([offset*np.sin(angleA*np.pi/180.0), offset*np.cos(angleA*np.pi/180.0)])
    # dpos2 = np.array([offset*np.sin(angleB*np.pi/180.0), offset*np.cos(angleB*np.pi/180.0)])

    # pe0 += dpos1
    # pe1a += dpos1
    # pe1b += dpos2
    # pe2 += dpos2

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
    # offset = 0.007
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
    # offset = 0.007
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

    # # show axial slices?
    # if len(clusterdataname) > 0:
    #     for rr, regionname in enumerate(regionlist):
    #         clusternum = cnums[rr]
    #         outputimg = display_anatomical_slices(clusterdataname, regionname, clusternum, templatename)
    #         # display it somewhere...

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
        # val1 = statvals[nn]
        # m,s = parse_statval(val1)
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

                # pe0, pe1a, pe1b, pe2, pe1ab_connectionstyle, specialcase = points_on_ellipses2(p0, p1, p2, ovalsize, offset = 0)
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


def generate_null_data_set(regiondataname, covariatesname, npeople=0):
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
                tc_run = np.random.randn(tsize)
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
    p,f = os.path.split(covariatesname)
    cc = np.load(covariatesname, allow_pickle=True).flat[0]
    if npeople > 0:
        ncov,numcovpeople = np.shape(cc['GRPcharacteristicsvalues'])
        cc['GRPcharacteristicsvalues'] = np.random.randn(ncov,npeople)
    outputcovname = os.path.join(p, 'nulldata_' + f)
    np.save(outputcovname, cc)
    print('wrote null covariates data to {}'.format(outputcovname))

    return outputname, outputcovname


# def sort_SAPM_results(SAPMresults, vintrinsic_count, fintrinsic_count, latent_flag):
#     # order the combos to match the order in the first person
#     NP,nvariants = np.shape(SAPMresults)
#     results = copy.deepcopy(SAPMresults)
#     resultsr = copy.deepcopy(SAPMresults)
#     Nintrinsics = vintrinsic_count + fintrinsic_count
#     # ncombos = 2 ** Nintrinsics
#     ncombos = 2 ** vintrinsic_count
#
#     clist = []
#     for nn in range(vintrinsic_count):
#         cc = np.where(latent_flag == (1 + nn + fintrinsic_count))[0]
#         clist.append({'c': cc})
#
#     # figure out how to reorder the results from each participant to match the order in the 1st participant
#     search_size = 2 * np.ones(vintrinsic_count)
#     scalefactors = np.zeros((ncombos, vintrinsic_count))
#     for nn in range(ncombos):
#         scalefactor = 1.0 - 2.0 * ind2sub_ndims(search_size, nn)
#         scalefactors[nn, :] = scalefactor
#
#     b0 = np.array([results[0][x]['betavals'] for x in range(ncombos)])
#     order_record = []
#     for personindex in range(1, NP):
#         b1 = np.array([results[personindex][x]['betavals'] for x in range(ncombos)])
#
#         intsign = np.zeros(vintrinsic_count)
#         for nn in range(vintrinsic_count):
#             c = clist[nn]['c']
#             cc = np.corrcoef(b0[:, c].flatten(), b1[:, c].flatten())[0, 1]
#             intsign[nn] = np.sign(cc)
#
#         actualfactors = copy.deepcopy(scalefactors)
#         for nn in range(ncombos):
#             actualfactors[nn, :] *= intsign
#
#         order = np.zeros(ncombos).astype(int)
#         for nn in range(ncombos):
#             # ref = actualfactors[nn, :]
#             ref = scalefactors[nn, :]
#             check = np.zeros(ncombos)
#             for nn2 in range(ncombos):
#                 # check[nn2] = (ref == scalefactors[nn2, :]).all()
#                 check[nn2] = (ref == actualfactors[nn2, :]).all()
#             x = np.where(check)[0]
#             order[nn] = x
#
#         for nn in range(ncombos):
#             resultsr[personindex][nn] = copy.deepcopy(results[personindex][order[nn]])
#
#         order_record += [order]
#
#     return resultsr
#
#
# # testing sorting methods-----------------------------------------
# def sort_SAPM_results2(SAPMresults, vintrinsic_count, fintrinsic_count, latent_flag):
#     NP,ncombos = np.shape(SAPMresults)
#
#     for person in range(NP):
#         for v in range(ncombos):
#             Meigv = SAPMresults[person][v]['Meigv']
#             if v == 0:
#                 Mset = copy.deepcopy(Meigv[:,:,np.newaxis])
#             else:
#                 Mset = np.concatenate((Mset,Meigv[:,:,np.newaxis]),axis=2)
#         if person == 0:
#             Meigv_total = copy.deepcopy(Mset[:,:,:,np.newaxis])
#         else:
#             Meigv_total = np.concatenate((Meigv_total,Mset[:,:,:,np.newaxis]),axis=3)
#
#     # result is ncon x nlatents x ncombos x NP
#     # need to sort the combos to be the closest matches
#
#     person0 = 0
#     order_list = []
#     for person1 in range(NP):
#         Rgrid = np.zeros((ncombos,ncombos))
#         order = np.zeros(ncombos)
#         for nl1 in range(ncombos):
#             for nl2 in range(ncombos):
#                 m1 = Meigv_total[:,:,nl1,person0]
#                 m2 = Meigv_total[:,:,nl2,person1]
#                 R = np.corrcoef(m1.flatten(),m2.flatten())[0,1]
#                 Rgrid[nl1,nl2] = R
#             x = np.argmax(Rgrid[nl1,:])
#             order[nl1] = x
#
#         if person1 == 0:
#             order_list = np.array(order[:,np.newaxis])
#         else:
#             order_list = np.concatenate((order_list,order[:,np.newaxis]),axis=1)
#     order_list = order_list.astype(int)
#
#     # apply the sorting
#     SAPMresultsr = copy.deepcopy(SAPMresults)
#     for person1 in range(NP):
#         for nl1 in range(ncombos):
#             SAPMresultsr[person1][nl1] = copy.deepcopy(SAPMresults[person1][order_list[nl1,person1]])
#
#     return SAPMresultsr   #, order_list
#
#
#
# def sort_SAPM_results_betavals(SAPMresults, vintrinsic_count, fintrinsic_count, latent_flag):
#     # order the combos to match the order in the first person
#     NP,nvariants = np.shape(SAPMresults)
#     results = copy.deepcopy(SAPMresults)
#     resultsr = copy.deepcopy(SAPMresults)
#     ncombos = 2 ** vintrinsic_count
#
#     clist = []
#     for nn in range(vintrinsic_count):
#         cc = np.where(latent_flag == (fintrinsic_count + 1 + nn))[0]
#         clist.append({'c': cc})
#
#     # figure out how to reorder the results from each participant to match the order in the 1st participant
#     search_size = 2 * np.ones(vintrinsic_count)
#     scalefactors = np.zeros((ncombos, vintrinsic_count))
#     for nn in range(ncombos):
#         scalefactor = 1.0 - 2.0 * ind2sub_ndims(search_size, nn)
#         scalefactors[nn, :] = scalefactor
#
#     b0 = np.array([results[0][x]['betavals'] for x in range(ncombos)])
#     for personindex in range(1, NP):
#         b1 = np.array([results[personindex][x]['betavals'] for x in range(ncombos)])
#
#         intsign = np.zeros(vintrinsic_count)
#         for nn in range(vintrinsic_count):
#             c = clist[nn]['c']
#             cc = np.corrcoef(b0[0, c], b1[0, c])[0, 1]
#             intsign[nn] = np.sign(cc)
#
#         actualfactors = copy.deepcopy(scalefactors)
#         for nn in range(ncombos):
#             actualfactors[nn, :] *= intsign
#
#         order = np.zeros(ncombos).astype(int)
#         for nn in range(ncombos):
#             ref = actualfactors[nn, :]
#             check = np.zeros(ncombos)
#             for nn2 in range(ncombos):
#                 check[nn2] = (ref == scalefactors[nn2, :]).all()
#             x = np.where(check)[0]
#             order[nn] = x
#
#         for nn in range(ncombos):
#             resultsr[personindex][nn] = copy.deepcopy(results[personindex][order[nn]])
#
#     return resultsr
#
#
# def compare_order_two_datasets(resultsname1, resultsname2, paramsname):
#     results1 = np.load(resultsname1,allow_pickle=True)
#     results2 = np.load(resultsname2,allow_pickle=True)
#
#     SAPMparams = np.load(paramsname, allow_pickle=True).flat[0]
#     fintrinsic_count = SAPMparams['fintrinsic_count']
#     vintrinsic_count = SAPMparams['vintrinsic_count']
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     latent_flag = SAPMparams['latent_flag']
#
#     # compare the data sets -----------------------------------------
#     Nintrinsics = vintrinsic_count + fintrinsic_count
#     # ncombos = 2 ** Nintrinsics
#     ncombos = 2 ** vintrinsic_count
#
#     clist = []
#     for nn in range(vintrinsic_count):
#         cc = np.where(latent_flag == (1 + nn + fintrinsic_count))[0]
#         clist.append({'c': cc})
#
#     # figure out how to reorder the results from each participant to match the order in the 1st participant
#     search_size = 2 * np.ones(vintrinsic_count)
#     scalefactors = np.zeros((ncombos, vintrinsic_count))
#     for nn in range(ncombos):
#         scalefactor = 1.0 - 2.0 * ind2sub_ndims(search_size, nn)
#         scalefactors[nn, :] = scalefactor
#
#     NP1, nvar1 = np.shape(results1)
#     NP2, nvar2 = np.shape(results2)
#     orderlist = []
#     personindex1 = 0
#     b0 = np.array([results1[personindex1][x]['betavals'] for x in range(ncombos)])
#     for personindex2 in range(NP2):
#         b1 = np.array([results2[personindex2][x]['betavals'] for x in range(ncombos)])
#
#         intsign = np.zeros(vintrinsic_count)
#         for nn in range(vintrinsic_count):
#             c = clist[nn]['c']
#             cc = np.corrcoef(b0[:, c].flatten(), b1[:, c].flatten())[0, 1]
#             intsign[nn] = np.sign(cc)
#
#         actualfactors = copy.deepcopy(scalefactors)
#         for nn in range(ncombos):
#             actualfactors[nn, :] *= intsign
#         # actualfactors *= intsign
#
#         order = np.zeros(ncombos).astype(int)
#         for nn in range(ncombos):
#             # ref = actualfactors[nn, :]
#             ref = scalefactors[nn, :]
#             check = np.zeros(ncombos)
#             for nn2 in range(ncombos):
#                 # check[nn2] = (ref == scalefactors[nn2, :]).all()
#                 check[nn2] = (ref == actualfactors[nn2, :]).all()
#             x = np.where(check)[0]
#             order[nn] = x
#
#         orderlist += [order]
#
#     orderlist = np.array(orderlist)
#     order_mean = np.mean(orderlist,axis=0)
#     order_std = np.std(orderlist,axis=0)
#     print('order of data set 2 compared to 1 is {} {} {}'.format(order_mean,chr(177),order_std))
#     return order_mean, order_std, orderlist
#
#
#
# def compare_order_two_datasets2(resultsname1, resultsname2, paramsname):
#     results1 = np.load(resultsname1,allow_pickle=True)
#     results2 = np.load(resultsname2,allow_pickle=True)
#
#     SAPMparams = np.load(paramsname, allow_pickle=True).flat[0]
#     fintrinsic_count = SAPMparams['fintrinsic_count']
#     vintrinsic_count = SAPMparams['vintrinsic_count']
#     ctarget = SAPMparams['ctarget']
#     csource = SAPMparams['csource']
#     latent_flag = SAPMparams['latent_flag']
#
#     # compare the data sets -----------------------------------------
#     Nintrinsics = vintrinsic_count + fintrinsic_count
#     # ncombos = 2 ** Nintrinsics
#     ncombos = 2 ** vintrinsic_count
#
#     # organize the data---------------
#     NP1,ncombos = np.shape(results1)
#     for person in range(NP1):
#         for v in range(ncombos):
#             Meigv = results1[person][v]['Meigv']
#             if v == 0:
#                 Mset = copy.deepcopy(Meigv[:,:,np.newaxis])
#             else:
#                 Mset = np.concatenate((Mset,Meigv[:,:,np.newaxis]),axis=2)
#         if person == 0:
#             Meigv_total1 = copy.deepcopy(Mset[:,:,:,np.newaxis])
#         else:
#             Meigv_total1 = np.concatenate((Meigv_total1,Mset[:,:,:,np.newaxis]),axis=3)
#
#
#     # organize the data---------------
#     NP2,ncombos = np.shape(results2)
#     for person in range(NP2):
#         for v in range(ncombos):
#             Meigv = results2[person][v]['Meigv']
#             if v == 0:
#                 Mset = copy.deepcopy(Meigv[:,:,np.newaxis])
#             else:
#                 Mset = np.concatenate((Mset,Meigv[:,:,np.newaxis]),axis=2)
#         if person == 0:
#             Meigv_total2 = copy.deepcopy(Mset[:,:,:,np.newaxis])
#         else:
#             Meigv_total2 = np.concatenate((Meigv_total2,Mset[:,:,:,np.newaxis]),axis=3)
#
#     # need to sort the combos to be the closest matches
#
#     big_order_list = []
#     for person1 in range(NP1):
#         order_list = []
#         for person2 in range(NP2):
#             Rgrid = np.zeros((ncombos,ncombos))
#             order = np.zeros(ncombos)
#             for nl1 in range(ncombos):
#                 for nl2 in range(ncombos):
#                     m1 = Meigv_total1[:,:,nl1,person1]
#                     m2 = Meigv_total2[:,:,nl2,person2]
#                     R = np.corrcoef(m1.flatten(),m2.flatten())[0,1]
#                     Rgrid[nl1,nl2] = R
#                 x = np.argmax(Rgrid[nl1,:])
#                 order[nl1] = x
#
#             if person2 == 0:
#                 order_list = np.array(order[:,np.newaxis])
#             else:
#                 order_list = np.concatenate((order_list,order[:,np.newaxis]),axis=1)
#         order_list = np.array(order_list).astype(int)
#
#         if person1 == 0:
#             big_order_list = np.array(order_list[:,:, np.newaxis])
#         else:
#             big_order_list = np.concatenate((big_order_list, order_list[:,:, np.newaxis]), axis=2)
#
#     orderm = [np.mean(big_order_list[x,:,:]) for x in range(ncombos)]
#     ordersd = [np.std(big_order_list[x,:,:]) for x in range(ncombos)]
#
#     final_order = np.argsort(orderm)
#
#     # apply the sorting
#     results2r = copy.deepcopy(results2)
#     for person2 in range(NP):
#         for nl2 in range(ncombos):
#             results2r[person2][nl2] = copy.deepcopy(results2[person2][final_order[nl2]])
#
#     return results2r, final_order


def check_network_for_defects(networkfile, clusterdataname):
    network, nclusterlist, sem_region_list, fintrinsic_count, vintrinsic_count = load_network_model_w_intrinsics(
        networkfile)

    cluster_data = np.load(clusterdataname, allow_pickle=True).flat[0]
    cluster_properties = load_filtered_cluster_properties(clusterdataname, networkfile)
    nregions = len(cluster_properties)
    nclusterlist = [cluster_properties[i]['nclusters'] for i in range(nregions)]
    rnamelist = [cluster_properties[i]['rname'] for i in range(nregions)]

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

    Mconn_test = copy.deepcopy(Mconn)
    Mconn_test[ctarget,csource] = 1.0
    check_det = np.linalg.det(Mconn_test)
    if check_det < 1.0e-6:
        print('connectivity matrix is not invertible ...')
        print('network might not have a single distinct result')
        print('Mconn matrix size is {} x {}, and has rank {}'.format(np.shape(Mconn)[0],np.shape(Mconn)[1], np.linalg.matrix_rank(Mconn)))
    else:
        print('connectivity matrix could be invertible ...')
        print('network might have a single distinct result')

    return Mconn, ctarget, csource





#
#
# if __name__ == '__main__':
#     main()
#
